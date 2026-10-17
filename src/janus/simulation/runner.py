"""Deterministic recovery simulation for a fixed state, action, and future tape."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import log

from pydantic import Field

from janus.core.config import AccountingConfig, MarketConfig, StrictFrozenModel
from janus.core.engine import PeriodResult, run_period
from janus.core.ledger import PolicyFlows
from janus.core.randomness import DisturbanceTape
from janus.core.state import MarketState
from janus.evaluation.observations import realized_log_volatility
from janus.interventions.actions import (
    ActionID,
    InterventionConfig,
    ResolvedIntervention,
    resolve_intervention,
)
from janus.participants.buyers import generate_bids
from janus.participants.sellers import generate_asks
from janus.scenarios.families import MarketPrimitives, ScenarioConfig, primitives_for_period


class DynamicsConfig(StrictFrozenModel):
    recovery_periods: int = Field(default=20, ge=1)
    value_noise_fraction: float = Field(default=0.015, ge=0)
    cost_noise_fraction: float = Field(default=0.015, ge=0)
    consumer_surplus_weight: float = Field(default=2.0, ge=0)
    producer_surplus_weight: float = Field(default=1.0, ge=0)
    stability_penalty_weight: float = Field(default=100.0, ge=0)
    producer_exit_cost: float = Field(default=50.0, ge=0)


@dataclass(frozen=True, slots=True)
class ExogenousSnapshot:
    period: int
    buyer_value_noise: tuple[tuple[str, float], ...]
    seller_cost_noise: tuple[tuple[str, tuple[float, ...]], ...]
    tie_priorities: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class RecoveryStep:
    period_result: PeriodResult
    exogenous: ExogenousSnapshot


@dataclass(frozen=True, slots=True)
class UtilityVector:
    primary: float
    consumer_protection: float
    producer_continuity: float
    stability_aware: float
    fiscal_neutral: float


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    action: ActionID
    resolved_intervention: ResolvedIntervention
    initial_state: MarketState
    final_state: MarketState
    steps: tuple[RecoveryStep, ...]
    utilities: UtilityVector


def _recent_returns(prices: list[float | None], window: int = 3) -> tuple[float, ...]:
    observed = [price for price in prices if price is not None]
    if len(observed) < 2:
        return ()
    values = observed[-(window + 1) :]
    return tuple(log(right / left) for left, right in pairwise(values))


def _with_noise(
    primitives: MarketPrimitives,
    *,
    state: MarketState,
    tape: DisturbanceTape,
    dynamics: DynamicsConfig,
    reference_price: float,
) -> tuple[MarketPrimitives, ExogenousSnapshot]:
    value_std = dynamics.value_noise_fraction * reference_price
    cost_std = dynamics.cost_noise_fraction * reference_price
    value_noise = {
        buyer_id: tape.normal("value", state.period, buyer_id, mean=0, std=value_std)
        for buyer_id in primitives.buyer_true_values
    }
    cost_noise = {
        seller_id: tuple(
            tape.normal("cost", state.period, seller_id, unit, mean=0, std=cost_std)
            for unit in range(len(costs))
        )
        for seller_id, costs in primitives.seller_marginal_costs.items()
    }
    noisy_values = {
        buyer_id: max(0.0, value + value_noise[buyer_id])
        for buyer_id, value in primitives.buyer_true_values.items()
    }
    noisy_costs = {
        seller_id: tuple(
            max(0.0, cost + cost_noise[seller_id][unit]) for unit, cost in enumerate(costs)
        )
        for seller_id, costs in primitives.seller_marginal_costs.items()
    }
    noisy = MarketPrimitives(
        buyer_true_values=noisy_values,
        buyer_bid_adjustments=dict(primitives.buyer_bid_adjustments),
        seller_marginal_costs=noisy_costs,
        seller_offer_limits=dict(primitives.seller_offer_limits),
        seller_markups=dict(primitives.seller_markups),
        colluding_seller_ids=primitives.colluding_seller_ids,
        momentum_buyer_ids=primitives.momentum_buyer_ids,
    )
    snapshot = ExogenousSnapshot(
        period=state.period,
        buyer_value_noise=tuple(sorted(value_noise.items())),
        seller_cost_noise=tuple(sorted(cost_noise.items())),
        tie_priorities=(),
    )
    return noisy, snapshot


def _flows_for_step(intervention: ResolvedIntervention, step: int) -> PolicyFlows:
    flows = intervention.policy_flows
    if intervention.action is not ActionID.COLLUSION_INVESTIGATION or step == 0:
        return flows
    return PolicyFlows()


def _utility_vector(steps: tuple[RecoveryStep, ...], dynamics: DynamicsConfig) -> UtilityVector:
    primary = sum(step.period_result.ledger.net_social_surplus for step in steps)
    consumer_protection = sum(
        dynamics.consumer_surplus_weight * ledger.consumer_surplus
        + dynamics.producer_surplus_weight * ledger.producer_surplus
        + ledger.government_balance
        - ledger.real_cost
        - ledger.financing_cost
        for ledger in (step.period_result.ledger for step in steps)
    )
    exits = sum(not seller.active for seller in steps[-1].period_result.next_state.sellers)
    producer_continuity = primary - dynamics.producer_exit_cost * exits
    prices = [step.period_result.clearing.transaction_price for step in steps]
    stability_aware = primary - dynamics.stability_penalty_weight * realized_log_volatility(
        prices, window=len(prices)
    )
    fiscal_neutral = sum(
        ledger.consumer_surplus
        + ledger.producer_surplus
        + ledger.government_balance
        - ledger.real_cost
        for ledger in (step.period_result.ledger for step in steps)
    )
    return UtilityVector(
        primary=primary,
        consumer_protection=consumer_protection,
        producer_continuity=producer_continuity,
        stability_aware=stability_aware,
        fiscal_neutral=fiscal_neutral,
    )


def run_recovery(
    initial_state: MarketState,
    scenario: ScenarioConfig,
    action: ActionID,
    tape: DisturbanceTape,
    *,
    market_config: MarketConfig | None = None,
    accounting_config: AccountingConfig | None = None,
    intervention_config: InterventionConfig | None = None,
    dynamics: DynamicsConfig | None = None,
) -> RecoveryResult:
    market_config = market_config or MarketConfig()
    accounting_config = accounting_config or AccountingConfig()
    intervention_config = intervention_config or InterventionConfig()
    dynamics = dynamics or DynamicsConfig()

    decision_primitives = primitives_for_period(initial_state, scenario)
    intervention = resolve_intervention(
        action,
        reference_price=scenario.reference_price,
        scenario=scenario,
        colluding_seller_ids=decision_primitives.colluding_seller_ids,
        tape=tape,
        decision_period=initial_state.period,
        config=intervention_config,
    )
    state = initial_state.clone()
    prices: list[float | None] = [state.last_price]
    steps: list[RecoveryStep] = []

    for step_index in range(dynamics.recovery_periods):
        primitives = primitives_for_period(
            state,
            scenario,
            recent_log_returns=_recent_returns(prices),
            collusion_disabled=intervention.collusion_disabled,
        )
        primitives, snapshot = _with_noise(
            primitives,
            state=state,
            tape=tape,
            dynamics=dynamics,
            reference_price=scenario.reference_price,
        )
        flows = _flows_for_step(intervention, step_index)
        bids = generate_bids(state, primitives, buyer_tax_rate=flows.buyer_tax_rate)
        asks = generate_asks(
            state,
            primitives,
            seller_subsidy_per_unit=flows.seller_subsidy_per_unit,
        )
        order_ids = tuple(bid.order_id for bid in bids) + tuple(ask.order_id for ask in asks)
        priority = tape.tie_priority(order_ids, state.period)
        snapshot = ExogenousSnapshot(
            snapshot.period,
            snapshot.buyer_value_noise,
            snapshot.seller_cost_noise,
            tuple(sorted(priority.items())),
        )
        result = run_period(
            state,
            bids,
            asks,
            tie_priority=priority,
            market_config=market_config,
            accounting_config=accounting_config,
            policy=flows,
            price_cap=intervention.price_cap,
        )
        steps.append(RecoveryStep(result, snapshot))
        prices.append(result.clearing.transaction_price)
        state = result.next_state

    frozen_steps = tuple(steps)
    return RecoveryResult(
        action=action,
        resolved_intervention=intervention,
        initial_state=initial_state,
        final_state=state,
        steps=frozen_steps,
        utilities=_utility_vector(frozen_steps, dynamics),
    )
