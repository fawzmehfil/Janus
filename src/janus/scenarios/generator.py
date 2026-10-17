"""Deterministic stratified candidate generation and public-history simulation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import log

import numpy as np
from pydantic import Field, model_validator

from janus.core.config import MarketConfig, StrictFrozenModel
from janus.core.fixtures import analytic_market
from janus.core.randomness import DisturbanceTape, NamedRNG
from janus.core.state import MarketState
from janus.evaluation.observations import (
    PublicObservationHistory,
    build_public_row,
    serialize_public_history,
)
from janus.interventions.actions import ActionID
from janus.scenarios.families import LatentFamily, ScenarioConfig
from janus.simulation.runner import DynamicsConfig, RecoveryResult, run_recovery


class CandidateGeneratorConfig(StrictFrozenModel):
    candidates_per_family: int = Field(default=80, ge=1)
    shock_start: int = Field(default=20, ge=1)
    observation_periods: int = Field(default=10, ge=1)
    shock_duration_min: int = Field(default=18, ge=1)
    shock_duration_max: int = Field(default=20, ge=1)
    initial_cash_multiple: float = Field(default=100.0, gt=0)
    root_seed: int = Field(default=20261017, ge=0)

    @model_validator(mode="after")
    def validate_duration_range(self) -> CandidateGeneratorConfig:
        if self.shock_duration_min > self.shock_duration_max:
            raise ValueError("shock_duration_min cannot exceed shock_duration_max")
        return self


@dataclass(frozen=True, slots=True)
class CandidateWorld:
    world_id: str
    split: str
    family: LatentFamily
    scenario: ScenarioConfig
    decision_state: MarketState
    observation: PublicObservationHistory
    serialized_observation: bytes
    summary_features: tuple[float, ...]


def _stratified_unit_values(count: int, generator: np.random.Generator) -> tuple[float, ...]:
    values = (np.arange(count, dtype=float) + generator.random(count)) / count
    generator.shuffle(values)
    return tuple(float(value) for value in values)


def _scale(unit: float, low: float, high: float) -> float:
    return low + unit * (high - low)


def generate_scenario_configs(
    split: str,
    family: LatentFamily,
    config: CandidateGeneratorConfig,
) -> tuple[ScenarioConfig, ...]:
    generator = NamedRNG(config.root_seed).generator("scenario_parameters", split, family.value)
    dimensions = [
        _stratified_unit_values(config.candidates_per_family, generator) for _ in range(6)
    ]
    scenarios: list[ScenarioConfig] = []
    for index in range(config.candidates_per_family):
        u = [dimension[index] for dimension in dimensions]
        duration = round(_scale(u[0], config.shock_duration_min, config.shock_duration_max))
        scenario_id = f"{split}-{family.value}-{index:04d}"
        if family is LatentFamily.SUPPLY_SHORTAGE:
            scenario = ScenarioConfig(
                scenario_id=scenario_id,
                family=family,
                shock_start=config.shock_start,
                shock_duration=duration,
                supply_cost_shift_fraction=_scale(u[1], 0.10, 0.40),
                supply_capacity_loss_fraction=_scale(u[2], 0.10, 0.40),
                affected_seller_fraction=_scale(u[3], 0.30, 0.80),
            )
        elif family is LatentFamily.DEMAND_INCREASE:
            scenario = ScenarioConfig(
                scenario_id=scenario_id,
                family=family,
                shock_start=config.shock_start,
                shock_duration=duration,
                demand_value_shift_fraction=_scale(u[1], 0.10, 0.40),
                affected_buyer_fraction=_scale(u[2], 0.30, 0.80),
            )
        elif family is LatentFamily.SELLER_COLLUSION:
            scenario = ScenarioConfig(
                scenario_id=scenario_id,
                family=family,
                shock_start=config.shock_start,
                shock_duration=duration,
                colluding_seller_fraction=_scale(u[1], 0.30, 0.80),
                collusive_markup_fraction=_scale(u[2], 0.10, 0.40),
                collusive_withholding_fraction=_scale(u[3], 0.0, 0.40),
            )
        else:
            scenario = ScenarioConfig(
                scenario_id=scenario_id,
                family=family,
                shock_start=config.shock_start,
                shock_duration=duration,
                momentum_buyer_fraction=_scale(u[1], 0.30, 0.50),
                momentum_feedback=_scale(u[2], 10.0, 50.0),
                momentum_initial_impulse_fraction=_scale(u[3], 0.10, 0.30),
                momentum_distortion_cap_fraction=_scale(u[4], 0.20, 0.50),
                momentum_decay=_scale(u[5], 0.70, 0.95),
            )
        scenarios.append(scenario)
    return tuple(scenarios)


def observation_from_history(
    history: RecoveryResult,
    *,
    reference_price: float = 100.0,
    reference_volume: float = 21.0,
    reference_inventory: float = 40.0,
) -> PublicObservationHistory:
    prices: list[float | None] = []
    rows = []
    for step in history.steps:
        result = step.period_result
        prices.append(result.clearing.transaction_price)
        rows.append(
            build_public_row(
                result.previous_state,
                result.clearing,
                price_history=prices,
                intervention=ActionID.DO_NOTHING,
                reference_price=reference_price,
                reference_volume=reference_volume,
                reference_inventory=reference_inventory,
            )
        )
    return PublicObservationHistory(rows=tuple(rows))


def public_summary(history: PublicObservationHistory) -> tuple[float, ...]:
    """Predeclared matching vector derived only from rounded public fields."""

    post_shock = history.rows[-10:]
    features: list[float] = []
    normalized_prices = [
        1.0 if row.transaction_price is None else row.transaction_price / row.reference_price
        for row in post_shock
    ]
    returns = [0.0]
    returns.extend(log(right / left) for left, right in pairwise(normalized_prices))
    for row, price_ratio, realized_return in zip(
        post_shock, normalized_prices, returns, strict=True
    ):
        features.extend(
            (
                price_ratio,
                realized_return,
                row.transaction_volume / row.reference_volume,
                row.aggregate_available_supply / row.reference_inventory,
                row.unfilled_submitted_demand / max(1.0, row.reference_volume),
                row.aggregate_inventory / row.reference_inventory,
                0.0 if row.spread is None else row.spread / row.reference_price,
            )
        )
    final = post_shock[-1]
    features.extend(
        (
            final.realized_log_price_volatility,
            final.seller_hhi,
            float(final.active_firm_count),
            float(final.bankruptcies),
        )
    )
    return tuple(features)


def simulate_candidate(
    split: str,
    scenario: ScenarioConfig,
    generator_config: CandidateGeneratorConfig,
) -> CandidateWorld:
    market_config = MarketConfig(initial_cash_multiple=generator_config.initial_cash_multiple)
    state = analytic_market(market_config)
    horizon = generator_config.shock_start + generator_config.observation_periods
    history = run_recovery(
        state,
        scenario,
        ActionID.DO_NOTHING,
        DisturbanceTape(generator_config.root_seed, f"history:{scenario.scenario_id}"),
        market_config=market_config,
        dynamics=DynamicsConfig(recovery_periods=horizon),
    )
    observation = observation_from_history(history)
    serialized = serialize_public_history(observation)
    return CandidateWorld(
        world_id=scenario.scenario_id,
        split=split,
        family=scenario.family,
        scenario=scenario,
        decision_state=history.final_state,
        observation=observation,
        serialized_observation=serialized,
        summary_features=public_summary(observation),
    )


def generate_candidates(split: str, config: CandidateGeneratorConfig) -> tuple[CandidateWorld, ...]:
    worlds: list[CandidateWorld] = []
    for family in LatentFamily:
        scenarios = generate_scenario_configs(split, family, config)
        worlds.extend(simulate_candidate(split, scenario, config) for scenario in scenarios)
    return tuple(worlds)
