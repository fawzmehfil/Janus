"""Monte Carlo full-information action oracle using common random numbers."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import fmean, stdev
from typing import Literal

from janus.core.clearing import ClearingPriceRule
from janus.core.config import AccountingConfig, MarketConfig
from janus.core.randomness import DisturbanceTape
from janus.core.state import MarketState
from janus.interventions.actions import ActionID, InterventionConfig
from janus.scenarios.families import ScenarioConfig
from janus.simulation.runner import DynamicsConfig, run_recovery

OracleObjective = Literal[
    "primary",
    "consumer_protection",
    "producer_continuity",
    "stability_aware",
    "fiscal_neutral",
]


@dataclass(frozen=True, slots=True)
class ActionValue:
    action: ActionID
    samples: tuple[float, ...]
    mean: float
    standard_error: float


@dataclass(frozen=True, slots=True)
class OracleResult:
    action_values: tuple[ActionValue, ...]
    best_action: ActionID
    epsilon_optimal_actions: frozenset[ActionID]
    epsilon_by_action: tuple[tuple[ActionID, float], ...]
    rollout_count: int

    def value_for(self, action: ActionID) -> ActionValue:
        return next(value for value in self.action_values if value.action is action)


def _standard_error(values: tuple[float, ...]) -> float:
    return 0.0 if len(values) < 2 else stdev(values) / sqrt(len(values))


def evaluate_oracle(
    state: MarketState,
    scenario: ScenarioConfig,
    *,
    root_seed: int,
    rollout_count: int = 256,
    reference_surplus_per_period: float = 840.0,
    market_config: MarketConfig | None = None,
    accounting_config: AccountingConfig | None = None,
    intervention_config: InterventionConfig | None = None,
    dynamics: DynamicsConfig | None = None,
    objective: OracleObjective = "primary",
    price_rule: ClearingPriceRule = "midpoint",
) -> OracleResult:
    if rollout_count < 1:
        raise ValueError("rollout count must be positive")
    if objective not in (
        "primary",
        "consumer_protection",
        "producer_continuity",
        "stability_aware",
        "fiscal_neutral",
    ):
        raise ValueError(f"unknown oracle objective: {objective}")
    dynamics = dynamics or DynamicsConfig()
    samples: dict[ActionID, list[float]] = {action: [] for action in ActionID}

    for rollout in range(rollout_count):
        tape = DisturbanceTape(root_seed, f"{scenario.scenario_id}:future:{rollout}")
        for action in ActionID:
            result = run_recovery(
                state,
                scenario,
                action,
                tape,
                market_config=market_config,
                accounting_config=accounting_config,
                intervention_config=intervention_config,
                dynamics=dynamics,
                price_rule=price_rule,
            )
            samples[action].append(getattr(result.utilities, objective))

    action_values = tuple(
        ActionValue(
            action=action,
            samples=tuple(samples[action]),
            mean=fmean(samples[action]),
            standard_error=_standard_error(tuple(samples[action])),
        )
        for action in ActionID
    )
    best_value = max(action_values, key=lambda value: (value.mean, value.action.value))
    # The specification defines the analytic reference as attainable *per-period*
    # surplus (840), so the 0.5% absolute tolerance is 4.2 rather than a
    # horizon-scaled 84. Paired Monte Carlo uncertainty still scales naturally
    # through the standard error of each action-value difference.
    absolute_floor = 0.005 * reference_surplus_per_period
    epsilon_pairs: list[tuple[ActionID, float]] = []
    optimal: set[ActionID] = set()
    for value in action_values:
        paired_differences = tuple(
            best - candidate
            for best, candidate in zip(best_value.samples, value.samples, strict=True)
        )
        epsilon = max(absolute_floor, 2 * _standard_error(paired_differences))
        epsilon_pairs.append((value.action, epsilon))
        if best_value.mean - value.mean <= epsilon:
            optimal.add(value.action)

    return OracleResult(
        action_values=action_values,
        best_action=best_value.action,
        epsilon_optimal_actions=frozenset(optimal),
        epsilon_by_action=tuple(epsilon_pairs),
        rollout_count=rollout_count,
    )


@dataclass(frozen=True, slots=True)
class ConvergenceResult:
    rollout_counts: tuple[int, ...]
    best_actions: tuple[ActionID, ...]
    stable: bool
    maximum_mean_delta: float


def check_oracle_convergence(
    state: MarketState,
    scenario: ScenarioConfig,
    *,
    root_seed: int,
    base_rollouts: int,
    dynamics: DynamicsConfig | None = None,
) -> ConvergenceResult:
    counts = (base_rollouts, base_rollouts * 2, base_rollouts * 4)
    results = tuple(
        evaluate_oracle(
            state,
            scenario,
            root_seed=root_seed,
            rollout_count=count,
            dynamics=dynamics,
        )
        for count in counts
    )
    largest = results[-1]
    maximum_delta = max(
        abs(result.value_for(action).mean - largest.value_for(action).mean)
        for result in results[:-1]
        for action in ActionID
    )
    best_actions = tuple(result.best_action for result in results)
    return ConvergenceResult(
        rollout_counts=counts,
        best_actions=best_actions,
        stable=len(set(best_actions)) == 1,
        maximum_mean_delta=maximum_delta,
    )
