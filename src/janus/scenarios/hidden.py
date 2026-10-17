"""Predeclared one-dimension severity shifts for hidden OOD candidates."""

from __future__ import annotations

import numpy as np

from janus.core.randomness import NamedRNG
from janus.scenarios.families import LatentFamily, ScenarioConfig
from janus.scenarios.generator import (
    CandidateGeneratorConfig,
    CandidateWorld,
    simulate_candidate,
)


def _strata(count: int, generator: np.random.Generator) -> tuple[float, ...]:
    values = (np.arange(count, dtype=float) + generator.random(count)) / count
    generator.shuffle(values)
    return tuple(float(value) for value in values)


def _scale(value: float, low: float, high: float) -> float:
    return low + value * (high - low)


def generate_ood_scenario_configs(
    split: str,
    family: LatentFamily,
    config: CandidateGeneratorConfig,
) -> tuple[ScenarioConfig, ...]:
    """Shift exactly one family-defining severity dimension beyond the ID range."""

    rng = NamedRNG(config.root_seed).generator("scenario_parameters", split, family.value, "ood")
    dimensions = [_strata(config.candidates_per_family, rng) for _ in range(6)]
    scenarios = []
    for index in range(config.candidates_per_family):
        u = [dimension[index] for dimension in dimensions]
        scenario_id = f"{split}-{family.value}-{index:04d}"
        duration = round(_scale(u[0], 18, 20))
        if family is LatentFamily.SUPPLY_SHORTAGE:
            scenario = ScenarioConfig(
                scenario_id=scenario_id,
                family=family,
                shock_start=config.shock_start,
                shock_duration=duration,
                supply_cost_shift_fraction=_scale(u[1], 0.40, 0.55),
                supply_capacity_loss_fraction=_scale(u[2], 0.10, 0.40),
                affected_seller_fraction=_scale(u[3], 0.30, 0.80),
            )
        elif family is LatentFamily.DEMAND_INCREASE:
            scenario = ScenarioConfig(
                scenario_id=scenario_id,
                family=family,
                shock_start=config.shock_start,
                shock_duration=duration,
                demand_value_shift_fraction=_scale(u[1], 0.40, 0.55),
                affected_buyer_fraction=_scale(u[2], 0.30, 0.80),
            )
        elif family is LatentFamily.SELLER_COLLUSION:
            scenario = ScenarioConfig(
                scenario_id=scenario_id,
                family=family,
                shock_start=config.shock_start,
                shock_duration=duration,
                colluding_seller_fraction=_scale(u[1], 0.30, 0.80),
                collusive_markup_fraction=_scale(u[2], 0.40, 0.55),
                collusive_withholding_fraction=_scale(u[3], 0.0, 0.40),
            )
        else:
            scenario = ScenarioConfig(
                scenario_id=scenario_id,
                family=family,
                shock_start=config.shock_start,
                shock_duration=duration,
                momentum_buyer_fraction=_scale(u[1], 0.30, 0.50),
                momentum_feedback=_scale(u[2], 50.0, 70.0),
                momentum_initial_impulse_fraction=_scale(u[3], 0.10, 0.30),
                momentum_distortion_cap_fraction=_scale(u[4], 0.20, 0.50),
                momentum_decay=_scale(u[5], 0.70, 0.95),
            )
        scenarios.append(scenario)
    return tuple(scenarios)


def generate_ood_candidates(
    split: str,
    config: CandidateGeneratorConfig,
) -> tuple[CandidateWorld, ...]:
    worlds: list[CandidateWorld] = []
    for family in LatentFamily:
        worlds.extend(
            simulate_candidate(split, scenario, config)
            for scenario in generate_ood_scenario_configs(split, family, config)
        )
    return tuple(worlds)
