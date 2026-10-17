"""Frozen Phase 0D validity gates and bundle-clustered uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import stdev

import numpy as np

from janus.evaluation.calibration import (
    BaselineAudit,
    OverlapAudit,
    action_value_matrix,
    audit_baselines,
    audit_overlap,
    tune_public_heuristic,
)
from janus.evaluation.statistics import (
    BootstrapInterval,
    cluster_bootstrap_mean,
    projected_half_width,
)
from janus.interventions.actions import ActionID
from janus.oracle.counterfactual import OracleResult
from janus.policies.partial_information import RidgeActionValuePolicy
from janus.scenarios.generator import CandidateWorld
from janus.scenarios.matching import AmbiguityBundle


@dataclass(frozen=True, slots=True)
class ValidityThresholds:
    maximum_pre_shock_price_deviation: float = 0.05
    maximum_pre_shock_volume_deviation: float = 0.15
    maximum_pre_shock_half_drift: float = 0.05
    maximum_bundle_distance: float = 1.25
    maximum_standardized_mean_difference: float = 2.5
    maximum_nearest_neighbor_distance: float = 1.0
    minimum_classifier_macro_f1: float = 0.35
    maximum_classifier_macro_f1: float = 0.80
    minimum_oracle_action_count: int = 3
    maximum_constant_epsilon_rate: float = 0.60
    minimum_resolvable_world_fraction: float = 0.50
    minimum_mc_resolved_gap_fraction: float = 0.50


@dataclass(frozen=True, slots=True)
class PreShockAudit:
    maximum_price_deviation: float
    maximum_volume_deviation: float
    maximum_half_drift: float
    mean_price: float
    mean_volume: float


@dataclass(frozen=True, slots=True)
class SeparationAudit:
    heuristic_minus_random: BootstrapInterval
    partial_minus_heuristic: BootstrapInterval
    oracle_minus_partial: BootstrapInterval


@dataclass(frozen=True, slots=True)
class PrecisionAudit:
    oracle_minus_random: BootstrapInterval
    projected_25_bundle_half_width: float


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    passed: bool
    observed: float | int
    criterion: str


@dataclass(frozen=True, slots=True)
class ValidityReport:
    pre_shock: PreShockAudit
    overlap: OverlapAudit
    baselines: BaselineAudit
    separation: SeparationAudit
    precision: PrecisionAudit
    resolvable_world_fraction: float
    mc_resolved_gap_fraction: float
    tied_world_count: int
    gates: tuple[GateResult, ...]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)


def _flatten(bundles: tuple[AmbiguityBundle, ...]) -> tuple[CandidateWorld, ...]:
    return tuple(world for bundle in bundles for world in bundle.worlds)


def audit_pre_shock(bundles: tuple[AmbiguityBundle, ...]) -> PreShockAudit:
    price_deviations: list[float] = []
    volume_deviations: list[float] = []
    half_drifts: list[float] = []
    all_prices: list[float] = []
    all_volumes: list[float] = []
    for world in _flatten(bundles):
        rows = tuple(
            row for row in world.observation.rows if row.period < world.scenario.shock_start
        )
        prices = [row.transaction_price for row in rows if row.transaction_price is not None]
        volumes = [float(row.transaction_volume) for row in rows]
        if len(prices) != len(rows) or len(rows) < 4:
            raise ValueError(f"world {world.world_id} lacks a complete pre-shock history")
        reference_price = rows[0].reference_price
        reference_volume = rows[0].reference_volume
        price_deviations.append(abs(float(np.mean(prices)) / reference_price - 1.0))
        volume_deviations.append(abs(float(np.mean(volumes)) / reference_volume - 1.0))
        midpoint = len(rows) // 2
        first_price = float(np.mean(prices[:midpoint]))
        second_price = float(np.mean(prices[midpoint:]))
        first_volume = float(np.mean(volumes[:midpoint]))
        second_volume = float(np.mean(volumes[midpoint:]))
        half_drifts.append(
            max(
                abs(second_price - first_price) / reference_price,
                abs(second_volume - first_volume) / reference_volume,
            )
        )
        all_prices.extend(prices)
        all_volumes.extend(volumes)
    return PreShockAudit(
        maximum_price_deviation=max(price_deviations),
        maximum_volume_deviation=max(volume_deviations),
        maximum_half_drift=max(half_drifts),
        mean_price=float(np.mean(all_prices)),
        mean_volume=float(np.mean(all_volumes)),
    )


def _paired_standard_error(best: OracleResult, alternative: ActionID) -> float:
    best_samples = best.value_for(best.best_action).samples
    other_samples = best.value_for(alternative).samples
    differences = tuple(
        left - right for left, right in zip(best_samples, other_samples, strict=True)
    )
    return 0.0 if len(differences) < 2 else stdev(differences) / sqrt(len(differences))


def _mc_resolved(oracle: OracleResult) -> bool:
    ordered = sorted(oracle.action_values, key=lambda value: value.mean, reverse=True)
    gap = ordered[0].mean - ordered[1].mean
    return gap > 2 * _paired_standard_error(oracle, ordered[1].action)


def _selected_utilities(
    worlds: tuple[CandidateWorld, ...],
    values: np.ndarray,
    actions: tuple[ActionID, ...],
) -> np.ndarray:
    indices = {action: index for index, action in enumerate(ActionID)}
    return np.asarray(
        [values[index, indices[action]] for index, action in enumerate(actions)],
        dtype=float,
    )


def _clustered(values: np.ndarray, bundles: int) -> np.ndarray:
    return values.reshape(bundles, 4)


def build_validity_report(
    development: tuple[AmbiguityBundle, ...],
    calibration: tuple[AmbiguityBundle, ...],
    development_oracles: dict[str, OracleResult],
    calibration_oracles: dict[str, OracleResult],
    *,
    thresholds: ValidityThresholds | None = None,
    root_seed: int = 20261017,
) -> tuple[ValidityReport, RidgeActionValuePolicy]:
    thresholds = thresholds or ValidityThresholds()
    combined = (*development, *calibration)
    pre_shock = audit_pre_shock(combined)
    overlap = audit_overlap(development, calibration)
    baselines, partial_policy = audit_baselines(
        development,
        calibration,
        development_oracles,
        calibration_oracles,
    )

    development_values = action_value_matrix(development, development_oracles)
    calibration_values = action_value_matrix(calibration, calibration_oracles)
    development_heuristic = tune_public_heuristic(development, development_values)
    worlds = _flatten(calibration)
    heuristic_actions = tuple(
        development_heuristic.act(world.serialized_observation) for world in worlds
    )
    partial_actions = tuple(partial_policy.act(world.serialized_observation) for world in worlds)
    random_utility = calibration_values.mean(axis=1)
    heuristic_utility = _selected_utilities(worlds, calibration_values, heuristic_actions)
    partial_utility = _selected_utilities(worlds, calibration_values, partial_actions)
    oracle_utility = calibration_values.max(axis=1)
    bundle_count = len(calibration)
    separation = SeparationAudit(
        heuristic_minus_random=cluster_bootstrap_mean(
            _clustered(heuristic_utility - random_utility, bundle_count),
            root_seed=root_seed,
        ),
        partial_minus_heuristic=cluster_bootstrap_mean(
            _clustered(partial_utility - heuristic_utility, bundle_count),
            root_seed=root_seed + 1,
        ),
        oracle_minus_partial=cluster_bootstrap_mean(
            _clustered(oracle_utility - partial_utility, bundle_count),
            root_seed=root_seed + 2,
        ),
    )
    oracle_random_by_bundle = _clustered(oracle_utility - random_utility, bundle_count)
    bundle_effects = oracle_random_by_bundle.mean(axis=1)
    precision = PrecisionAudit(
        oracle_minus_random=cluster_bootstrap_mean(
            oracle_random_by_bundle,
            root_seed=root_seed + 3,
        ),
        projected_25_bundle_half_width=projected_half_width(bundle_effects, 25),
    )

    oracles = tuple(calibration_oracles[world.world_id] for world in worlds)
    resolvable_fraction = float(
        np.mean([len(oracle.epsilon_optimal_actions) == 1 for oracle in oracles])
    )
    mc_resolved_fraction = float(np.mean([_mc_resolved(oracle) for oracle in oracles]))
    tied_world_count = sum(len(oracle.epsilon_optimal_actions) > 1 for oracle in oracles)
    maximum_bundle_distance = max(
        bundle.maximum_pairwise_distance for bundle in (*development, *calibration)
    )
    classifier_scores = [result.macro_f1 for result in overlap.classifiers]
    gates = (
        GateResult(
            "pre_shock_price",
            pre_shock.maximum_price_deviation <= thresholds.maximum_pre_shock_price_deviation,
            pre_shock.maximum_price_deviation,
            f"<= {thresholds.maximum_pre_shock_price_deviation}",
        ),
        GateResult(
            "pre_shock_volume",
            pre_shock.maximum_volume_deviation <= thresholds.maximum_pre_shock_volume_deviation,
            pre_shock.maximum_volume_deviation,
            f"<= {thresholds.maximum_pre_shock_volume_deviation}",
        ),
        GateResult(
            "pre_shock_drift",
            pre_shock.maximum_half_drift <= thresholds.maximum_pre_shock_half_drift,
            pre_shock.maximum_half_drift,
            f"<= {thresholds.maximum_pre_shock_half_drift}",
        ),
        GateResult(
            "bundle_distance",
            maximum_bundle_distance <= thresholds.maximum_bundle_distance,
            maximum_bundle_distance,
            f"<= {thresholds.maximum_bundle_distance}",
        ),
        GateResult(
            "standardized_mean_difference",
            overlap.maximum_absolute_standardized_mean_difference
            <= thresholds.maximum_standardized_mean_difference,
            overlap.maximum_absolute_standardized_mean_difference,
            f"<= {thresholds.maximum_standardized_mean_difference}",
        ),
        GateResult(
            "nearest_neighbor_distance",
            overlap.mean_cross_family_nearest_neighbor_distance
            <= thresholds.maximum_nearest_neighbor_distance,
            overlap.mean_cross_family_nearest_neighbor_distance,
            f"<= {thresholds.maximum_nearest_neighbor_distance}",
        ),
        GateResult(
            "classifier_lower_bound",
            min(classifier_scores) >= thresholds.minimum_classifier_macro_f1,
            min(classifier_scores),
            f">= {thresholds.minimum_classifier_macro_f1}",
        ),
        GateResult(
            "classifier_upper_bound",
            max(classifier_scores) <= thresholds.maximum_classifier_macro_f1,
            max(classifier_scores),
            f"<= {thresholds.maximum_classifier_macro_f1}",
        ),
        GateResult(
            "oracle_action_diversity",
            baselines.oracle_optimal_action_count >= thresholds.minimum_oracle_action_count,
            baselines.oracle_optimal_action_count,
            f">= {thresholds.minimum_oracle_action_count}",
        ),
        GateResult(
            "constant_policy_nontriviality",
            baselines.maximum_constant_epsilon_optimal_rate
            <= thresholds.maximum_constant_epsilon_rate,
            baselines.maximum_constant_epsilon_optimal_rate,
            f"<= {thresholds.maximum_constant_epsilon_rate}",
        ),
        GateResult(
            "resolvable_world_fraction",
            resolvable_fraction >= thresholds.minimum_resolvable_world_fraction,
            resolvable_fraction,
            f">= {thresholds.minimum_resolvable_world_fraction}",
        ),
        GateResult(
            "mc_resolved_gap_fraction",
            mc_resolved_fraction >= thresholds.minimum_mc_resolved_gap_fraction,
            mc_resolved_fraction,
            f">= {thresholds.minimum_mc_resolved_gap_fraction}",
        ),
        GateResult(
            "heuristic_above_random",
            separation.heuristic_minus_random.lower > 0,
            separation.heuristic_minus_random.lower,
            "95% cluster-bootstrap lower bound > 0",
        ),
        GateResult(
            "partial_above_heuristic",
            separation.partial_minus_heuristic.lower > 0,
            separation.partial_minus_heuristic.lower,
            "95% cluster-bootstrap lower bound > 0",
        ),
        GateResult(
            "oracle_above_partial",
            separation.oracle_minus_partial.lower >= 0,
            separation.oracle_minus_partial.lower,
            "95% cluster-bootstrap lower bound >= 0",
        ),
    )
    return (
        ValidityReport(
            pre_shock=pre_shock,
            overlap=overlap,
            baselines=baselines,
            separation=separation,
            precision=precision,
            resolvable_world_fraction=resolvable_fraction,
            mc_resolved_gap_fraction=mc_resolved_fraction,
            tied_world_count=tied_world_count,
            gates=gates,
        ),
        partial_policy,
    )
