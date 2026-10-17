"""Overlap, action-diversity, and baseline-separation validity audits."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from janus.interventions.actions import ActionID
from janus.oracle.counterfactual import OracleResult
from janus.policies.heuristic import PublicHeuristicPolicy
from janus.policies.partial_information import RidgeActionValuePolicy
from janus.scenarios.families import LatentFamily
from janus.scenarios.generator import CandidateWorld
from janus.scenarios.matching import AmbiguityBundle


@dataclass(frozen=True, slots=True)
class ClassifierResult:
    name: str
    accuracy: float
    macro_f1: float
    log_loss: float


@dataclass(frozen=True, slots=True)
class OverlapAudit:
    maximum_absolute_standardized_mean_difference: float
    mean_cross_family_nearest_neighbor_distance: float
    classifiers: tuple[ClassifierResult, ...]


@dataclass(frozen=True, slots=True)
class BaselineAudit:
    oracle_optimal_action_count: int
    maximum_constant_epsilon_optimal_rate: float
    random_mean_utility: float
    heuristic_mean_utility: float
    partial_information_mean_utility: float
    oracle_mean_utility: float
    estimated_information_gap: float
    heuristic_minus_random: float
    partial_minus_heuristic: float
    heuristic_parameters: tuple[tuple[str, float], ...]
    partial_regularization: float


def _flatten(bundles: tuple[AmbiguityBundle, ...]) -> tuple[CandidateWorld, ...]:
    return tuple(world for bundle in bundles for world in bundle.worlds)


def _matrix_and_labels(bundles: tuple[AmbiguityBundle, ...]) -> tuple[np.ndarray, np.ndarray]:
    worlds = _flatten(bundles)
    matrix = np.asarray([world.summary_features for world in worlds], dtype=float)
    labels = np.asarray([world.family.value for world in worlds])
    return matrix, labels


def audit_overlap(
    development: tuple[AmbiguityBundle, ...],
    calibration: tuple[AmbiguityBundle, ...],
) -> OverlapAudit:
    train_x, train_y = _matrix_and_labels(development)
    test_x, test_y = _matrix_and_labels(calibration)
    pooled_scale = train_x.std(axis=0)
    pooled_scale[pooled_scale < 1e-12] = 1.0
    means = {family.value: train_x[train_y == family.value].mean(axis=0) for family in LatentFamily}
    maximum_smd = max(
        float(np.max(np.abs((means[left.value] - means[right.value]) / pooled_scale)))
        for left, right in combinations(LatentFamily, 2)
    )

    standardized = (test_x - train_x.mean(axis=0)) / pooled_scale
    nearest_distances = []
    for index, label in enumerate(test_y):
        other = standardized[test_y != label]
        distances = np.sqrt(np.mean((other - standardized[index]) ** 2, axis=1))
        nearest_distances.append(float(distances.min()))

    classifiers = (
        (
            "multinomial_logistic",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2_000, random_state=0),
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=3,
                learning_rate=0.08,
                l2_regularization=1.0,
                random_state=0,
            ),
        ),
    )
    results = []
    for name, classifier in classifiers:
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(test_x)
        probabilities = classifier.predict_proba(test_x)
        results.append(
            ClassifierResult(
                name=name,
                accuracy=float(accuracy_score(test_y, predictions)),
                macro_f1=float(f1_score(test_y, predictions, average="macro")),
                log_loss=float(log_loss(test_y, probabilities, labels=classifier.classes_)),
            )
        )
    return OverlapAudit(
        maximum_absolute_standardized_mean_difference=maximum_smd,
        mean_cross_family_nearest_neighbor_distance=float(np.mean(nearest_distances)),
        classifiers=tuple(results),
    )


def action_value_matrix(
    bundles: tuple[AmbiguityBundle, ...],
    oracles: dict[str, OracleResult],
) -> np.ndarray:
    worlds = _flatten(bundles)
    return np.asarray(
        [
            [oracles[world.world_id].value_for(action).mean for action in ActionID]
            for world in worlds
        ],
        dtype=float,
    )


def _chosen_utility(
    worlds: tuple[CandidateWorld, ...],
    values: np.ndarray,
    policy: PublicHeuristicPolicy | RidgeActionValuePolicy,
) -> np.ndarray:
    action_indices = {action: index for index, action in enumerate(ActionID)}
    return np.asarray(
        [
            values[index, action_indices[policy.act(world.serialized_observation)]]
            for index, world in enumerate(worlds)
        ]
    )


def tune_public_heuristic(
    development: tuple[AmbiguityBundle, ...],
    development_values: np.ndarray,
) -> PublicHeuristicPolicy:
    worlds = _flatten(development)
    candidates = (
        PublicHeuristicPolicy(inventory, concentration, price, excess)
        for inventory in (0.65, 0.75, 0.85)
        for concentration in (0.10, 0.125, 0.15)
        for price in (1.05, 1.10, 1.15)
        for excess in (0.10, 0.25, 0.40)
    )
    return max(
        candidates,
        key=lambda policy: (
            float(_chosen_utility(worlds, development_values, policy).mean()),
            -policy.inventory_ratio_threshold,
            -policy.concentration_threshold,
            -policy.price_ratio_threshold,
            -policy.excess_demand_ratio_threshold,
        ),
    )


def fit_partial_information_policy(
    development: tuple[AmbiguityBundle, ...],
    development_values: np.ndarray,
) -> RidgeActionValuePolicy:
    worlds = _flatten(development)
    regularizations = (0.1, 1.0, 10.0, 100.0)
    scores: dict[float, list[float]] = {value: [] for value in regularizations}
    for held_out_index, held_out_bundle in enumerate(development):
        train_indices = [
            index for index in range(len(worlds)) if index // len(LatentFamily) != held_out_index
        ]
        validation_start = held_out_index * len(LatentFamily)
        validation_indices = list(
            range(validation_start, validation_start + len(held_out_bundle.worlds))
        )
        train_observations = tuple(worlds[index].serialized_observation for index in train_indices)
        validation_worlds = tuple(worlds[index] for index in validation_indices)
        for regularization in regularizations:
            policy = RidgeActionValuePolicy.fit(
                train_observations,
                development_values[train_indices],
                regularization=regularization,
            )
            score = _chosen_utility(
                validation_worlds,
                development_values[validation_indices],
                policy,
            ).mean()
            scores[regularization].append(float(score))
    selected = max(
        regularizations,
        key=lambda value: (float(np.mean(scores[value])), value),
    )
    return RidgeActionValuePolicy.fit(
        tuple(world.serialized_observation for world in worlds),
        development_values,
        regularization=selected,
    )


def audit_baselines(
    development: tuple[AmbiguityBundle, ...],
    calibration: tuple[AmbiguityBundle, ...],
    development_oracles: dict[str, OracleResult],
    calibration_oracles: dict[str, OracleResult],
) -> tuple[BaselineAudit, RidgeActionValuePolicy]:
    calibration_worlds = _flatten(calibration)
    development_values = action_value_matrix(development, development_oracles)
    calibration_values = action_value_matrix(calibration, calibration_oracles)
    policy = fit_partial_information_policy(development, development_values)
    action_indices = {action: index for index, action in enumerate(ActionID)}
    partial_actions = [policy.act(world.serialized_observation) for world in calibration_worlds]
    heuristic = tune_public_heuristic(development, development_values)
    heuristic_actions = [
        heuristic.act(world.serialized_observation) for world in calibration_worlds
    ]
    partial_utility = np.asarray(
        [
            calibration_values[index, action_indices[action]]
            for index, action in enumerate(partial_actions)
        ]
    )
    heuristic_utility = np.asarray(
        [
            calibration_values[index, action_indices[action]]
            for index, action in enumerate(heuristic_actions)
        ]
    )
    random_utility = calibration_values.mean(axis=1)
    oracle_utility = calibration_values.max(axis=1)
    optimal_actions = {oracle.best_action for oracle in calibration_oracles.values()}
    constant_rates = [
        np.mean(
            [
                action in calibration_oracles[world.world_id].epsilon_optimal_actions
                for world in calibration_worlds
            ]
        )
        for action in ActionID
    ]
    audit = BaselineAudit(
        oracle_optimal_action_count=len(optimal_actions),
        maximum_constant_epsilon_optimal_rate=float(max(constant_rates)),
        random_mean_utility=float(random_utility.mean()),
        heuristic_mean_utility=float(heuristic_utility.mean()),
        partial_information_mean_utility=float(partial_utility.mean()),
        oracle_mean_utility=float(oracle_utility.mean()),
        estimated_information_gap=float((oracle_utility - partial_utility).mean()),
        heuristic_minus_random=float((heuristic_utility - random_utility).mean()),
        partial_minus_heuristic=float((partial_utility - heuristic_utility).mean()),
        heuristic_parameters=(
            ("inventory_ratio_threshold", heuristic.inventory_ratio_threshold),
            ("concentration_threshold", heuristic.concentration_threshold),
            ("price_ratio_threshold", heuristic.price_ratio_threshold),
            ("excess_demand_ratio_threshold", heuristic.excess_demand_ratio_threshold),
        ),
        partial_regularization=policy.regularization,
    )
    return audit, policy
