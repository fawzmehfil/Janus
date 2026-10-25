"""Action-margin, contrast-set, and structural OOD design gates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from statistics import stdev
from typing import Any

import numpy as np

from janus.interventions.actions import ActionID
from janus.oracle.counterfactual import OracleResult
from janus.scenarios.generator import CandidateWorld
from janus.scenarios.matching import AmbiguityBundle
from janus.v2.contracts import ACTION_ROLES, ActionRole


@dataclass(frozen=True, slots=True)
class ActionCoverageThresholds:
    minimum_substantive_unique_optimal_rate: float = 0.10
    minimum_contrast_pairs: int = 1
    maximum_contrast_distance: float = 1.25


def audit_action_coverage(
    bundles: tuple[AmbiguityBundle, ...],
    oracles: dict[str, OracleResult],
    *,
    thresholds: ActionCoverageThresholds | None = None,
    documented_exceptions: dict[ActionID, str] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or ActionCoverageThresholds()
    documented_exceptions = documented_exceptions or {}
    worlds = tuple(world for bundle in bundles for world in bundle.worlds)
    strict_counts = {
        action: sum(oracles[world.world_id].best_action is action for world in worlds)
        for action in ActionID
    }
    epsilon_counts = {
        action: sum(action in oracles[world.world_id].epsilon_optimal_actions for world in worlds)
        for action in ActionID
    }
    gaps = []
    paired_standard_errors = []
    item_rows = []
    bundle_by_world = {
        world.world_id: bundle.bundle_id for bundle in bundles for world in bundle.worlds
    }
    for world in worlds:
        oracle = oracles[world.world_id]
        ordered = sorted(
            oracle.action_values,
            key=lambda value: (value.mean, value.action.value),
            reverse=True,
        )
        gaps.append(ordered[0].mean - ordered[1].mean)
        differences = [
            left - right for left, right in zip(ordered[0].samples, ordered[1].samples, strict=True)
        ]
        paired_standard_errors.append(
            0.0 if len(differences) < 2 else stdev(differences) / sqrt(len(differences))
        )
        item_rows.append(
            {
                "bundle_id": bundle_by_world[world.world_id],
                "world_id": world.world_id,
                "family": world.family.value,
                "split": world.split,
                "strict_best_action": oracle.best_action.value,
                "epsilon_optimal_actions": sorted(
                    action.value for action in oracle.epsilon_optimal_actions
                ),
                "best_vs_second_gap": gaps[-1],
                "paired_monte_carlo_standard_error": paired_standard_errors[-1],
                "difficulty_proxy": (
                    "near_tie"
                    if gaps[-1] <= max(4.2, 2 * paired_standard_errors[-1])
                    else "resolved"
                ),
            }
        )
    rows = []
    passed = True
    for action in ActionID:
        unique = sum(
            len(oracles[world.world_id].epsilon_optimal_actions) == 1
            and oracles[world.world_id].best_action is action
            for world in worlds
        )
        rate = unique / len(worlds)
        role = ACTION_ROLES[action]
        exception = documented_exceptions.get(action)
        action_passed = (
            role is not ActionRole.SUBSTANTIVE
            or rate >= thresholds.minimum_substantive_unique_optimal_rate
            or exception is not None
        )
        passed = passed and action_passed
        rows.append(
            {
                "action": action.value,
                "role": role.value,
                "unique_optimal_worlds": unique,
                "unique_optimal_rate": rate,
                "strict_best_worlds": strict_counts[action],
                "epsilon_optimal_worlds": epsilon_counts[action],
                "threshold": (
                    thresholds.minimum_substantive_unique_optimal_rate
                    if role is ActionRole.SUBSTANTIVE
                    else None
                ),
                "documented_exception": exception,
                "passed": action_passed,
            }
        )
    by_family = {
        family.value: {
            action.value: sum(
                world.family is family and oracles[world.world_id].best_action is action
                for world in worlds
            )
            for action in ActionID
        }
        for family in sorted({world.family for world in worlds}, key=lambda value: value.value)
    }
    constant_rates = {action.value: epsilon_counts[action] / len(worlds) for action in ActionID}
    return {
        "schema_version": "janus-v2-action-coverage-v1",
        "world_count": len(worlds),
        "actions": rows,
        "strict_best_action_counts": {action.value: strict_counts[action] for action in ActionID},
        "epsilon_optimal_action_counts": {
            action.value: epsilon_counts[action] for action in ActionID
        },
        "strict_best_action_counts_by_family": by_family,
        "maximum_constant_policy_epsilon_optimal_rate": max(constant_rates.values()),
        "constant_policy_epsilon_optimal_rates": constant_rates,
        "unique_statistically_resolved_worlds": sum(
            len(oracles[world.world_id].epsilon_optimal_actions) == 1 for world in worlds
        ),
        "unique_statistically_resolved_rate": float(
            np.mean([len(oracles[world.world_id].epsilon_optimal_actions) == 1 for world in worlds])
        ),
        "best_vs_second_action_gap": {
            "mean": float(np.mean(gaps)),
            "median": float(np.median(gaps)),
            "minimum": float(np.min(gaps)),
            "maximum": float(np.max(gaps)),
        },
        "paired_monte_carlo_standard_error": {
            "mean": float(np.mean(paired_standard_errors)),
            "maximum": float(np.max(paired_standard_errors)),
        },
        "item_difficulty": item_rows,
        "split_counts": dict(
            sorted(
                {
                    split: sum(world.split == split for world in worlds)
                    for split in {world.split for world in worlds}
                }.items()
            )
        ),
        "passed": passed,
    }


def _standardized_features(worlds: tuple[CandidateWorld, ...]) -> dict[str, np.ndarray]:
    matrix = np.asarray([world.summary_features for world in worlds], dtype=float)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    transformed = (matrix - matrix.mean(axis=0)) / scale
    return {world.world_id: transformed[index] for index, world in enumerate(worlds)}


def audit_matched_contrasts(
    bundles: tuple[AmbiguityBundle, ...],
    oracles: dict[str, OracleResult],
    *,
    maximum_distance: float = 1.25,
) -> dict[str, Any]:
    worlds = tuple(world for bundle in bundles for world in bundle.worlds)
    features = _standardized_features(worlds)
    pairs = []
    for bundle in bundles:
        for left, right in combinations(bundle.worlds, 2):
            left_action = oracles[left.world_id].best_action
            right_action = oracles[right.world_id].best_action
            if left_action is right_action:
                continue
            distance = float(
                np.sqrt(np.mean((features[left.world_id] - features[right.world_id]) ** 2))
            )
            if distance <= maximum_distance:
                pairs.append(
                    {
                        "bundle_id": bundle.bundle_id,
                        "left_world_id": left.world_id,
                        "right_world_id": right.world_id,
                        "left_action": left_action.value,
                        "right_action": right_action.value,
                        "public_distance": distance,
                    }
                )
    return {
        "schema_version": "janus-v2-matched-contrasts-v1",
        "maximum_public_distance": maximum_distance,
        "pair_count": len(pairs),
        "bundles_with_contrast": len({row["bundle_id"] for row in pairs}),
        "pairs": pairs,
        "passed": bool(pairs),
    }


def audit_structural_ood_reversals(
    id_worlds: tuple[CandidateWorld, ...],
    ood_worlds: tuple[CandidateWorld, ...],
    id_oracles: dict[str, OracleResult],
    ood_oracles: dict[str, OracleResult],
    *,
    maximum_distance: float = 1.25,
    minimum_pairs: int = 4,
    minimum_families: int = 2,
) -> dict[str, Any]:
    """Require close public cues whose oracle action changes across ID/OOD.

    This is a direct cue-to-action relationship shift, unlike a severity-only
    extrapolation where the action mapping is unchanged.
    """

    all_worlds = (*id_worlds, *ood_worlds)
    features = _standardized_features(all_worlds)
    candidates: list[tuple[float, CandidateWorld, CandidateWorld]] = []
    for id_world in id_worlds:
        for ood_world in ood_worlds:
            if id_world.family is not ood_world.family:
                continue
            if (
                id_oracles[id_world.world_id].best_action
                is ood_oracles[ood_world.world_id].best_action
            ):
                continue
            distance = float(
                np.sqrt(np.mean((features[id_world.world_id] - features[ood_world.world_id]) ** 2))
            )
            if distance <= maximum_distance:
                candidates.append((distance, id_world, ood_world))
    candidates.sort(key=lambda row: (row[0], row[1].world_id, row[2].world_id))
    selected = []
    used_id: set[str] = set()
    used_ood: set[str] = set()
    for distance, id_world, ood_world in candidates:
        if id_world.world_id in used_id or ood_world.world_id in used_ood:
            continue
        used_id.add(id_world.world_id)
        used_ood.add(ood_world.world_id)
        selected.append(
            {
                "family": id_world.family.value,
                "id_world_id": id_world.world_id,
                "ood_world_id": ood_world.world_id,
                "id_best_action": id_oracles[id_world.world_id].best_action.value,
                "ood_best_action": ood_oracles[ood_world.world_id].best_action.value,
                "public_distance": distance,
            }
        )
    family_count = len({row["family"] for row in selected})
    return {
        "schema_version": "janus-v2-structural-ood-v1",
        "definition": "same-family, close-public-cue pairs with different oracle actions",
        "maximum_public_distance": maximum_distance,
        "minimum_pairs": minimum_pairs,
        "minimum_families": minimum_families,
        "pair_count": len(selected),
        "family_count": family_count,
        "pairs": selected,
        "passed": len(selected) >= minimum_pairs and family_count >= minimum_families,
    }
