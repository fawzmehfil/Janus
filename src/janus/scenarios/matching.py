"""Bundle construction based exclusively on standardized public summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from janus.scenarios.families import LatentFamily
from janus.scenarios.generator import CandidateWorld


@dataclass(frozen=True, slots=True)
class AmbiguityBundle:
    bundle_id: str
    worlds: tuple[CandidateWorld, ...]
    mean_pairwise_distance: float
    maximum_pairwise_distance: float

    def __post_init__(self) -> None:
        if {world.family for world in self.worlds} != set(LatentFamily):
            raise ValueError("a bundle must contain exactly one world from each family")


def _standardized_features(
    worlds: tuple[CandidateWorld, ...],
) -> dict[str, np.ndarray]:
    matrix = np.asarray([world.summary_features for world in worlds], dtype=float)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (matrix - mean) / scale
    return {world.world_id: standardized[index] for index, world in enumerate(worlds)}


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _assign(
    centroids: list[np.ndarray],
    candidates: tuple[CandidateWorld, ...],
    features: dict[str, np.ndarray],
) -> tuple[CandidateWorld, ...]:
    costs = np.asarray(
        [
            [_distance(centroid, features[candidate.world_id]) for candidate in candidates]
            for centroid in centroids
        ]
    )
    row_indices, column_indices = linear_sum_assignment(costs)
    assignment: list[CandidateWorld | None] = [None] * len(centroids)
    for row, column in zip(row_indices, column_indices, strict=True):
        assignment[int(row)] = candidates[int(column)]
    if any(world is None for world in assignment):
        raise AssertionError("matching failed to assign every provisional bundle")
    return tuple(world for world in assignment if world is not None)


def match_ambiguity_bundles(
    worlds: tuple[CandidateWorld, ...],
    *,
    bundle_count: int,
    split: str,
) -> tuple[AmbiguityBundle, ...]:
    by_family = {
        family: tuple(world for world in worlds if world.family is family)
        for family in LatentFamily
    }
    family_counts = {len(items) for items in by_family.values()}
    if len(family_counts) != 1:
        raise ValueError("candidate families must have equal size before matching")
    candidate_count = family_counts.pop()
    if bundle_count > candidate_count:
        raise ValueError("requested more bundles than candidates per family")

    features = _standardized_features(worlds)
    anchors = by_family[LatentFamily.SUPPLY_SHORTAGE]
    provisional: list[list[CandidateWorld]] = [[world] for world in anchors]
    centroids = [features[world.world_id].copy() for world in anchors]

    for family in (
        LatentFamily.DEMAND_INCREASE,
        LatentFamily.SELLER_COLLUSION,
        LatentFamily.MOMENTUM_DEMAND,
    ):
        assigned = _assign(centroids, by_family[family], features)
        for index, world in enumerate(assigned):
            provisional[index].append(world)
            centroids[index] = np.mean(
                [features[item.world_id] for item in provisional[index]], axis=0
            )

    ranked: list[tuple[float, float, list[CandidateWorld]]] = []
    for bundle in provisional:
        distances = [
            _distance(features[left.world_id], features[right.world_id])
            for index, left in enumerate(bundle)
            for right in bundle[index + 1 :]
        ]
        ranked.append((float(np.mean(distances)), max(distances), bundle))
    ranked.sort(key=lambda item: (item[0], item[1], item[2][0].world_id))

    selected = []
    family_order = tuple(LatentFamily)
    for index, (mean_distance, maximum_distance, bundle) in enumerate(ranked[:bundle_count]):
        ordered = tuple(
            next(world for world in bundle if world.family is family) for family in family_order
        )
        selected.append(
            AmbiguityBundle(
                bundle_id=f"{split}-bundle-{index:03d}",
                worlds=ordered,
                mean_pairwise_distance=mean_distance,
                maximum_pairwise_distance=maximum_distance,
            )
        )
    return tuple(selected)
