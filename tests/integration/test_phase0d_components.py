import numpy as np
import pytest

from janus.evaluation.statistics import cluster_bootstrap_mean, projected_half_width
from janus.evaluation.validity import audit_pre_shock
from janus.interventions.actions import ActionID
from janus.policies.partial_information import RidgeActionValuePolicy, features_from_serialized
from janus.scenarios.families import LatentFamily
from janus.scenarios.generator import CandidateGeneratorConfig, generate_candidates
from janus.scenarios.matching import match_ambiguity_bundles


@pytest.fixture(scope="module")
def small_candidates():
    return generate_candidates(
        "component-test",
        CandidateGeneratorConfig(candidates_per_family=2, root_seed=71),
    )


def test_candidate_generation_and_matching_are_deterministic(small_candidates) -> None:
    repeated = generate_candidates(
        "component-test",
        CandidateGeneratorConfig(candidates_per_family=2, root_seed=71),
    )
    assert [world.world_id for world in small_candidates] == [world.world_id for world in repeated]
    assert [world.serialized_observation for world in small_candidates] == [
        world.serialized_observation for world in repeated
    ]
    bundles = match_ambiguity_bundles(
        small_candidates,
        bundle_count=2,
        split="component-test",
    )
    assert len(bundles) == 2
    assert all({world.family for world in bundle.worlds} == set(LatentFamily) for bundle in bundles)
    assert all(len({world.world_id for world in bundle.worlds}) == 4 for bundle in bundles)


def test_public_feature_parser_and_ridge_policy_use_exact_payload(small_candidates) -> None:
    observations = tuple(world.serialized_observation for world in small_candidates)
    assert all(features_from_serialized(item).shape == (74,) for item in observations)
    targets = np.tile(np.arange(len(ActionID), dtype=float), (len(observations), 1))
    policy = RidgeActionValuePolicy.fit(observations, targets, regularization=1.0)
    assert policy.act(observations[0]) is ActionID.COLLUSION_INVESTIGATION


def test_pre_shock_and_clustered_statistics_are_reproducible(small_candidates) -> None:
    bundles = match_ambiguity_bundles(
        small_candidates,
        bundle_count=2,
        split="component-test",
    )
    audit = audit_pre_shock(bundles)
    assert audit.maximum_price_deviation < 0.05
    assert audit.maximum_volume_deviation < 0.15
    values = np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
    first = cluster_bootstrap_mean(values, root_seed=9, bootstrap_samples=100)
    second = cluster_bootstrap_mean(values, root_seed=9, bootstrap_samples=100)
    assert first == second
    assert projected_half_width(values.mean(axis=1), target_bundles=25) > 0


def test_generator_rejects_reversed_duration_range() -> None:
    with pytest.raises(ValueError, match="shock_duration_min"):
        CandidateGeneratorConfig(shock_duration_min=20, shock_duration_max=10)
