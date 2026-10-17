from math import sqrt

from janus.core.config import MarketConfig
from janus.evaluation.phase0d_runner import (
    _cache_key,
    _load_or_evaluate,
    _oracle_dict,
    _oracle_from_dict,
    _split_manifest,
)
from janus.evaluation.validity import build_validity_report
from janus.interventions.actions import ActionID, InterventionConfig
from janus.oracle.counterfactual import ActionValue, OracleResult
from janus.scenarios.families import LatentFamily
from janus.scenarios.generator import CandidateGeneratorConfig, generate_candidates
from janus.scenarios.matching import match_ambiguity_bundles
from janus.simulation.runner import DynamicsConfig


def _small_splits():
    config = CandidateGeneratorConfig(candidates_per_family=2, root_seed=83)
    development_candidates = generate_candidates("validity-development", config)
    calibration_candidates = generate_candidates("validity-calibration", config)
    development = match_ambiguity_bundles(
        development_candidates,
        bundle_count=2,
        split="validity-development",
    )
    calibration = match_ambiguity_bundles(
        calibration_candidates,
        bundle_count=2,
        split="validity-calibration",
    )
    return config, development_candidates, development, calibration


def _fake_oracle(family: LatentFamily) -> OracleResult:
    preferred = {
        LatentFamily.SUPPLY_SHORTAGE: ActionID.DO_NOTHING,
        LatentFamily.DEMAND_INCREASE: ActionID.DO_NOTHING,
        LatentFamily.SELLER_COLLUSION: ActionID.COLLUSION_INVESTIGATION,
        LatentFamily.MOMENTUM_DEMAND: ActionID.TRANSACTION_TAX,
    }[family]
    means = {action: 100.0 - 10.0 * index for index, action in enumerate(ActionID)}
    means[preferred] = 120.0
    values = tuple(
        ActionValue(
            action=action,
            samples=(mean - 1.0, mean + 1.0),
            mean=mean,
            standard_error=1.0 / sqrt(2),
        )
        for action, mean in means.items()
    )
    return OracleResult(
        action_values=values,
        best_action=preferred,
        epsilon_optimal_actions=frozenset({preferred}),
        epsilon_by_action=tuple((action, 4.2) for action in ActionID),
        rollout_count=2,
    )


def test_validity_report_builds_from_bundle_level_oracles() -> None:
    _, _, development, calibration = _small_splits()
    development_oracles = {
        world.world_id: _fake_oracle(world.family)
        for bundle in development
        for world in bundle.worlds
    }
    calibration_oracles = {
        world.world_id: _fake_oracle(world.family)
        for bundle in calibration
        for world in bundle.worlds
    }
    report, policy = build_validity_report(
        development,
        calibration,
        development_oracles,
        calibration_oracles,
        root_seed=11,
    )
    assert report.baselines.oracle_optimal_action_count == 3
    assert report.resolvable_world_fraction == 1.0
    assert len(report.gates) == 15
    assert policy.regularization in {0.1, 1.0, 10.0, 100.0}


def test_oracle_cache_round_trip_and_split_rejection_manifest(tmp_path) -> None:
    config, candidates, development, _ = _small_splits()
    world = development[0].worlds[0]
    market = MarketConfig(initial_cash_multiple=config.initial_cash_multiple)
    intervention = InterventionConfig()
    dynamics = DynamicsConfig(recovery_periods=1)
    key = _cache_key(
        world,
        root_seed=config.root_seed,
        rollout_count=2,
        market=market,
        intervention=intervention,
        dynamics=dynamics,
    )
    assert len(key) == 64
    results, hashes = _load_or_evaluate(
        (world,),
        output_directory=tmp_path,
        root_seed=config.root_seed,
        rollout_count=2,
        workers=1,
        market=market,
        intervention=intervention,
        dynamics=dynamics,
    )
    restored = _oracle_from_dict(_oracle_dict(results[world.world_id]))
    assert restored == results[world.world_id]
    cached, cached_hashes = _load_or_evaluate(
        (world,),
        output_directory=tmp_path,
        root_seed=config.root_seed,
        rollout_count=2,
        workers=1,
        market=market,
        intervention=intervention,
        dynamics=dynamics,
    )
    assert cached == results
    assert cached_hashes == hashes
    manifest = _split_manifest(candidates, development)
    assert len(manifest["worlds"]) == 8
    assert len(manifest["rejected_candidates"]) == 0
