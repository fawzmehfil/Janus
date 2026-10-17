"""Resumable, local-only Phase 0D calibration workflow."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from janus.core.config import MarketConfig
from janus.core.replay import content_hash
from janus.evaluation.validity import ValidityThresholds, build_validity_report
from janus.interventions.actions import ActionID, InterventionConfig
from janus.oracle.counterfactual import ActionValue, OracleResult, evaluate_oracle
from janus.scenarios.generator import (
    CandidateGeneratorConfig,
    CandidateWorld,
    generate_candidates,
)
from janus.scenarios.matching import AmbiguityBundle, match_ambiguity_bundles
from janus.simulation.runner import DynamicsConfig


def _generate_split(job: tuple[str, CandidateGeneratorConfig]) -> tuple[CandidateWorld, ...]:
    return generate_candidates(*job)


def _evaluate_job(
    job: tuple[
        CandidateWorld,
        int,
        int,
        MarketConfig,
        InterventionConfig,
        DynamicsConfig,
    ],
) -> tuple[str, OracleResult]:
    world, root_seed, rollout_count, market, intervention, dynamics = job
    return (
        world.world_id,
        evaluate_oracle(
            world.decision_state,
            world.scenario,
            root_seed=root_seed,
            rollout_count=rollout_count,
            market_config=market,
            intervention_config=intervention,
            dynamics=dynamics,
        ),
    )


def _oracle_dict(result: OracleResult) -> dict[str, Any]:
    return {
        "action_values": [
            {
                "action": value.action.value,
                "samples": list(value.samples),
                "mean": value.mean,
                "standard_error": value.standard_error,
            }
            for value in result.action_values
        ],
        "best_action": result.best_action.value,
        "epsilon_optimal_actions": sorted(
            action.value for action in result.epsilon_optimal_actions
        ),
        "epsilon_by_action": [
            [action.value, epsilon] for action, epsilon in result.epsilon_by_action
        ],
        "rollout_count": result.rollout_count,
    }


def _oracle_from_dict(payload: dict[str, Any]) -> OracleResult:
    return OracleResult(
        action_values=tuple(
            ActionValue(
                action=ActionID(row["action"]),
                samples=tuple(float(value) for value in row["samples"]),
                mean=float(row["mean"]),
                standard_error=float(row["standard_error"]),
            )
            for row in payload["action_values"]
        ),
        best_action=ActionID(payload["best_action"]),
        epsilon_optimal_actions=frozenset(
            ActionID(value) for value in payload["epsilon_optimal_actions"]
        ),
        epsilon_by_action=tuple(
            (ActionID(action), float(epsilon)) for action, epsilon in payload["epsilon_by_action"]
        ),
        rollout_count=int(payload["rollout_count"]),
    )


def _cache_key(
    world: CandidateWorld,
    *,
    root_seed: int,
    rollout_count: int,
    market: MarketConfig,
    intervention: InterventionConfig,
    dynamics: DynamicsConfig,
) -> str:
    return content_hash(
        {
            "world_id": world.world_id,
            "state": asdict(world.decision_state),
            "scenario": world.scenario.model_dump(mode="json"),
            "root_seed": root_seed,
            "rollout_count": rollout_count,
            "market": market.model_dump(mode="json"),
            "intervention": intervention.model_dump(mode="json"),
            "dynamics": dynamics.model_dump(mode="json"),
        }
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    )


def _load_or_evaluate(
    worlds: tuple[CandidateWorld, ...],
    *,
    output_directory: Path,
    root_seed: int,
    rollout_count: int,
    workers: int,
    market: MarketConfig,
    intervention: InterventionConfig,
    dynamics: DynamicsConfig,
) -> tuple[dict[str, OracleResult], dict[str, str]]:
    oracle_directory = output_directory / "oracles"
    oracle_directory.mkdir(parents=True, exist_ok=True)
    results: dict[str, OracleResult] = {}
    hashes: dict[str, str] = {}
    missing: list[CandidateWorld] = []
    keys: dict[str, str] = {}
    for world in worlds:
        key = _cache_key(
            world,
            root_seed=root_seed,
            rollout_count=rollout_count,
            market=market,
            intervention=intervention,
            dynamics=dynamics,
        )
        keys[world.world_id] = key
        path = oracle_directory / f"{world.world_id}-k{rollout_count}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("cache_key") == key:
                results[world.world_id] = _oracle_from_dict(payload["oracle"])
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                continue
        missing.append(world)

    jobs = tuple(
        (world, root_seed, rollout_count, market, intervention, dynamics) for world in missing
    )
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            evaluated = executor.map(_evaluate_job, jobs)
            for world_id, result in evaluated:
                results[world_id] = result
                path = oracle_directory / f"{world_id}-k{rollout_count}.json"
                _write_json(
                    path,
                    {
                        "cache_key": keys[world_id],
                        "world_id": world_id,
                        "oracle": _oracle_dict(result),
                    },
                )
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return results, hashes


def _split_manifest(
    candidates: tuple[CandidateWorld, ...],
    bundles: tuple[AmbiguityBundle, ...],
) -> dict[str, Any]:
    selected_ids = {world.world_id for bundle in bundles for world in bundle.worlds}
    return {
        "bundles": [
            {
                "bundle_id": bundle.bundle_id,
                "world_ids": [world.world_id for world in bundle.worlds],
                "mean_pairwise_distance": bundle.mean_pairwise_distance,
                "maximum_pairwise_distance": bundle.maximum_pairwise_distance,
            }
            for bundle in bundles
        ],
        "worlds": [
            {
                "world_id": world.world_id,
                "family": world.family.value,
                "scenario": world.scenario.model_dump(mode="json"),
                "decision_state_hash": content_hash(asdict(world.decision_state)),
                "observation_sha256": hashlib.sha256(world.serialized_observation).hexdigest(),
            }
            for bundle in bundles
            for world in bundle.worlds
        ],
        "rejected_candidates": [
            {
                "world_id": world.world_id,
                "family": world.family.value,
                "reason": "not selected by frozen minimum-distance bundle ranking",
            }
            for world in candidates
            if world.world_id not in selected_ids
        ],
    }


def _convergence_report(
    calibration: tuple[AmbiguityBundle, ...],
    base_oracles: dict[str, OracleResult],
    *,
    output_directory: Path,
    root_seed: int,
    base_rollouts: int,
    workers: int,
    market: MarketConfig,
    intervention: InterventionConfig,
    dynamics: DynamicsConfig,
) -> tuple[dict[str, Any], dict[str, str]]:
    representatives = calibration[0].worlds
    all_results = {
        base_rollouts: {world.world_id: base_oracles[world.world_id] for world in representatives}
    }
    all_hashes: dict[str, str] = {}
    for rollout_count in (base_rollouts * 2, base_rollouts * 4):
        results, hashes = _load_or_evaluate(
            representatives,
            output_directory=output_directory,
            root_seed=root_seed,
            rollout_count=rollout_count,
            workers=workers,
            market=market,
            intervention=intervention,
            dynamics=dynamics,
        )
        all_results[rollout_count] = results
        all_hashes.update(hashes)

    rows = []
    for world in representatives:
        sequence = tuple(all_results[count][world.world_id] for count in sorted(all_results))
        largest = sequence[-1]
        maximum_delta = max(
            abs(result.value_for(action).mean - largest.value_for(action).mean)
            for result in sequence[:-1]
            for action in ActionID
        )
        best_actions = [result.best_action.value for result in sequence]
        rows.append(
            {
                "world_id": world.world_id,
                "family": world.family.value,
                "rollout_counts": sorted(all_results),
                "best_actions": best_actions,
                "stable": len(set(best_actions)) == 1,
                "maximum_mean_delta": maximum_delta,
            }
        )
    return {"worlds": rows, "passed": all(row["stable"] for row in rows)}, all_hashes


def run_phase0d(
    output_directory: Path,
    *,
    rollout_count: int = 256,
    workers: int = 4,
    run_convergence: bool = True,
) -> bool:
    if rollout_count < 2:
        raise ValueError("Phase 0D requires at least two rollouts")
    if workers < 1:
        raise ValueError("workers must be positive")
    generator = CandidateGeneratorConfig()
    market = MarketConfig(initial_cash_multiple=generator.initial_cash_multiple)
    intervention = InterventionConfig()
    dynamics = DynamicsConfig(recovery_periods=20)
    thresholds = ValidityThresholds()
    root_seed = generator.root_seed

    with ProcessPoolExecutor(max_workers=min(workers, 2)) as executor:
        development_candidates, calibration_candidates = executor.map(
            _generate_split,
            (("development", generator), ("calibration", generator)),
        )
    development = match_ambiguity_bundles(
        development_candidates,
        bundle_count=10,
        split="development",
    )
    calibration = match_ambiguity_bundles(
        calibration_candidates,
        bundle_count=15,
        split="calibration",
    )
    worlds = tuple(world for bundle in (*development, *calibration) for world in bundle.worlds)
    oracles, oracle_hashes = _load_or_evaluate(
        worlds,
        output_directory=output_directory,
        root_seed=root_seed,
        rollout_count=rollout_count,
        workers=workers,
        market=market,
        intervention=intervention,
        dynamics=dynamics,
    )
    development_ids = {world.world_id for bundle in development for world in bundle.worlds}
    development_oracles = {key: value for key, value in oracles.items() if key in development_ids}
    calibration_oracles = {
        key: value for key, value in oracles.items() if key not in development_ids
    }
    report, partial_policy = build_validity_report(
        development,
        calibration,
        development_oracles,
        calibration_oracles,
        thresholds=thresholds,
        root_seed=root_seed,
    )
    convergence: dict[str, Any] = {"passed": True, "skipped": True}
    convergence_hashes: dict[str, str] = {}
    if run_convergence:
        convergence, convergence_hashes = _convergence_report(
            calibration,
            calibration_oracles,
            output_directory=output_directory,
            root_seed=root_seed,
            base_rollouts=rollout_count,
            workers=workers,
            market=market,
            intervention=intervention,
            dynamics=dynamics,
        )

    _write_json(
        output_directory / "split_manifest.json",
        {
            "generator": generator.model_dump(mode="json"),
            "development": _split_manifest(development_candidates, development),
            "calibration": _split_manifest(calibration_candidates, calibration),
        },
    )
    _write_json(output_directory / "partial_policy.json", asdict(partial_policy))
    validity_payload = asdict(report)
    validity_payload["passed"] = report.passed
    validity_payload["oracle_convergence"] = convergence
    validity_payload["overall_passed"] = report.passed and bool(convergence["passed"])
    _write_json(output_directory / "validity_report.json", validity_payload)

    primary_files = ("split_manifest.json", "partial_policy.json", "validity_report.json")
    artifact_hashes = {
        name: hashlib.sha256((output_directory / name).read_bytes()).hexdigest()
        for name in primary_files
    }
    artifact_hashes.update({f"oracles/{key}": value for key, value in oracle_hashes.items()})
    artifact_hashes.update({f"oracles/{key}": value for key, value in convergence_hashes.items()})
    _write_json(
        output_directory / "run_manifest.json",
        {
            "phase": "0D",
            "paid_api_calls": 0,
            "root_seed": root_seed,
            "rollout_count": rollout_count,
            "workers": workers,
            "market_config": market.model_dump(mode="json"),
            "intervention_config": intervention.model_dump(mode="json"),
            "dynamics_config": dynamics.model_dump(mode="json"),
            "thresholds": asdict(thresholds),
            "artifact_sha256": dict(sorted(artifact_hashes.items())),
            "overall_passed": validity_payload["overall_passed"],
        },
    )
    return bool(validity_payload["overall_passed"])
