"""Local-only Janus v2 scientific gate runner."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np

from janus.core.randomness import DisturbanceTape
from janus.evaluation.phase0d_runner import _oracle_from_dict
from janus.interventions.actions import ActionID
from janus.oracle.counterfactual import OracleResult
from janus.scenarios.generator import CandidateGeneratorConfig, CandidateWorld, generate_candidates
from janus.scenarios.matching import AmbiguityBundle
from janus.simulation.runner import DynamicsConfig, run_recovery
from janus.v2.baselines import BaselineDataset, fit_and_freeze_baseline
from janus.v2.design import audit_action_coverage, audit_matched_contrasts
from janus.v2.precision import run_precision_analysis

OBJECTIVES = (
    "primary",
    "consumer_protection",
    "producer_continuity",
    "stability_aware",
    "fiscal_neutral",
)


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _selected_bundles(
    split: str,
    rows: list[dict[str, Any]],
) -> tuple[AmbiguityBundle, ...]:
    candidates = {
        world.world_id: world for world in generate_candidates(split, CandidateGeneratorConfig())
    }
    return tuple(
        AmbiguityBundle(
            bundle_id=str(row["bundle_id"]),
            worlds=tuple(candidates[str(world_id)] for world_id in row["world_ids"]),
            mean_pairwise_distance=float(row["mean_pairwise_distance"]),
            maximum_pairwise_distance=float(row["maximum_pairwise_distance"]),
        )
        for row in rows
    )


def _primary_oracles(
    root: Path,
    worlds: tuple[CandidateWorld, ...],
    *,
    rollouts: int = 256,
) -> dict[str, OracleResult]:
    result = {}
    for world in worlds:
        payload = _read(
            root / "artifacts/phase0d-release/oracles" / f"{world.world_id}-k{rollouts}.json"
        )
        result[world.world_id] = _oracle_from_dict(payload["oracle"])
    return result


def _vector_job(job: tuple[CandidateWorld, int, int]) -> tuple[str, dict[str, list[float]]]:
    world, root_seed, rollout_count = job
    values = {objective: [0.0] * len(ActionID) for objective in OBJECTIVES}
    samples: dict[str, dict[ActionID, list[float]]] = {
        objective: {action: [] for action in ActionID} for objective in OBJECTIVES
    }
    for rollout in range(rollout_count):
        tape = DisturbanceTape(root_seed, f"{world.scenario.scenario_id}:v2-objectives:{rollout}")
        for action in ActionID:
            result = run_recovery(
                world.decision_state,
                world.scenario,
                action,
                tape,
                dynamics=DynamicsConfig(recovery_periods=20),
            )
            for objective in OBJECTIVES:
                samples[objective][action].append(float(getattr(result.utilities, objective)))
    for objective in OBJECTIVES:
        values[objective] = [float(np.mean(samples[objective][action])) for action in ActionID]
    return world.world_id, values


def _objective_values(
    worlds: tuple[CandidateWorld, ...],
    cache_path: Path,
    *,
    rollouts: int,
    workers: int,
) -> dict[str, dict[str, list[float]]]:
    source_hash = hashlib.sha256(
        json.dumps(
            {
                "worlds": [world.world_id for world in worlds],
                "rollouts": rollouts,
                "seed": 20261017,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if cache_path.exists():
        cached = _read(cache_path)
        if cached.get("source_sha256") == source_hash:
            return cast(dict[str, dict[str, list[float]]], cached["worlds"])
    jobs = tuple((world, 20261017, rollouts) for world in worlds)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = dict(executor.map(_vector_job, jobs))
    _write(
        cache_path,
        {
            "schema_version": "janus-v2-development-objective-values-v1",
            "source_sha256": source_hash,
            "rollouts": rollouts,
            "paid_api_calls": 0,
            "worlds": rows,
        },
    )
    return rows


def run_v2_local_development(
    project_root: Path,
    output_directory: Path,
    *,
    objective_rollouts: int = 64,
    workers: int = 4,
) -> dict[str, Any]:
    if objective_rollouts < 16:
        raise ValueError("v2 objective baseline development requires at least 16 rollouts")
    split_path = project_root / "artifacts/phase0d-release/split_manifest.json"
    split = _read(split_path)
    development = _selected_bundles("development", split["development"]["bundles"])
    calibration = _selected_bundles("calibration", split["calibration"]["bundles"])
    development_worlds = tuple(world for bundle in development for world in bundle.worlds)
    calibration_worlds = tuple(world for bundle in calibration for world in bundle.worlds)
    development_oracles = _primary_oracles(project_root, development_worlds)
    calibration_oracles = _primary_oracles(project_root, calibration_worlds)

    objective_values_path = output_directory / "objective_values.json"
    objective_values = _objective_values(
        development_worlds,
        objective_values_path,
        rollouts=objective_rollouts,
        workers=workers,
    )
    primary_values = np.asarray(
        [
            [development_oracles[world.world_id].value_for(action).mean for action in ActionID]
            for world in development_worlds
        ]
    )
    baseline_directory = project_root / "manifests/v2/baselines"
    bundle_by_world = {
        world.world_id: bundle.bundle_id for bundle in development for world in bundle.worlds
    }
    observations = tuple(world.serialized_observation for world in development_worlds)
    bundle_ids = tuple(bundle_by_world[world.world_id] for world in development_worlds)
    world_ids = tuple(world.world_id for world in development_worlds)
    primary_source_digest = hashlib.sha256(split_path.read_bytes())
    for world in sorted(development_worlds, key=lambda value: value.world_id):
        primary_source_digest.update(
            (
                project_root / "artifacts/phase0d-release/oracles" / f"{world.world_id}-k256.json"
            ).read_bytes()
        )
    primary_source_hash = primary_source_digest.hexdigest()
    objective_source_digest = hashlib.sha256(split_path.read_bytes())
    objective_source_digest.update(objective_values_path.read_bytes())
    objective_source_hash = objective_source_digest.hexdigest()
    baselines = {
        "primary": fit_and_freeze_baseline(
            BaselineDataset(observations, primary_values, bundle_ids, world_ids),
            baseline_directory,
            objective="primary",
            training_source_sha256=primary_source_hash,
        )
    }
    for objective in OBJECTIVES[1:]:
        matrix = np.asarray(
            [objective_values[world.world_id][objective] for world in development_worlds]
        )
        baselines[objective] = fit_and_freeze_baseline(
            BaselineDataset(observations, matrix, bundle_ids, world_ids),
            baseline_directory,
            objective=objective,
            training_source_sha256=objective_source_hash,
        )

    action_coverage = audit_action_coverage(calibration, calibration_oracles)
    contrasts = audit_matched_contrasts(calibration, calibration_oracles)
    precision = run_precision_analysis(project_root, project_root / "reports/v2/precision.json")
    _write(project_root / "reports/v2/action-coverage.json", action_coverage)
    _write(project_root / "reports/v2/matched-contrasts.json", contrasts)
    baseline_summary = {
        objective: {
            "selected_candidate": manifest["selected_candidate"],
            "nested_mean_decision_utility": manifest["nested_mean_decision_utility"],
            "training_world_count": manifest["training_world_count"],
        }
        for objective, manifest in baselines.items()
    }
    report = {
        "schema_version": "janus-v2-local-development-v1",
        "paid_api_calls": 0,
        "hidden_v2_generated": False,
        "development_bundles": len(development),
        "calibration_bundles": len(calibration),
        "objective_rollouts": objective_rollouts,
        "baseline_summary": baseline_summary,
        "action_coverage_passed": action_coverage["passed"],
        "matched_contrasts_passed": contrasts["passed"],
        "precision_selected_design": precision["selected_design"],
        "passed": bool(action_coverage["passed"] and contrasts["passed"]),
    }
    _write(output_directory / "run_manifest.json", report)
    return report
