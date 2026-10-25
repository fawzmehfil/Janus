"""Pilot variance decomposition and simulation-based v2 allocation selection."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from janus.interventions.actions import ActionID


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _v1_paired_losses(project_root: Path) -> list[dict[str, Any]]:
    response_manifest = _read(
        project_root / "artifacts/openai-hidden-v1/manifests/run-20261019T083004.963054Z.json"
    )
    private = _read(project_root / "artifacts/hidden-v1/private_manifest.json")
    worlds = {str(row["opaque_world_id"]): row for row in private["worlds"]}
    cache = project_root / "artifacts/openai-hidden-v1/cache"
    rows = []
    for name in response_manifest["response_artifact_sha256"]:
        response = _read(cache / name)
        world_id = str(response["world_id"])
        world = worlds[world_id]
        oracle = _read(project_root / "artifacts/hidden-v1" / world["oracle_file"])["oracle"]
        values = {str(row["action"]): float(row["mean"]) for row in oracle["action_values"]}
        chosen = str(response["fallback_action"])
        # The v1 partial-information result is available in the sealed analysis,
        # but action-level reconstruction is deliberately avoided here. Oracle
        # regret supplies the conservative variance pilot for allocation.
        rows.append(
            {
                "bundle_id": str(world["opaque_bundle_id"]),
                "world_id": world_id,
                "replicate_id": str(response["replicate_id"]),
                "family": str(world["family"]),
                "stratum": str(world["stratum"]),
                "loss": max(values.values())
                - values.get(chosen, values[ActionID.DO_NOTHING.value]),
            }
        )
    return rows


def _variance_components(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_world: dict[str, list[float]] = defaultdict(list)
    world_bundle: dict[str, str] = {}
    for row in rows:
        world_id = str(row["world_id"])
        by_world[world_id].append(float(row["loss"]))
        world_bundle[world_id] = str(row["bundle_id"])
    world_means = {world: float(np.mean(values)) for world, values in by_world.items()}
    by_bundle: dict[str, list[float]] = defaultdict(list)
    for world, value in world_means.items():
        by_bundle[world_bundle[world]].append(value)
    bundle_means = {bundle: float(np.mean(values)) for bundle, values in by_bundle.items()}
    replicate_residuals = [
        value - world_means[world] for world, values in by_world.items() for value in values
    ]
    world_residuals = [
        value - bundle_means[world_bundle[world]] for world, value in world_means.items()
    ]
    family_means = {
        family: float(np.mean([float(row["loss"]) for row in rows if row["family"] == family]))
        for family in sorted({str(row["family"]) for row in rows})
    }
    stratum_means = {
        stratum: float(np.mean([float(row["loss"]) for row in rows if row["stratum"] == stratum]))
        for stratum in sorted({str(row["stratum"]) for row in rows})
    }
    return {
        "bundle": float(np.var(list(bundle_means.values()), ddof=1)),
        "world_within_bundle": float(np.var(world_residuals, ddof=1)),
        "replicate_within_world": float(np.var(replicate_residuals, ddof=1)),
        "family_mean_dispersion": float(np.var(list(family_means.values()), ddof=1)),
        "ood_id_mean_dispersion": float(np.var(list(stratum_means.values()), ddof=1)),
    }


def _simulate_design(
    components: dict[str, float],
    *,
    bundles: int,
    replicates: float,
    simulations: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(simulations)
    four_worlds = 4
    for index in range(simulations):
        bundle_effect = rng.normal(0, np.sqrt(components["bundle"]), size=bundles)
        world_effect = rng.normal(
            0,
            np.sqrt(components["world_within_bundle"]),
            size=(bundles, four_worlds),
        )
        # Fractional replicates represent a preregistered balanced split-replicate design.
        replicate_variance = components["replicate_within_world"] / replicates
        replicate_effect = rng.normal(0, np.sqrt(replicate_variance), size=(bundles, four_worlds))
        estimates[index] = np.mean(bundle_effect[:, None] + world_effect + replicate_effect)
    standard_error = float(np.std(estimates, ddof=1))
    return {
        "bundles": bundles,
        "worlds": bundles * four_worlds,
        "average_replicates_per_world": replicates,
        "planned_trials_per_model": round(bundles * four_worlds * replicates),
        "simulated_standard_error": standard_error,
        "expected_95_half_width": 1.96 * standard_error,
        "minimum_detectable_effect_80pct_power": (1.96 + 0.8416) * standard_error,
    }


def run_precision_analysis(project_root: Path, output_path: Path) -> dict[str, Any]:
    rows = _v1_paired_losses(project_root)
    components = _variance_components(rows)
    candidates = {
        "v1_three_replicates": (25, 3.0),
        "more_bundles_two_replicates": (38, 2.0),
        "split_replicate": (40, 1.875),
    }
    designs = {
        name: _simulate_design(
            components,
            bundles=bundles,
            replicates=replicates,
            simulations=20_000,
            seed=20261020 + index,
        )
        for index, (name, (bundles, replicates)) in enumerate(candidates.items())
    }
    selected = min(
        designs,
        key=lambda name: (
            float(designs[name]["expected_95_half_width"]),
            int(designs[name]["planned_trials_per_model"]),
        ),
    )
    report = {
        "schema_version": "janus-v2-precision-v1",
        "pilot": "completed hidden-v1 Luna oracle regret; allocation use only",
        "pilot_trials": len(rows),
        "pilot_bundles": len({row["bundle_id"] for row in rows}),
        "variance_components": components,
        "simulation_count_per_design": 20_000,
        "designs": designs,
        "selected_design": selected,
        "selection_rule": "smallest simulated 95% half-width at approximately 300 trials/model",
        "independent_unit": "ambiguity bundle",
        "worlds_are_nested_not_independent": True,
        "model_count_not_used_to_inflate_n": True,
        "paid_api_calls": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report
