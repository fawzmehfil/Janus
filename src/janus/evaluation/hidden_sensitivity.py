"""Post-seal hidden-v1 sensitivity analyses with no provider calls."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, cast

import numpy as np

from janus.core.config import MarketConfig
from janus.core.randomness import DisturbanceTape
from janus.core.replay import canonical_json
from janus.core.state import BuyerState, MarketState, SellerState
from janus.evaluation.hidden_analysis import (
    BOOTSTRAP_SEED,
    REFERENCE_SURPLUS_PER_PERIOD,
    _group_interval,
)
from janus.interventions.actions import ActionID, InterventionConfig
from janus.policies.heuristic import PublicHeuristicPolicy
from janus.policies.partial_information import RidgeActionValuePolicy
from janus.scenarios.families import ScenarioConfig
from janus.simulation.runner import DynamicsConfig, run_recovery

OBJECTIVES = (
    "primary",
    "consumer_protection",
    "producer_continuity",
    "stability_aware",
    "fiscal_neutral",
)
ALTERNATIVE_OBJECTIVES = OBJECTIVES[1:]
ROLLOUT_COUNT = 256
ABSOLUTE_EPSILON_FLOOR = 0.005 * REFERENCE_SURPLUS_PER_PERIOD


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _market_state(payload: dict[str, Any]) -> MarketState:
    return MarketState(
        period=int(payload["period"]),
        buyers=tuple(BuyerState(**buyer) for buyer in payload["buyers"]),
        sellers=tuple(
            SellerState(
                seller_id=str(seller["seller_id"]),
                cash=float(seller["cash"]),
                inventory=int(seller["inventory"]),
                capacity=int(seller["capacity"]),
                marginal_costs=tuple(float(value) for value in seller["marginal_costs"]),
                active=bool(seller["active"]),
            )
            for seller in payload["sellers"]
        ),
        government_balance=float(payload["government_balance"]),
        last_price=(None if payload["last_price"] is None else float(payload["last_price"])),
        recent_transaction_prices=tuple(
            float(value) for value in payload["recent_transaction_prices"]
        ),
        cumulative_real_cost=float(payload["cumulative_real_cost"]),
    )


def _standard_error(values: tuple[float, ...]) -> float:
    return 0.0 if len(values) < 2 else stdev(values) / sqrt(len(values))


def _objective_summary(samples_by_action: dict[str, tuple[float, ...]]) -> dict[str, Any]:
    means = {action: fmean(samples) for action, samples in samples_by_action.items()}
    best_action = max(means, key=lambda action: (means[action], action))
    best_samples = samples_by_action[best_action]
    epsilon_by_action: dict[str, float] = {}
    epsilon_optimal_actions: list[str] = []
    for action in ActionID:
        action_id = action.value
        samples = samples_by_action[action_id]
        paired = tuple(
            best - candidate for best, candidate in zip(best_samples, samples, strict=True)
        )
        epsilon = max(ABSOLUTE_EPSILON_FLOOR, 2 * _standard_error(paired))
        epsilon_by_action[action_id] = epsilon
        if means[best_action] - means[action_id] <= epsilon:
            epsilon_optimal_actions.append(action_id)
    return {
        "action_values": {
            action.value: {
                "mean": means[action.value],
                "standard_error": _standard_error(samples_by_action[action.value]),
                "samples": samples_by_action[action.value],
            }
            for action in ActionID
        },
        "best_action": best_action,
        "epsilon_by_action": epsilon_by_action,
        "epsilon_optimal_actions": epsilon_optimal_actions,
    }


def _normative_world_job(
    item: tuple[dict[str, Any], int, int],
) -> tuple[str, dict[str, Any]]:
    world, root_seed, rollout_count = item
    state = _market_state(cast(dict[str, Any], world["decision_state"]))
    scenario = ScenarioConfig.model_validate(world["scenario"])
    samples: dict[str, dict[str, list[float]]] = {
        objective: {action.value: [] for action in ActionID} for objective in OBJECTIVES
    }
    market = MarketConfig(initial_cash_multiple=100.0)
    intervention = InterventionConfig()
    dynamics = DynamicsConfig(recovery_periods=20)
    for rollout in range(rollout_count):
        tape = DisturbanceTape(root_seed, f"{scenario.scenario_id}:future:{rollout}")
        for action in ActionID:
            result = run_recovery(
                state,
                scenario,
                action,
                tape,
                market_config=market,
                intervention_config=intervention,
                dynamics=dynamics,
            )
            for objective in OBJECTIVES:
                samples[objective][action.value].append(float(getattr(result.utilities, objective)))
    return str(world["opaque_world_id"]), {
        "schema_version": "janus-hidden-normative-oracle-v1",
        "world_id": world["opaque_world_id"],
        "internal_world_id": world["internal_world_id"],
        "rollout_count": rollout_count,
        "objectives": {
            objective: _objective_summary(
                {action: tuple(values) for action, values in samples[objective].items()}
            )
            for objective in OBJECTIVES
        },
    }


def _verify_normative_artifact(
    project_root: Path,
    world: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    world_id = str(world["opaque_world_id"])
    if (
        payload.get("schema_version") != "janus-hidden-normative-oracle-v1"
        or payload.get("world_id") != world_id
        or payload.get("internal_world_id") != world["internal_world_id"]
        or payload.get("rollout_count") != ROLLOUT_COUNT
        or set(cast(dict[str, Any], payload.get("objectives", {}))) != set(OBJECTIVES)
    ):
        raise ValueError(f"invalid normative oracle artifact: {world_id}")
    primary_path = project_root / "artifacts/hidden-v1" / str(world["oracle_file"])
    primary = _read(primary_path)["oracle"]
    expected = {str(row["action"]): float(row["mean"]) for row in primary["action_values"]}
    objectives = cast(dict[str, Any], payload["objectives"])
    action_values = cast(dict[str, Any], objectives["primary"])["action_values"]
    actual = {
        str(action): float(values["mean"])
        for action, values in cast(dict[str, dict[str, Any]], action_values).items()
    }
    if set(actual) != set(expected) or any(
        abs(actual[action] - expected[action]) > 1e-9 for action in expected
    ):
        raise ValueError(f"primary objective recomputation mismatch: {world_id}")


def generate_normative_oracles(
    project_root: Path,
    output_directory: Path,
    *,
    workers: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Recompute all preregistered utility vectors from the sealed hidden worlds."""

    if workers < 1:
        raise ValueError("workers must be positive")
    if not output_directory.is_absolute():
        output_directory = project_root / output_directory
    private_path = project_root / "artifacts/hidden-v1/private_manifest.json"
    seed_path = project_root / "artifacts/hidden-v1/seed_manifest.json"
    commitment_path = project_root / "manifests/hidden-v1-commitment.json"
    private = _read(private_path)
    seed = _read(seed_path)
    commitment = _read(commitment_path)
    if _sha256(private_path) != commitment["private_manifest_sha256"]:
        raise ValueError("private hidden manifest commitment mismatch")
    if hashlib.sha256(canonical_json(seed).encode()).hexdigest() != commitment["sha256"]:
        raise ValueError("hidden seed commitment mismatch")
    root_seed = int(seed["root_seed"])
    worlds = cast(list[dict[str, Any]], private["worlds"])
    output_directory.mkdir(parents=True, exist_ok=True)
    output_directory.chmod(0o700)

    manifest_path = output_directory / "manifest.json"
    if manifest_path.exists():
        existing_manifest = _read(manifest_path)
        expected_artifacts = cast(dict[str, str], existing_manifest.get("artifacts", {}))
        if (
            existing_manifest.get("completed") is True
            and existing_manifest.get("provider_calls") == 0
            and existing_manifest.get("private_manifest_sha256") == _sha256(private_path)
            and len(expected_artifacts) == 100
            and all(
                (output_directory / name).exists()
                and _sha256(output_directory / name) == expected_hash
                for name, expected_hash in expected_artifacts.items()
            )
        ):
            if progress is not None:
                progress(100, 100)
            return existing_manifest

    hashes: dict[str, str] = {}
    pending_worlds: list[dict[str, Any]] = []
    for world in worlds:
        path = output_directory / f"{world['opaque_world_id']}.json"
        if not path.exists():
            pending_worlds.append(world)
            continue
        payload = _read(path)
        _verify_normative_artifact(project_root, world, payload)
        hashes[path.name] = _sha256(path)
    if progress is not None:
        progress(len(hashes), len(worlds))

    jobs = tuple((world, root_seed, ROLLOUT_COUNT) for world in pending_worlds)
    if workers == 1:
        results: Iterable[tuple[str, dict[str, Any]]] = map(_normative_world_job, jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_normative_world_job, jobs)

    try:
        for world_id, payload in results:
            world = next(item for item in worlds if item["opaque_world_id"] == world_id)
            _verify_normative_artifact(project_root, world, payload)
            path = output_directory / f"{world_id}.json"
            _write(path, payload)
            path.chmod(0o600)
            hashes[path.name] = _sha256(path)
            if progress is not None:
                progress(len(hashes), len(worlds))
    finally:
        if workers != 1:
            executor.shutdown()
    if len(hashes) != 100:
        raise ValueError("normative audit must contain exactly 100 worlds")
    manifest = {
        "schema_version": "janus-hidden-normative-manifest-v1",
        "completed": True,
        "provider_calls": 0,
        "world_count": len(hashes),
        "rollout_count": ROLLOUT_COUNT,
        "root_seed_commitment_sha256": commitment["sha256"],
        "private_manifest_sha256": _sha256(private_path),
        "primary_recomputation_verified": True,
        "objectives": list(OBJECTIVES),
        "artifacts": hashes,
    }
    _write(manifest_path, manifest)
    manifest_path.chmod(0o600)
    return manifest


def _load_partial_policy(path: Path) -> RidgeActionValuePolicy:
    payload = _read(path)
    return RidgeActionValuePolicy(
        feature_mean=tuple(float(value) for value in payload["feature_mean"]),
        feature_scale=tuple(float(value) for value in payload["feature_scale"]),
        coefficients=tuple(tuple(float(value) for value in row) for row in payload["coefficients"]),
        intercepts=tuple(float(value) for value in payload["intercepts"]),
        regularization=float(payload["regularization"]),
    )


def _summarize(
    rows: list[dict[str, Any]],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    utility_fields = (
        "utility",
        "oracle_utility",
        "random_utility",
        "do_nothing_utility",
        "heuristic_utility",
        "partial_information_utility",
    )
    means = {field: float(np.mean([float(row[field]) for row in rows])) for field in utility_fields}
    constant_means = {
        action.value: float(
            np.mean([cast(dict[str, float], row["action_values"])[action.value] for row in rows])
        )
        for action in ActionID
    }
    best_constant_action = max(
        constant_means,
        key=lambda action: (constant_means[action], action),
    )
    return {
        "trials": len(rows),
        "mean_normalized_regret": _group_interval(
            rows,
            lambda row: float(row["normalized_regret"]),
            seed=BOOTSTRAP_SEED + seed_offset,
        ),
        "epsilon_optimal_rate": _group_interval(
            rows,
            lambda row: float(row["epsilon_optimal"]),
            seed=BOOTSTRAP_SEED + seed_offset + 1,
        ),
        "mean_utility_difference_vs_random": _group_interval(
            rows,
            lambda row: float(row["utility"]) - float(row["random_utility"]),
            seed=BOOTSTRAP_SEED + seed_offset + 2,
        ),
        "mean_utility_difference_vs_do_nothing": _group_interval(
            rows,
            lambda row: float(row["utility"]) - float(row["do_nothing_utility"]),
            seed=BOOTSTRAP_SEED + seed_offset + 3,
        ),
        "mean_utility_difference_vs_heuristic": _group_interval(
            rows,
            lambda row: float(row["utility"]) - float(row["heuristic_utility"]),
            seed=BOOTSTRAP_SEED + seed_offset + 4,
        ),
        "mean_utility_difference_vs_partial_information": _group_interval(
            rows,
            lambda row: float(row["utility"]) - float(row["partial_information_utility"]),
            seed=BOOTSTRAP_SEED + seed_offset + 5,
        ),
        "mean_utilities": means,
        "mean_constant_action_utilities": constant_means,
        "best_constant_action": best_constant_action,
        "gap_decomposition": {
            "oracle_minus_partial_information": (
                means["oracle_utility"] - means["partial_information_utility"]
            ),
            "partial_information_minus_llm": (
                means["partial_information_utility"] - means["utility"]
            ),
        },
        "effective_action_counts": dict(Counter(str(row["action"]) for row in rows)),
    }


def run_hidden_sensitivity(
    project_root: Path,
    response_manifest_path: Path,
    normative_directory: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Score valid-only, worst-action, and alternative-objective estimands."""

    paths = [response_manifest_path, normative_directory, output_path]
    response_manifest_path, normative_directory, output_path = [
        path if path.is_absolute() else project_root / path for path in paths
    ]
    response_manifest = _read(response_manifest_path)
    private_path = project_root / "artifacts/hidden-v1/private_manifest.json"
    public_path = project_root / "manifests/hidden-v1-public.json"
    commitment = _read(project_root / "manifests/hidden-v1-commitment.json")
    private = _read(private_path)
    public = _read(public_path)
    normative_manifest_path = normative_directory / "manifest.json"
    normative_manifest = _read(normative_manifest_path)
    if not response_manifest.get("passed") or not response_manifest.get("completed"):
        raise ValueError("response collection did not complete its frozen protocol")
    if response_manifest.get("hidden_labels_or_oracles_accessed") is not False:
        raise ValueError("response collection was not blinded")
    if not normative_manifest.get("completed") or normative_manifest.get("provider_calls") != 0:
        raise ValueError("normative oracle audit is incomplete or non-local")
    if _sha256(private_path) != commitment["private_manifest_sha256"]:
        raise ValueError("private hidden manifest commitment mismatch")
    if normative_manifest["private_manifest_sha256"] != _sha256(private_path):
        raise ValueError("normative oracle private-manifest hash mismatch")
    if _sha256(public_path) != commitment["public_manifest_sha256"]:
        raise ValueError("public hidden manifest commitment mismatch")

    response_cache = project_root / "artifacts/openai-hidden-v1/cache"
    responses: dict[str, dict[str, Any]] = {}
    for name, expected_hash in response_manifest["response_artifact_sha256"].items():
        path = response_cache / name
        if _sha256(path) != expected_hash:
            raise ValueError(f"sealed response hash mismatch: {name}")
        artifact = _read(path)
        responses[str(artifact["logical_trial_id"])] = artifact
    if len(responses) != 300:
        raise ValueError("sensitivity analysis requires exactly 300 sealed responses")

    observations = {
        str(world["world_id"]): canonical_json(world["observation"]).encode()
        for bundle in public["bundles"]
        for world in bundle["worlds"]
    }
    worlds = {str(world["opaque_world_id"]): world for world in private["worlds"]}
    partial = _load_partial_policy(project_root / "artifacts/phase0d-release/partial_policy.json")
    validity = _read(project_root / "artifacts/phase0d-release/validity_report.json")
    heuristic = PublicHeuristicPolicy(**dict(validity["baselines"]["heuristic_parameters"]))
    recovery_periods = int(
        _read(project_root / "artifacts/phase0d-release/run_manifest.json")["dynamics_config"][
            "recovery_periods"
        ]
    )
    normalization_scale = REFERENCE_SURPLUS_PER_PERIOD * recovery_periods

    normative: dict[str, dict[str, Any]] = {}
    for name, expected_hash in normative_manifest["artifacts"].items():
        path = normative_directory / name
        if _sha256(path) != expected_hash:
            raise ValueError(f"normative oracle hash mismatch: {name}")
        artifact = _read(path)
        normative[str(artifact["world_id"])] = artifact
    if set(normative) != set(worlds):
        raise ValueError("normative oracle worlds differ from private manifest")

    objective_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in responses.values():
        world_id = str(response["world_id"])
        world = worlds[world_id]
        valid = response["validation_status"] == "valid"
        frozen_action = str(response["fallback_action"])
        partial_action = partial.act(observations[world_id]).value
        heuristic_action = heuristic.act(observations[world_id]).value
        for objective_index, objective in enumerate(OBJECTIVES):
            oracle = normative[world_id]["objectives"][objective]
            action_values = {
                str(action): float(values["mean"])
                for action, values in oracle["action_values"].items()
            }
            worst_action = min(
                action_values,
                key=lambda action: (action_values[action], action),
            )
            oracle_utility = max(action_values.values())
            base = {
                "bundle_id": world["opaque_bundle_id"],
                "world_id": world_id,
                "replicate_id": response["replicate_id"],
                "validation_status": response["validation_status"],
                "frozen_action": frozen_action,
                "worst_action": worst_action,
                "oracle_utility": oracle_utility,
                "random_utility": float(np.mean(list(action_values.values()))),
                "do_nothing_utility": action_values[ActionID.DO_NOTHING.value],
                "heuristic_utility": action_values[heuristic_action],
                "partial_information_utility": action_values[partial_action],
                "epsilon_optimal_actions": oracle["epsilon_optimal_actions"],
                "action_values": action_values,
                "valid": valid,
                "seed_offset": objective_index * 100,
            }
            objective_rows[objective].append(base)

    analyses: dict[str, Any] = {}
    for objective_index, objective in enumerate(OBJECTIVES):
        base_rows = objective_rows[objective]

        def scored_rows(source_rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
            scored: list[dict[str, Any]] = []
            for base in source_rows:
                if mode == "valid_only" and not base["valid"]:
                    continue
                action = str(base["frozen_action"])
                if mode == "incomplete_as_worst" and not base["valid"]:
                    action = str(base["worst_action"])
                action_values = cast(dict[str, float], base["action_values"])
                utility = action_values[action]
                row = dict(base)
                row.update(
                    {
                        "action": action,
                        "utility": utility,
                        "normalized_regret": (float(base["oracle_utility"]) - utility)
                        / normalization_scale,
                        "epsilon_optimal": action
                        in cast(list[str], base["epsilon_optimal_actions"]),
                    }
                )
                scored.append(row)
            return scored

        seed_base = 0 if objective == "primary" else 1_000 + objective_index * 100
        analyses[objective] = {
            "intention_to_evaluate": _summarize(
                scored_rows(base_rows, "intention_to_evaluate"), seed_offset=seed_base
            ),
            "valid_only": _summarize(
                scored_rows(base_rows, "valid_only"), seed_offset=seed_base + 10
            ),
            "incomplete_as_worst": _summarize(
                scored_rows(base_rows, "incomplete_as_worst"), seed_offset=seed_base + 20
            ),
        }

    report = {
        "schema_version": "janus-hidden-sensitivity-v1",
        "completed": True,
        "provider_calls": 0,
        "response_manifest_sha256": _sha256(response_manifest_path),
        "normative_manifest_sha256": _sha256(normative_manifest_path),
        "primary_recomputation_verified": normative_manifest["primary_recomputation_verified"],
        "verified_normative_artifact_count": len(normative),
        "worlds": len(worlds),
        "trials": len(responses),
        "normalization_scale": normalization_scale,
        "objectives": list(OBJECTIVES),
        "scoring_rules": [
            "intention_to_evaluate",
            "valid_only",
            "incomplete_as_worst",
        ],
        "analyses": analyses,
    }
    _write(output_path, report)
    return report
