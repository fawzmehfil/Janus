"""Reproducible post-seal analysis of the frozen hidden evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from math import log
from pathlib import Path
from typing import Any, cast

import numpy as np

from janus.core.replay import canonical_json
from janus.interventions.actions import ActionID
from janus.policies.heuristic import PublicHeuristicPolicy
from janus.policies.partial_information import RidgeActionValuePolicy

REFERENCE_SURPLUS_PER_PERIOD = 840.0
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261019
FAMILIES = (
    "supply_shortage",
    "demand_increase",
    "seller_collusion",
    "momentum_demand",
)


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interval(values_by_bundle: dict[str, list[float]], *, seed: int) -> dict[str, float | int]:
    bundle_ids = sorted(values_by_bundle)
    if len(bundle_ids) < 2:
        raise ValueError("cluster bootstrap requires at least two bundles")
    observed = float(np.mean([value for values in values_by_bundle.values() for value in values]))
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        selected = rng.integers(0, len(bundle_ids), size=len(bundle_ids))
        sample = [
            value
            for selected_index in selected
            for value in values_by_bundle[bundle_ids[int(selected_index)]]
        ]
        estimates[index] = np.mean(sample)
    return {
        "estimate": observed,
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


def _group_interval(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    seed: int,
) -> dict[str, float | int]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["bundle_id"])].append(value(row))
    return _interval(dict(grouped), seed=seed)


def _stratum_difference(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
) -> dict[str, float | int]:
    grouped: dict[str, dict[str, list[float]]] = {
        "id": defaultdict(list),
        "ood": defaultdict(list),
    }
    for row in rows:
        grouped[str(row["stratum"])][str(row["bundle_id"])].append(value(row))
    ids = sorted(grouped["id"])
    oods = sorted(grouped["ood"])
    rng = np.random.default_rng(BOOTSTRAP_SEED + 20)
    estimates = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        id_selected = rng.integers(0, len(ids), size=len(ids))
        ood_selected = rng.integers(0, len(oods), size=len(oods))
        id_values = [item for selected in id_selected for item in grouped["id"][ids[int(selected)]]]
        ood_values = [
            item for selected in ood_selected for item in grouped["ood"][oods[int(selected)]]
        ]
        estimates[index] = np.mean(ood_values) - np.mean(id_values)
    observed = float(
        np.mean([value(row) for row in rows if row["stratum"] == "ood"])
        - np.mean([value(row) for row in rows if row["stratum"] == "id"])
    )
    return {
        "estimate": observed,
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


def _clustered_confidence_correlation(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["bundle_id"])].append(row)
    bundle_ids = sorted(grouped)

    def correlation(selected_rows: list[dict[str, Any]]) -> float:
        confidence = np.asarray([float(row["confidence"]) for row in selected_rows])
        optimal = np.asarray([float(row["epsilon_optimal"]) for row in selected_rows])
        if confidence.std() < 1e-12 or optimal.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(confidence, optimal)[0, 1])

    observed = correlation(rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 30)
    estimates = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        selected = rng.integers(0, len(bundle_ids), size=len(bundle_ids))
        sampled_rows = [
            row for selected_index in selected for row in grouped[bundle_ids[int(selected_index)]]
        ]
        estimates[index] = correlation(sampled_rows)
    return {
        "estimate": observed,
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


def _load_partial_policy(path: Path) -> RidgeActionValuePolicy:
    payload = _read(path)
    return RidgeActionValuePolicy(
        feature_mean=tuple(float(value) for value in payload["feature_mean"]),
        feature_scale=tuple(float(value) for value in payload["feature_scale"]),
        coefficients=tuple(tuple(float(value) for value in row) for row in payload["coefficients"]),
        intercepts=tuple(float(value) for value in payload["intercepts"]),
        regularization=float(payload["regularization"]),
    )


def run_hidden_analysis(
    project_root: Path,
    response_manifest_path: Path,
    output_path: Path,
    *,
    response_commit: str,
) -> dict[str, Any]:
    """Verify the sealed responses, unblind, score, and write one JSON report."""

    if not response_manifest_path.is_absolute():
        response_manifest_path = project_root / response_manifest_path
    if not output_path.is_absolute():
        output_path = project_root / output_path
    commitment_path = project_root / "manifests/hidden-v1-commitment.json"
    public_path = project_root / "manifests/hidden-v1-public.json"
    private_path = project_root / "artifacts/hidden-v1/private_manifest.json"
    seed_path = project_root / "artifacts/hidden-v1/seed_manifest.json"
    response_cache = project_root / "artifacts/openai-hidden-v1/cache"
    partial_path = project_root / "artifacts/phase0d-release/partial_policy.json"
    validity_path = project_root / "artifacts/phase0d-release/validity_report.json"
    phase0d_manifest_path = project_root / "artifacts/phase0d-release/run_manifest.json"

    commitment = _read(commitment_path)
    public = _read(public_path)
    private = _read(private_path)
    seed = _read(seed_path)
    response_manifest = _read(response_manifest_path)
    if _sha256(public_path) != commitment["public_manifest_sha256"]:
        raise ValueError("public hidden manifest commitment mismatch")
    if _sha256(private_path) != commitment["private_manifest_sha256"]:
        raise ValueError("private hidden manifest commitment mismatch")
    if hashlib.sha256(canonical_json(seed).encode()).hexdigest() != commitment["sha256"]:
        raise ValueError("hidden seed commitment mismatch")
    if not response_manifest.get("passed") or not response_manifest.get("completed"):
        raise ValueError("response collection did not complete its frozen protocol")
    if response_manifest.get("hidden_labels_or_oracles_accessed") is not False:
        raise ValueError("response collection was not blinded")

    response_artifacts: dict[str, dict[str, Any]] = {}
    for name, expected_hash in response_manifest["response_artifact_sha256"].items():
        path = response_cache / name
        if _sha256(path) != expected_hash:
            raise ValueError(f"sealed response hash mismatch: {name}")
        artifact = _read(path)
        logical_id = str(artifact["logical_trial_id"])
        if logical_id in response_artifacts:
            raise ValueError("duplicate sealed logical trial")
        response_artifacts[logical_id] = artifact
    if len(response_artifacts) != 300:
        raise ValueError("sealed response set must contain exactly 300 trials")

    observations = {
        str(world["world_id"]): canonical_json(world["observation"]).encode()
        for bundle in public["bundles"]
        for world in bundle["worlds"]
    }
    private_worlds = {str(world["opaque_world_id"]): world for world in private["worlds"]}
    if set(observations) != set(private_worlds):
        raise ValueError("public and private hidden world IDs differ")

    partial = _load_partial_policy(partial_path)
    validity = _read(validity_path)
    heuristic_parameters = dict(validity["baselines"]["heuristic_parameters"])
    heuristic = PublicHeuristicPolicy(**heuristic_parameters)
    recovery_periods = int(_read(phase0d_manifest_path)["dynamics_config"]["recovery_periods"])
    normalization_scale = REFERENCE_SURPLUS_PER_PERIOD * recovery_periods

    rows: list[dict[str, Any]] = []
    verified_oracles: set[str] = set()
    for artifact in response_artifacts.values():
        world_id = str(artifact["world_id"])
        world = private_worlds[world_id]
        oracle_path = project_root / "artifacts/hidden-v1" / str(world["oracle_file"])
        if _sha256(oracle_path) != world["oracle_sha256"]:
            raise ValueError(f"oracle hash mismatch for {world_id}")
        verified_oracles.add(world_id)
        oracle = _read(oracle_path)["oracle"]
        action_values = {
            str(item["action"]): float(item["mean"]) for item in oracle["action_values"]
        }
        selected_action = str(artifact["fallback_action"])
        decision = artifact["parsed_decision"]
        oracle_utility = max(action_values.values())
        selected_utility = action_values[selected_action]
        partial_action = partial.act(observations[world_id]).value
        heuristic_action = heuristic.act(observations[world_id]).value
        rows.append(
            {
                "bundle_id": world["opaque_bundle_id"],
                "world_id": world_id,
                "replicate_id": artifact["replicate_id"],
                "stratum": world["stratum"],
                "family": world["family"],
                "validation_status": artifact["validation_status"],
                "action": selected_action,
                "diagnosis": None if decision is None else decision["diagnosis"],
                "diagnosis_probabilities": (
                    None if decision is None else decision["diagnosis_probabilities"]
                ),
                "confidence": None if decision is None else float(decision["confidence"]),
                "utility": selected_utility,
                "oracle_utility": oracle_utility,
                "random_utility": float(np.mean(list(action_values.values()))),
                "do_nothing_utility": action_values[ActionID.DO_NOTHING.value],
                "partial_information_utility": action_values[partial_action],
                "heuristic_utility": action_values[heuristic_action],
                "constant_action_utilities": action_values,
                "regret": oracle_utility - selected_utility,
                "normalized_regret": (oracle_utility - selected_utility) / normalization_scale,
                "epsilon_optimal": selected_action in oracle["epsilon_optimal_actions"],
            }
        )
    if len(verified_oracles) != 100:
        raise ValueError("analysis must verify all 100 hidden oracle files")

    valid_rows = [row for row in rows if row["validation_status"] == "valid"]
    status_counts = Counter(str(row["validation_status"]) for row in rows)
    action_counts = Counter(str(row["action"]) for row in rows)
    valid_action_counts = Counter(str(row["action"]) for row in valid_rows)
    utility_names = (
        "utility",
        "oracle_utility",
        "partial_information_utility",
        "heuristic_utility",
        "random_utility",
        "do_nothing_utility",
    )
    mean_utilities = {
        name: float(np.mean([float(row[name]) for row in rows])) for name in utility_names
    }
    mean_constant_action_utilities = {
        action.value: float(
            np.mean(
                [
                    float(cast(dict[str, float], row["constant_action_utilities"])[action.value])
                    for row in rows
                ]
            )
        )
        for action in ActionID
    }

    confusion = {family: {predicted: 0 for predicted in FAMILIES} for family in FAMILIES}
    diagnosis_brier: list[float] = []
    diagnosis_log_loss: list[float] = []
    confidence_brier: list[float] = []
    for row in valid_rows:
        family = str(row["family"])
        diagnosis = str(row["diagnosis"])
        confusion[family][diagnosis] += 1
        probabilities = cast(dict[str, float], row["diagnosis_probabilities"])
        diagnosis_brier.append(
            sum((float(probabilities[label]) - float(label == family)) ** 2 for label in FAMILIES)
        )
        diagnosis_log_loss.append(-log(max(1e-15, float(probabilities[family]))))
        confidence_brier.append((float(row["confidence"]) - float(row["epsilon_optimal"])) ** 2)

    world_actions: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        world_actions[str(row["world_id"])].append(str(row["action"]))
    agreements = []
    entropies = []
    for actions in world_actions.values():
        counts = Counter(actions)
        agreements.append(max(counts.values()) / len(actions))
        entropy_probabilities = [count / len(actions) for count in counts.values()]
        entropies.append(-sum(value * log(value) for value in entropy_probabilities))

    confidence_correlation = _clustered_confidence_correlation(valid_rows)
    diagnosis_accuracy_interval = _group_interval(
        rows,
        lambda row: float(row["diagnosis"] == row["family"]),
        seed=BOOTSTRAP_SEED + 31,
    )

    subgroup: dict[str, Any] = {}
    for field, names in (("stratum", ("id", "ood")), ("family", FAMILIES)):
        subgroup[field] = {}
        for name in names:
            selected = [row for row in rows if row[field] == name]
            subgroup[field][name] = {
                "trials": len(selected),
                "valid_rate": float(
                    np.mean([row["validation_status"] == "valid" for row in selected])
                ),
                "mean_normalized_regret": float(
                    np.mean([float(row["normalized_regret"]) for row in selected])
                ),
                "epsilon_optimal_rate": float(
                    np.mean([float(row["epsilon_optimal"]) for row in selected])
                ),
            }

    regret_sorted = sorted(rows, key=lambda row: float(row["regret"]), reverse=True)
    lower_tail_count = max(1, int(np.ceil(0.10 * len(rows))))
    report: dict[str, Any] = {
        "schema_version": "janus-hidden-analysis-v1",
        "response_commit": response_commit,
        "unblinded_after_response_commit": True,
        "integrity": {
            "response_manifest": str(response_manifest_path.relative_to(project_root)),
            "response_manifest_sha256": _sha256(response_manifest_path),
            "sealed_response_count": len(response_artifacts),
            "verified_oracle_count": len(verified_oracles),
            "public_manifest_sha256": _sha256(public_path),
            "private_manifest_sha256": _sha256(private_path),
            "seed_commitment_verified": True,
        },
        "sample": {
            "worlds": len(world_actions),
            "bundles": len({row["bundle_id"] for row in rows}),
            "replicates_per_world": 3,
            "trials": len(rows),
            "status_counts": dict(status_counts),
            "valid_rate": len(valid_rows) / len(rows),
            "normalization_scale": normalization_scale,
        },
        "cost": {
            "incremental_usd": response_manifest["invocation_cost_usd"],
            "cumulative_usd": response_manifest["cumulative_cost_usd"],
            "approved_cumulative_cap_usd": response_manifest["local_hard_ceiling_usd"],
        },
        "primary": {
            "mean_normalized_regret": _group_interval(
                rows, lambda row: float(row["normalized_regret"]), seed=BOOTSTRAP_SEED
            ),
            "median_normalized_regret": float(
                np.median([float(row["normalized_regret"]) for row in rows])
            ),
            "epsilon_optimal_rate": _group_interval(
                rows, lambda row: float(row["epsilon_optimal"]), seed=BOOTSTRAP_SEED + 1
            ),
            "mean_utility_difference_vs_random": _group_interval(
                rows,
                lambda row: float(row["utility"]) - float(row["random_utility"]),
                seed=BOOTSTRAP_SEED + 2,
            ),
            "mean_utility_difference_vs_do_nothing": _group_interval(
                rows,
                lambda row: float(row["utility"]) - float(row["do_nothing_utility"]),
                seed=BOOTSTRAP_SEED + 3,
            ),
            "mean_utility_difference_vs_heuristic": _group_interval(
                rows,
                lambda row: float(row["utility"]) - float(row["heuristic_utility"]),
                seed=BOOTSTRAP_SEED + 4,
            ),
            "mean_utility_difference_vs_partial_information": _group_interval(
                rows,
                lambda row: float(row["utility"]) - float(row["partial_information_utility"]),
                seed=BOOTSTRAP_SEED + 5,
            ),
            "utility_cvar_worst_10_percent": float(
                np.mean(sorted(float(row["utility"]) for row in rows)[:lower_tail_count])
            ),
        },
        "mean_utilities": mean_utilities,
        "mean_constant_action_utilities": mean_constant_action_utilities,
        "gap_decomposition": {
            "estimated_information_gap_oracle_minus_partial": (
                mean_utilities["oracle_utility"] - mean_utilities["partial_information_utility"]
            ),
            "decision_gap_partial_minus_llm": (
                mean_utilities["partial_information_utility"] - mean_utilities["utility"]
            ),
        },
        "subgroups": subgroup,
        "ood_minus_id_mean_normalized_regret": _stratum_difference(
            rows, lambda row: float(row["normalized_regret"])
        ),
        "diagnosis": {
            "accuracy_all_trials_invalid_as_incorrect": float(
                np.mean([row["diagnosis"] == row["family"] for row in rows])
            ),
            "accuracy_all_trials_cluster_interval": diagnosis_accuracy_interval,
            "accuracy_valid_only": float(
                np.mean([row["diagnosis"] == row["family"] for row in valid_rows])
            ),
            "confusion_valid_only": confusion,
            "multiclass_brier_valid_only": float(np.mean(diagnosis_brier)),
            "log_loss_valid_only": float(np.mean(diagnosis_log_loss)),
        },
        "behavior": {
            "action_counts_all": dict(action_counts),
            "action_counts_valid_only": dict(valid_action_counts),
            "mean_within_world_action_agreement": float(np.mean(agreements)),
            "unanimous_world_rate": float(np.mean([value == 1.0 for value in agreements])),
            "mean_action_entropy_nats": float(np.mean(entropies)),
            "action_confidence_brier_valid_only": float(np.mean(confidence_brier)),
            "confidence_epsilon_optimal_cluster_interval": confidence_correlation,
        },
        "highest_regret_trials": [
            {
                "world_id": row["world_id"],
                "replicate_id": row["replicate_id"],
                "family": row["family"],
                "stratum": row["stratum"],
                "action": row["action"],
                "validation_status": row["validation_status"],
                "regret": row["regret"],
                "normalized_regret": row["normalized_regret"],
            }
            for row in regret_sorted[:5]
        ],
    }
    primary = report["primary"]
    report["hypotheses"] = {
        "H1_better_than_uniform_random": primary["mean_utility_difference_vs_random"]["lower_95"]
        > 0,
        "H2_better_than_do_nothing": primary["mean_utility_difference_vs_do_nothing"]["lower_95"]
        > 0,
        "H3_ood_increases_regret": report["ood_minus_id_mean_normalized_regret"]["lower_95"] > 0,
        "H4_confidence_positive_association": (float(confidence_correlation["lower_95"]) > 0),
        "H5_diagnosis_above_chance_all_trials": float(diagnosis_accuracy_interval["lower_95"])
        > 0.25,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    )
    return report
