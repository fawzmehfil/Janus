"""Four-manifest integrity verification, unblinding, and v2 scoring."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

from janus.core.replay import canonical_json
from janus.policies.heuristic import PublicHeuristicPolicy
from janus.v2.baselines import FrozenPublicPolicy
from janus.v2.statistics import analyze_comparative_rows


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _committed(project_root: Path, commit: str, path: Path) -> bool:
    relative = path.resolve().relative_to(project_root.resolve())
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{relative}"],
            cwd=project_root,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _action_rows(oracle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["action"]): row for row in oracle["action_outcomes"]}


def _economic_outcome(
    actions: dict[str, dict[str, Any]], economic_keys: set[str], action: str
) -> dict[str, float]:
    return {key: float(actions[action]["means"][key]) for key in economic_keys}


def _scored_action(artifact: dict[str, Any], actions: dict[str, dict[str, Any]]) -> str:
    """Apply the frozen intention-to-evaluate action rule without using diagnosis fields."""

    decision = artifact.get("parsed_decision")
    if artifact["validation_status"] == "valid" and isinstance(decision, dict):
        return str(decision["action"])
    return min(
        actions,
        key=lambda action: (float(actions[action]["means"]["primary"]), action),
    )


def run_v2_unblinding(
    project_root: Path,
    response_manifests: tuple[Path, ...],
    output_path: Path,
    *,
    response_commit: str,
) -> dict[str, Any]:
    if len(response_manifests) != 4:
        raise ValueError("unblinding requires exactly four response manifests")
    if not all(_committed(project_root, response_commit, path) for path in response_manifests):
        raise ValueError("all four response manifests must be committed before unblinding")
    manifests = [_read(path) for path in response_manifests]
    if any(not row.get("completed") or row.get("unblinded", True) for row in manifests):
        raise ValueError("every response manifest must be complete and explicitly blinded")
    trial_sets = [
        {(str(row["world_id"]), str(row["replicate_id"])) for row in manifest["trials"]}
        for manifest in manifests
    ]
    if not all(values == trial_sets[0] for values in trial_sets[1:]):
        raise ValueError("model response manifests do not contain identical sealed trials")

    commitment_path = project_root / "manifests/hidden-v2-commitment.json"
    public_path = project_root / "manifests/hidden-v2-public.json"
    private_path = project_root / "artifacts/hidden-v2/private_manifest.json"
    seed_path = project_root / "artifacts/hidden-v2/seed_manifest.json"
    commitment = _read(commitment_path)
    public = _read(public_path)
    private = _read(private_path)
    seed = _read(seed_path)
    if _sha256(public_path) != commitment["public_manifest_sha256"]:
        raise ValueError("hidden-v2 public manifest commitment mismatch")
    if _sha256(private_path) != commitment["private_manifest_sha256"]:
        raise ValueError("hidden-v2 private manifest commitment mismatch")
    if hashlib.sha256(canonical_json(seed).encode()).hexdigest() != commitment["sha256"]:
        raise ValueError("hidden-v2 seed commitment mismatch")
    if any(manifest["public_manifest_sha256"] != _sha256(public_path) for manifest in manifests):
        raise ValueError("a model used a different hidden-v2 public manifest")

    observations = {
        str(world["world_id"]): canonical_json(world["observation"]).encode()
        for bundle in public["bundles"]
        for world in bundle["worlds"]
    }
    private_worlds = {str(row["opaque_world_id"]): row for row in private["worlds"]}
    objectives = (
        "primary",
        "consumer_protection",
        "producer_continuity",
        "stability_aware",
        "fiscal_neutral",
    )
    baselines = {
        objective: FrozenPublicPolicy.load(
            project_root / f"manifests/v2/baselines/{objective}.json"
        )
        for objective in objectives
    }
    v1_validity = _read(project_root / "artifacts/phase0d-release/validity_report.json")
    heuristic = PublicHeuristicPolicy(**dict(v1_validity["baselines"]["heuristic_parameters"]))
    rows = []
    response_hashes = {}
    for manifest_path, manifest in zip(response_manifests, manifests, strict=True):
        cache = project_root / str(manifest["cache_directory"])
        artifacts = {}
        for name, expected in manifest["response_artifact_sha256"].items():
            path = cache / name
            if _sha256(path) != expected:
                raise ValueError(f"sealed v2 response hash mismatch: {name}")
            artifact = _read(path)
            artifacts[(str(artifact["world_id"]), str(artifact["replicate_id"]))] = artifact
        response_hashes[str(manifest["model_role"])] = _sha256(manifest_path)
        for trial_key in sorted(trial_sets[0]):
            artifact = artifacts[trial_key]
            world_id = trial_key[0]
            world = private_worlds[world_id]
            oracle_path = project_root / "artifacts/hidden-v2" / str(world["oracle_file"])
            if _sha256(oracle_path) != world["oracle_sha256"]:
                raise ValueError(f"hidden-v2 oracle hash mismatch: {world_id}")
            oracle = _read(oracle_path)["oracle"]
            actions = _action_rows(oracle)
            economic_keys = {
                key for value in actions.values() for key in value["means"] if key not in objectives
            }

            public_action = baselines["primary"].act(observations[world_id]).value
            heuristic_action = heuristic.act(observations[world_id]).value
            decision = artifact.get("parsed_decision")
            selected_action = _scored_action(artifact, actions)
            selected = actions[selected_action]
            oracle_utility = max(float(value["means"]["primary"]) for value in actions.values())
            epsilon_optimal = selected_action in oracle["epsilon_optimal_actions"]
            rows.append(
                {
                    "model_id": str(manifest["requested_model_id"]),
                    "model_family": str(manifest["provider"]),
                    "model_role": str(manifest["model_role"]),
                    "bundle_id": str(world["opaque_bundle_id"]),
                    "world_id": world_id,
                    "replicate_id": trial_key[1],
                    "family": str(world["family"]),
                    "stratum": str(world["stratum"]),
                    "validation_status": str(artifact["validation_status"]),
                    "selected_action": selected_action,
                    "oracle_optimal_action": str(oracle["best_action"]),
                    "public_policy_action": public_action,
                    "heuristic_action": heuristic_action,
                    "model_utility": float(selected["means"]["primary"]),
                    "public_policy_utility": float(actions[public_action]["means"]["primary"]),
                    "oracle_utility": oracle_utility,
                    "heuristic_utility": float(actions[heuristic_action]["means"]["primary"]),
                    "random_utility": float(
                        sum(float(value["means"]["primary"]) for value in actions.values())
                        / len(actions)
                    ),
                    "do_nothing_utility": float(actions["do_nothing"]["means"]["primary"]),
                    "constant_action_utilities": {
                        action: float(value["means"]["primary"])
                        for action, value in actions.items()
                    },
                    "objective_comparisons": {
                        objective: {
                            "model_utility": float(selected["means"][objective]),
                            "public_policy_action": (
                                objective_action := baselines[objective]
                                .act(observations[world_id])
                                .value
                            ),
                            "public_policy_utility": float(
                                actions[objective_action]["means"][objective]
                            ),
                            "public_policy_economic_outcome": _economic_outcome(
                                actions, economic_keys, objective_action
                            ),
                            "oracle_utility": max(
                                float(value["means"][objective]) for value in actions.values()
                            ),
                        }
                        for objective in objectives
                    },
                    "epsilon_optimal": epsilon_optimal,
                    "diagnosis_probabilities": (
                        None
                        if not isinstance(decision, dict)
                        else decision["diagnosis_probabilities"]
                    ),
                    "confidence": (
                        None if not isinstance(decision, dict) else decision["confidence"]
                    ),
                    "economic_outcome": _economic_outcome(actions, economic_keys, selected_action),
                    "public_policy_economic_outcome": _economic_outcome(
                        actions, economic_keys, public_action
                    ),
                    "heuristic_economic_outcome": _economic_outcome(
                        actions, economic_keys, heuristic_action
                    ),
                    "do_nothing_economic_outcome": _economic_outcome(
                        actions, economic_keys, "do_nothing"
                    ),
                    "uniform_random_economic_outcome": {
                        key: float(
                            sum(float(value["means"][key]) for value in actions.values())
                            / len(actions)
                        )
                        for key in economic_keys
                    },
                    "constant_action_economic_outcomes": {
                        action: _economic_outcome(actions, economic_keys, action)
                        for action in actions
                    },
                    "usage": artifact.get("usage", {}),
                    "latency_ms": artifact.get("latency_ms"),
                    "actual_cost_usd": artifact.get("actual_cost_usd", 0),
                }
            )
    report = analyze_comparative_rows(rows)
    report.update(
        {
            "response_commit": response_commit,
            "unblinded_after_response_commit": True,
            "response_manifest_sha256": response_hashes,
            "hidden_commitment_sha256": _sha256(commitment_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    rows_path = output_path.with_name("trial-rows.jsonl")
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return report
