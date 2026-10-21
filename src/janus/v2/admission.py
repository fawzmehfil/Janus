"""Cross-model development admission gate for exact frozen settings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from janus.v2.contracts import AdmissionThresholds, evaluate_admission

DEFAULT_THRESHOLDS = AdmissionThresholds()


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def build_provider_admission(
    manifest_paths: tuple[Path, ...],
    output_path: Path,
    *,
    thresholds: AdmissionThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    if len(manifest_paths) != 4:
        raise ValueError("exactly four development response manifests are required")
    manifests = [_read(path) for path in manifest_paths]
    trial_sets = [
        {(str(row["world_id"]), str(row["replicate_id"])) for row in manifest["trials"]}
        for manifest in manifests
    ]
    shared_trials = all(values == trial_sets[0] for values in trial_sets[1:])
    prompt_hashes = {str(value["prompt_sha256"]) for value in manifests}
    schema_hashes = {str(value["schema_sha256"]) for value in manifests}
    roles = {str(value["model_role"]) for value in manifests}
    requested_ids = {str(value["requested_model_id"]) for value in manifests}
    models = []
    for path, manifest in zip(manifest_paths, manifests, strict=True):
        statuses = tuple(str(row["validation_status"]) for row in manifest["trials"])
        result = evaluate_admission(str(manifest["model_role"]), statuses, thresholds)
        model_row = result.model_dump(mode="json")
        identity_match = set(manifest["returned_model_ids"]) == {
            str(manifest["requested_model_id"])
        }
        model_row.update(
            {
                "model_role": manifest["model_role"],
                "serving_provider": manifest["provider"],
                "manifest": str(path),
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "requested_model_id": manifest["requested_model_id"],
                "returned_model_ids": manifest["returned_model_ids"],
                "model_identity_match": identity_match,
                "sdk_version": manifest["sdk_version"],
                "endpoint": manifest["endpoint"],
                "maximum_output_tokens": manifest["maximum_output_tokens"],
                "paid_api_calls": manifest["provider_calls"],
                "cost_usd": manifest["cost_usd"],
            }
        )
        models.append(model_row)
    passed = (
        shared_trials
        and len(prompt_hashes) == 1
        and len(schema_hashes) == 1
        and len(trial_sets[0]) == 100
        and len(roles) == 4
        and len(requested_ids) == 4
        and all(row["admitted"] and row["model_identity_match"] for row in models)
    )
    report = {
        "schema_version": "janus-v2-model-admission-v2",
        "thresholds": thresholds.model_dump(mode="json"),
        "same_representative_trials": shared_trials,
        "identical_prompt_hash": len(prompt_hashes) == 1,
        "identical_schema_hash": len(schema_hashes) == 1,
        "exactly_100_trials_per_model": all(len(values) == 100 for values in trial_sets),
        "four_unique_model_roles": len(roles) == 4,
        "four_unique_model_ids": len(requested_ids) == 4,
        "models": models,
        "passed": passed,
        "model_freeze_authorized": passed,
        "hidden_evaluation_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report
