"""Provider-neutral, resumable response collection for a frozen v2 manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from janus.evaluation.budget import BudgetLedger
from janus.evaluation.harness import PromptRenderer
from janus.models.base import ModelAdapter, ModelSettings, TrialContext
from janus.storage.cache import FileTrialCache
from janus.v2 import BENCHMARK_VERSION
from janus.v2.harness import V2EvaluationRunner


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def run_v2_collection(
    project_root: Path,
    public_manifest_path: Path,
    output_directory: Path,
    *,
    model_role: str,
    adapter: ModelAdapter,
    settings: ModelSettings,
    budget: BudgetLedger,
) -> tuple[bool, Path]:
    """Collect one model role; callers own credentials and explicit approval."""

    public = _read(public_manifest_path)
    if public.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError("collection requires the frozen hidden-v2 public manifest")
    prompt_path = project_root / "configs/prompts/primary-v2.txt"
    prompt = PromptRenderer(prompt_path.read_text(), version="primary-v2")
    cache_directory = output_directory / "cache"
    record_directory = output_directory / "records"
    record_directory.mkdir(parents=True, exist_ok=True)
    runner = V2EvaluationRunner(
        adapter=adapter,
        settings=settings,
        prompt=prompt,
        cache=FileTrialCache(cache_directory),
        budget=budget,
    )
    started = datetime.now(UTC).isoformat()
    artifacts = []
    for bundle in public["bundles"]:
        for world in bundle["worlds"]:
            observation = json.dumps(
                world["observation"], sort_keys=True, separators=(",", ":")
            ).encode()
            for replicate_id in world["replicate_ids"]:
                artifact = runner.run(
                    observation,
                    TrialContext(
                        world_id=str(world["world_id"]),
                        replicate_id=str(replicate_id),
                        benchmark_version=BENCHMARK_VERSION,
                        stage="hidden-v2",
                    ),
                )
                record_path = record_directory / f"{artifact['cache_key']}.json"
                encoded = json.dumps(artifact, sort_keys=True, indent=2) + "\n"
                if record_path.exists() and record_path.read_text() != encoded:
                    raise ValueError("immutable response record collision")
                record_path.write_text(encoded)
                artifacts.append(artifact)
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(record_directory.glob("*.json"))
    }
    planned = int(public["trials_per_model"])
    completed = len(artifacts) == planned
    manifest = {
        "schema_version": "janus-v2-response-manifest-v1",
        "model_role": model_role,
        "provider": settings.provider,
        "requested_model_id": settings.requested_model_id,
        "returned_model_ids": sorted({str(row["returned_model_id"]) for row in artifacts}),
        "sdk_version": settings.sdk_version,
        "endpoint": settings.endpoint,
        "maximum_output_tokens": settings.maximum_output_tokens,
        "prompt_sha256": prompt.hash,
        "schema_sha256": str(public["decision_schema_sha256"]),
        "public_manifest_sha256": hashlib.sha256(public_manifest_path.read_bytes()).hexdigest(),
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "provider_calls": sum(row["validation_status"] != "budget_stop" for row in artifacts),
        "cost_usd": budget.total_spent_usd,
        "local_hard_ceiling_usd": budget.total_ceiling_usd,
        "completed": completed,
        "passed": completed,
        "unblinded": False,
        "private_hidden_artifacts_accessed": False,
        "retry_count": 0,
        "response_artifact_sha256": hashes,
        "cache_directory": str(record_directory.relative_to(project_root)),
        "trials": [
            {
                "world_id": row["world_id"],
                "replicate_id": row["replicate_id"],
                "logical_trial_id": row["logical_trial_id"],
                "cache_key": row["cache_key"],
                "validation_status": row["validation_status"],
            }
            for row in artifacts
        ],
    }
    manifest_directory = output_directory / "manifests"
    manifest_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_directory / "run-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return completed, manifest_path
