"""Guarded OpenAI collection for the frozen 300-request hidden evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from janus.core.replay import canonical_json
from janus.evaluation.budget import (
    BudgetExceeded,
    BudgetLedger,
    PricingConfig,
    conservative_usage_estimate,
)
from janus.evaluation.harness import (
    EvaluationRunner,
    PromptRenderer,
    decision_schema_bytes,
)
from janus.evaluation.observations import PublicObservationHistory, serialize_public_history
from janus.evaluation.openai_smoke import (
    CORRECTED_SMOKE_STAGE,
    DIAGNOSTIC_STAGE,
    SMOKE_MODEL,
    SMOKE_REASONING_EFFORT,
    SMOKE_STAGE,
    CountingTransport,
    diagnostic_settings,
)
from janus.models.base import ResponseStatus, TrialContext
from janus.models.providers.openai import OpenAIResponsesAdapter, OpenAITransport
from janus.storage.cache import FileTrialCache

HIDDEN_STAGE = "openai-hidden-v1"
HIDDEN_WORLD_COUNT = 100
HIDDEN_REPLICATES = 3
HIDDEN_TRIAL_COUNT = HIDDEN_WORLD_COUNT * HIDDEN_REPLICATES
HIDDEN_MAXIMUM_OUTPUT_TOKENS = 2_048
HIDDEN_MAXIMUM_CUMULATIVE_CEILING_USD = 1.00
RECOMMENDED_CUMULATIVE_CEILING_USD = 0.84
CORRECTED_SMOKE_MANIFEST = Path(
    "artifacts/openai-smoke-2048-v2/manifests/run-20261019T065249.067530Z.json"
)


@dataclass(frozen=True, slots=True)
class HiddenInput:
    bundle_id: str
    world_id: str
    observation: bytes
    observation_sha256: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _write_manifest(output_directory: Path, payload: dict[str, Any]) -> Path:
    manifests = output_directory / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = manifests / f"run-{stamp}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)
    return path


def validate_hidden_ceiling(ceiling_usd: float) -> None:
    if not 0 < ceiling_usd <= HIDDEN_MAXIMUM_CUMULATIVE_CEILING_USD:
        raise ValueError(
            "hidden cumulative ceiling must be greater than zero and at most "
            f"${HIDDEN_MAXIMUM_CUMULATIVE_CEILING_USD:.2f}"
        )
    if round(ceiling_usd, 2) != ceiling_usd:
        raise ValueError("hidden cumulative ceiling must be stated in whole cents")


def hidden_confirmation(ceiling_usd: float) -> str:
    validate_hidden_ceiling(ceiling_usd)
    return f"HIDDEN_V1_CUMULATIVE_USD_{ceiling_usd:.2f}".replace(".", "_")


def _hidden_inputs(public_manifest_path: Path) -> tuple[HiddenInput, ...]:
    payload = _json_object(public_manifest_path)
    if payload.get("schema_version") != "janus-hidden-public-v1":
        raise ValueError("unexpected hidden public-manifest schema")
    if payload.get("benchmark_version") != "janus-v1":
        raise ValueError("unexpected hidden benchmark version")
    if payload.get("bundle_count") != 25 or payload.get("world_count") != HIDDEN_WORLD_COUNT:
        raise ValueError("hidden public manifest must contain 25 bundles and 100 worlds")
    bundles = payload.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != 25:
        raise ValueError("hidden public manifest has an invalid bundle list")

    inputs: list[HiddenInput] = []
    bundle_ids: set[str] = set()
    world_ids: set[str] = set()
    for bundle in bundles:
        if not isinstance(bundle, dict) or set(bundle) != {"bundle_id", "worlds"}:
            raise ValueError("public bundle contains unexpected or missing fields")
        bundle_id = str(bundle["bundle_id"])
        worlds = bundle["worlds"]
        if bundle_id in bundle_ids or not isinstance(worlds, list) or len(worlds) != 4:
            raise ValueError("hidden bundle IDs must be unique and contain four worlds")
        bundle_ids.add(bundle_id)
        for world in worlds:
            if not isinstance(world, dict) or set(world) != {
                "world_id",
                "observation",
                "observation_sha256",
            }:
                raise ValueError("public world contains unexpected or missing fields")
            world_id = str(world["world_id"])
            if world_id in world_ids:
                raise ValueError("hidden world IDs must be unique")
            world_ids.add(world_id)
            observation = PublicObservationHistory.model_validate(world["observation"])
            serialized = serialize_public_history(observation)
            observation_sha256 = str(world["observation_sha256"])
            if _sha256(serialized) != observation_sha256:
                raise ValueError(f"observation hash mismatch for {world_id}")
            inputs.append(
                HiddenInput(
                    bundle_id=bundle_id,
                    world_id=world_id,
                    observation=serialized,
                    observation_sha256=observation_sha256,
                )
            )
    if len(inputs) != HIDDEN_WORLD_COUNT:
        raise ValueError("hidden public manifest must contain exactly 100 unique worlds")
    return tuple(inputs)


def _components(
    project_root: Path,
    sdk_version: str,
) -> tuple[PromptRenderer, PricingConfig, Path, Path, Path, Path]:
    prompt_path = project_root / "configs/prompts/primary-v1.txt"
    pricing_path = project_root / "configs/pricing/openai-gpt-6-luna-2026-10-17.json"
    public_manifest_path = project_root / "manifests/hidden-v1-public.json"
    commitment_path = project_root / "manifests/hidden-v1-commitment.json"
    prompt = PromptRenderer(prompt_path.read_text(), version="primary-v1")
    pricing = PricingConfig.model_validate(_json_object(pricing_path))
    settings = diagnostic_settings(sdk_version)
    if pricing.model_id != settings.requested_model_id:
        raise ValueError("pricing model and requested hidden model differ")
    if settings.maximum_output_tokens != HIDDEN_MAXIMUM_OUTPUT_TOKENS:
        raise ValueError("hidden output ceiling differs from the corrected smoke setting")
    return prompt, pricing, prompt_path, pricing_path, public_manifest_path, commitment_path


def _validate_release_gate(
    project_root: Path,
    prompt_path: Path,
    public_manifest_path: Path,
    commitment_path: Path,
) -> dict[str, Any]:
    public = _json_object(public_manifest_path)
    commitment = _json_object(commitment_path)
    public_hash = _sha256(public_manifest_path.read_bytes())
    prompt_hash = _sha256(prompt_path.read_bytes())
    schema_hash = _sha256(decision_schema_bytes())
    if commitment.get("public_manifest_sha256") != public_hash:
        raise ValueError("hidden public manifest no longer matches its commitment")
    if public.get("code_commit") != commitment.get("code_commit"):
        raise ValueError("hidden public manifest and commitment revisions differ")
    if public.get("prompt_sha256") != prompt_hash:
        raise ValueError("frozen hidden prompt hash does not match the local prompt")
    if public.get("decision_schema_sha256") != schema_hash:
        raise ValueError("frozen hidden schema hash does not match the local schema")

    smoke_path = project_root / CORRECTED_SMOKE_MANIFEST
    smoke = _json_object(smoke_path)
    settings = smoke.get("settings", {})
    statuses = smoke.get("status_counts", {})
    gate_passed = (
        smoke.get("passed") is True
        and smoke.get("completed") is True
        and smoke.get("hidden_artifacts_accessed") is False
        and isinstance(settings, dict)
        and settings.get("requested_model_id") == SMOKE_MODEL
        and settings.get("reasoning_effort") == SMOKE_REASONING_EFFORT
        and settings.get("maximum_output_tokens") == HIDDEN_MAXIMUM_OUTPUT_TOKENS
        and smoke.get("prompt_sha256") == prompt_hash
        and smoke.get("schema_sha256") == schema_hash
        and isinstance(statuses, dict)
        and statuses.get(ResponseStatus.VALID.value) == 5
        and sum(int(value) for value in statuses.values()) == 5
    )
    if not gate_passed:
        raise ValueError("the committed corrected five-world smoke gate is not valid")
    return {
        "passed": True,
        "manifest": str(CORRECTED_SMOKE_MANIFEST),
        "manifest_sha256": _sha256(smoke_path.read_bytes()),
        "public_manifest_sha256": public_hash,
        "frozen_code_commit": public["code_commit"],
    }


def _cache_artifacts(cache_directory: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(cache_directory.glob("*.json")):
        payload = _json_object(path)
        if payload.get("stage") != HIDDEN_STAGE:
            continue
        logical_id = payload.get("logical_trial_id")
        if not isinstance(logical_id, str) or logical_id in artifacts:
            raise ValueError("hidden cache contains a missing or duplicate logical trial ID")
        artifacts[logical_id] = payload
    return artifacts


def _recorded_spend(cache_directory: Path, stage: str) -> float:
    total = 0.0
    for path in sorted(cache_directory.glob("*.json")):
        payload = _json_object(path)
        if payload.get("stage") != stage:
            continue
        cost = payload.get("actual_cost_usd")
        if not isinstance(cost, int | float) or float(cost) < 0:
            raise ValueError(f"invalid cached cost in {path}")
        total += float(cost)
    return total


def _spend_breakdown(project_root: Path, output_directory: Path) -> dict[str, float]:
    return {
        SMOKE_STAGE: _recorded_spend(project_root / "artifacts/openai-smoke-v1/cache", SMOKE_STAGE),
        DIAGNOSTIC_STAGE: _recorded_spend(
            project_root / "artifacts/openai-diagnostic-2048-v1/cache", DIAGNOSTIC_STAGE
        ),
        CORRECTED_SMOKE_STAGE: _recorded_spend(
            project_root / "artifacts/openai-smoke-2048-v2/cache", CORRECTED_SMOKE_STAGE
        ),
        HIDDEN_STAGE: _recorded_spend(output_directory / "cache", HIDDEN_STAGE),
    }


def _logical_trial_id(world_id: str, replicate_index: int) -> str:
    return f"{world_id}:hidden-replicate-{replicate_index:02d}"


def hidden_preflight(
    project_root: Path,
    output_directory: Path,
    *,
    ceiling_usd: float = RECOMMENDED_CUMULATIVE_CEILING_USD,
    sdk_version: str = "preflight-only",
) -> dict[str, Any]:
    """Validate and price the hidden collection without a key or network access."""

    validate_hidden_ceiling(ceiling_usd)
    (
        prompt,
        pricing,
        prompt_path,
        pricing_path,
        public_manifest_path,
        commitment_path,
    ) = _components(project_root, sdk_version)
    release_gate = _validate_release_gate(
        project_root,
        prompt_path,
        public_manifest_path,
        commitment_path,
    )
    inputs = _hidden_inputs(public_manifest_path)
    cached = _cache_artifacts(output_directory / "cache")
    settings = diagnostic_settings(sdk_version)

    plan: list[dict[str, Any]] = []
    remaining_upper_bound = 0.0
    min_request = float("inf")
    max_request = 0.0
    for item in inputs:
        input_bytes = len(prompt.render(item.observation)) + len(decision_schema_bytes())
        usage = conservative_usage_estimate(input_bytes, settings.maximum_output_tokens)
        estimate = pricing.cost(usage)
        min_request = min(min_request, estimate)
        max_request = max(max_request, estimate)
        for replicate_index in range(HIDDEN_REPLICATES):
            logical_id = _logical_trial_id(item.world_id, replicate_index)
            is_cached = logical_id in cached
            if not is_cached:
                remaining_upper_bound += estimate
            plan.append(
                {
                    "logical_trial_id": logical_id,
                    "bundle_id": item.bundle_id,
                    "world_id": item.world_id,
                    "observation_sha256": item.observation_sha256,
                    "input_token_upper_bound": usage.input_tokens,
                    "output_token_ceiling": usage.output_tokens,
                    "request_cost_upper_bound_usd": estimate,
                    "already_collected": is_cached,
                }
            )

    planned_ids = {str(item["logical_trial_id"]) for item in plan}
    unexpected_cached_ids = sorted(set(cached).difference(planned_ids))
    if unexpected_cached_ids:
        raise ValueError("hidden cache contains logical trials outside the frozen plan")
    spend = _spend_breakdown(project_root, output_directory)
    cumulative_spend = sum(spend.values())
    cumulative_upper_bound = cumulative_spend + remaining_upper_bound
    return {
        "schema_version": "janus-openai-hidden-preflight-v1",
        "network_calls": 0,
        "paid_api_calls": 0,
        "public_hidden_observations_accessed": True,
        "private_hidden_artifacts_accessed": False,
        "hidden_labels_or_oracles_accessed": False,
        "world_count": len(inputs),
        "replicates_per_world": HIDDEN_REPLICATES,
        "trial_count": len(plan),
        "trial_count_already_collected": len(cached),
        "trial_count_remaining": len(plan) - len(cached),
        "repairs_allowed": False,
        "maximum_provider_calls_remaining": len(plan) - len(cached),
        "ceiling_usd": ceiling_usd,
        "recommended_cumulative_ceiling_usd": RECOMMENDED_CUMULATIVE_CEILING_USD,
        "recorded_spend_by_stage_usd": spend,
        "cumulative_spend_before_usd": cumulative_spend,
        "remaining_primary_request_upper_bound_usd": remaining_upper_bound,
        "cumulative_upper_bound_usd": cumulative_upper_bound,
        "fits_ceiling": cumulative_upper_bound <= ceiling_usd + 1e-12,
        "minimum_request_upper_bound_usd": min_request,
        "maximum_request_upper_bound_usd": max_request,
        "required_confirmation": hidden_confirmation(ceiling_usd),
        "settings": settings.model_dump(mode="json"),
        "prompt_sha256": _sha256(prompt_path.read_bytes()),
        "schema_sha256": _sha256(decision_schema_bytes()),
        "pricing_sha256": _sha256(pricing_path.read_bytes()),
        "pricing_source_url": pricing.source_url,
        "release_gate": release_gate,
        "trial_plan_sha256": _sha256(canonical_json(plan).encode()),
    }


def _validate_cached_artifact(
    artifact: dict[str, Any],
    item: HiddenInput,
    replicate_id: str,
    prompt_hash: str,
    schema_hash: str,
) -> None:
    settings = artifact.get("model_settings", {})
    if not isinstance(settings, dict):
        raise ValueError("cached hidden artifact has invalid settings")
    expected = {
        "stage": HIDDEN_STAGE,
        "world_id": item.world_id,
        "replicate_id": replicate_id,
        "benchmark_version": "janus-v1",
        "requested_model_id": SMOKE_MODEL,
        "prompt_sha256": prompt_hash,
        "schema_sha256": schema_hash,
    }
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError("cached hidden artifact does not match the frozen trial")
    if (
        settings.get("maximum_output_tokens") != HIDDEN_MAXIMUM_OUTPUT_TOKENS
        or settings.get("reasoning_effort") != SMOKE_REASONING_EFFORT
    ):
        raise ValueError("cached hidden artifact has non-frozen model settings")


def run_openai_hidden(
    project_root: Path,
    output_directory: Path,
    *,
    transport: OpenAITransport,
    sdk_version: str,
    ceiling_usd: float = RECOMMENDED_CUMULATIVE_CEILING_USD,
) -> tuple[bool, Path]:
    """Collect or safely resume all frozen hidden responses without unblinding."""

    validate_hidden_ceiling(ceiling_usd)
    started_at = _utc_now()
    (
        prompt,
        pricing,
        prompt_path,
        pricing_path,
        public_manifest_path,
        _,
    ) = _components(project_root, sdk_version)
    preflight = hidden_preflight(
        project_root,
        output_directory,
        ceiling_usd=ceiling_usd,
        sdk_version=sdk_version,
    )
    if not preflight["fits_ceiling"]:
        raise BudgetExceeded("remaining hidden requests do not fit the cumulative local ceiling")

    spend = cast(dict[str, float], preflight["recorded_spend_by_stage_usd"])
    prior_hidden_spend = spend[HIDDEN_STAGE]
    external_prior_spend = sum(spend.values()) - prior_hidden_spend
    cumulative_prior_spend = sum(spend.values())
    budget = BudgetLedger(
        pricing=pricing,
        total_ceiling_usd=ceiling_usd,
        stage_ceilings_usd={HIDDEN_STAGE: ceiling_usd - external_prior_spend},
        total_spent_usd=cumulative_prior_spend,
        stage_spent_usd={HIDDEN_STAGE: prior_hidden_spend},
    )
    counted_transport = CountingTransport(transport)
    adapter = OpenAIResponsesAdapter(
        counted_transport,
        sdk_version=sdk_version,
        timestamp=_utc_now,
    )
    cache_directory = output_directory / "cache"
    runner = EvaluationRunner(
        adapter=adapter,
        settings=diagnostic_settings(sdk_version),
        prompt=prompt,
        cache=FileTrialCache(cache_directory),
        budget=budget,
        allow_repair=False,
    )
    existing = _cache_artifacts(cache_directory)
    prompt_hash = _sha256(prompt_path.read_bytes())
    schema_hash = _sha256(decision_schema_bytes())

    artifacts: list[dict[str, Any]] = []
    budget_stop: str | None = None
    fatal_error_type: str | None = None
    inputs = _hidden_inputs(public_manifest_path)
    for item in inputs:
        for replicate_index in range(HIDDEN_REPLICATES):
            replicate_id = f"hidden-replicate-{replicate_index:02d}"
            logical_id = _logical_trial_id(item.world_id, replicate_index)
            cached = existing.get(logical_id)
            if cached is not None:
                _validate_cached_artifact(
                    cached,
                    item,
                    replicate_id,
                    prompt_hash,
                    schema_hash,
                )
                artifacts.append(cached)
                continue
            try:
                artifact = runner.run(
                    item.observation,
                    TrialContext(
                        world_id=item.world_id,
                        replicate_id=replicate_id,
                        benchmark_version="janus-v1",
                        stage=HIDDEN_STAGE,
                    ),
                )
            except BudgetExceeded as error:
                budget_stop = str(error)
                break
            except Exception as error:  # provider errors stop the stage; no blind retries
                fatal_error_type = type(error).__name__
                break
            artifacts.append(dict(artifact))
        if budget_stop is not None or fatal_error_type is not None:
            break

    completed = len(artifacts) == HIDDEN_TRIAL_COUNT
    passed = (
        completed
        and budget_stop is None
        and fatal_error_type is None
        and budget.total_spent_usd <= ceiling_usd + 1e-12
        and counted_transport.call_count <= int(preflight["maximum_provider_calls_remaining"])
    )
    cache_hashes = {
        path.name: _sha256(path.read_bytes()) for path in sorted(cache_directory.glob("*.json"))
    }
    manifest = {
        "schema_version": "janus-openai-hidden-run-v1",
        "stage": HIDDEN_STAGE,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "completed": completed,
        "passed": passed,
        "public_hidden_observations_accessed": True,
        "private_hidden_artifacts_accessed": False,
        "hidden_labels_or_oracles_accessed": False,
        "unblinded": False,
        "response_manifest_must_be_committed_before_unblinding": True,
        "world_count_planned": HIDDEN_WORLD_COUNT,
        "replicates_per_world": HIDDEN_REPLICATES,
        "trial_count_planned": HIDDEN_TRIAL_COUNT,
        "trial_count_completed": len(artifacts),
        "trial_count_previously_collected": int(preflight["trial_count_already_collected"]),
        "provider_calls_this_invocation": counted_transport.call_count,
        "repairs_allowed": False,
        "prior_cumulative_cost_usd": cumulative_prior_spend,
        "invocation_cost_usd": budget.total_spent_usd - cumulative_prior_spend,
        "cumulative_cost_usd": budget.total_spent_usd,
        "local_hard_ceiling_usd": ceiling_usd,
        "budget_stop": budget_stop,
        "fatal_error_type": fatal_error_type,
        "settings": diagnostic_settings(sdk_version).model_dump(mode="json"),
        "prompt_sha256": prompt_hash,
        "schema_sha256": schema_hash,
        "pricing_sha256": _sha256(pricing_path.read_bytes()),
        "public_manifest_sha256": _sha256(public_manifest_path.read_bytes()),
        "preflight": preflight,
        "returned_model_ids": sorted({str(item["returned_model_id"]) for item in artifacts}),
        "status_counts": {
            status.value: sum(item["validation_status"] == status.value for item in artifacts)
            for status in ResponseStatus
        },
        "response_artifact_sha256": cache_hashes,
        "trials": [
            {
                "logical_trial_id": item["logical_trial_id"],
                "cache_key": item["cache_key"],
                "validation_status": item["validation_status"],
                "repair_attempted": item["repair_attempted"],
                "fallback_action": item["fallback_action"],
                "actual_cost_usd": item["actual_cost_usd"],
                "estimated_cost_usd": item["estimated_cost_usd"],
                "usage": item["usage"],
                "provider_attempts": item["provider_attempts"],
            }
            for item in artifacts
        ],
    }
    return passed, _write_manifest(output_directory, manifest)
