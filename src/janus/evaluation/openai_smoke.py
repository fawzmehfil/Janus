"""Five-world OpenAI development smoke test with a one-cent local ceiling."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from janus.evaluation.budget import (
    BudgetExceeded,
    BudgetLedger,
    PricingConfig,
    conservative_usage_estimate,
)
from janus.evaluation.harness import (
    SCHEMA_VERSION,
    EvaluationRunner,
    PromptRenderer,
    decision_schema_bytes,
)
from janus.models.base import ModelSettings, ResponseStatus, TrialContext
from janus.models.providers.openai import OpenAIResponsesAdapter, OpenAITransport
from janus.scenarios.generator import CandidateGeneratorConfig, generate_candidates
from janus.scenarios.matching import match_ambiguity_bundles
from janus.storage.cache import FileTrialCache

SMOKE_STAGE = "openai-development-smoke-v1"
SMOKE_MODEL = "gpt-6-luna"
SMOKE_TRIAL_COUNT = 5
SMOKE_MAXIMUM_CEILING_USD = 0.01
SMOKE_MAXIMUM_OUTPUT_TOKENS = 512
SMOKE_REASONING_EFFORT = "high"
SPEND_CONFIRMATION = "ONE_CENT_SMOKE"


@dataclass(frozen=True, slots=True)
class SmokeInput:
    world_id: str
    observation: bytes


class CountingTransport:
    def __init__(self, transport: OpenAITransport) -> None:
        self.transport = transport
        self.call_count = 0

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        return self.transport(payload)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


def validate_smoke_ceiling(ceiling_usd: float) -> None:
    if not 0 < ceiling_usd <= SMOKE_MAXIMUM_CEILING_USD:
        raise ValueError(
            f"smoke ceiling must be greater than zero and at most ${SMOKE_MAXIMUM_CEILING_USD:.2f}"
        )


@lru_cache(maxsize=1)
def development_smoke_inputs() -> tuple[SmokeInput, ...]:
    """Regenerate five frozen development observations without reading oracle files."""

    generator = CandidateGeneratorConfig()
    candidates = generate_candidates("development", generator)
    bundles = match_ambiguity_bundles(candidates, bundle_count=10, split="development")
    selected = (*bundles[0].worlds, bundles[1].worlds[0])
    return tuple(
        SmokeInput(
            world_id=f"janus-development-smoke-{index:03d}",
            observation=world.serialized_observation,
        )
        for index, world in enumerate(selected)
    )


def smoke_settings(sdk_version: str) -> ModelSettings:
    return ModelSettings(
        provider="openai",
        requested_model_id=SMOKE_MODEL,
        endpoint="v1/responses",
        temperature=None,
        top_p=None,
        reasoning_effort=SMOKE_REASONING_EFFORT,
        maximum_output_tokens=SMOKE_MAXIMUM_OUTPUT_TOKENS,
        sdk_version=sdk_version,
        prompt_version="primary-v1",
        schema_version=SCHEMA_VERSION,
    )


def _components(
    project_root: Path,
    sdk_version: str,
) -> tuple[PromptRenderer, PricingConfig, ModelSettings, Path, Path]:
    prompt_path = project_root / "configs/prompts/primary-v1.txt"
    pricing_path = project_root / "configs/pricing/openai-gpt-6-luna-2026-10-17.json"
    prompt = PromptRenderer(prompt_path.read_text(), version="primary-v1")
    pricing = PricingConfig.model_validate(_json_object(pricing_path))
    settings = smoke_settings(sdk_version)
    if pricing.model_id != settings.requested_model_id:
        raise ValueError("pricing model and requested smoke model differ")
    return prompt, pricing, settings, prompt_path, pricing_path


def _request_input_bytes(prompt: PromptRenderer, observation: bytes) -> int:
    return len(prompt.render(observation)) + len(decision_schema_bytes())


def smoke_preflight(
    project_root: Path,
    *,
    ceiling_usd: float = SMOKE_MAXIMUM_CEILING_USD,
    sdk_version: str = "preflight-only",
) -> dict[str, Any]:
    validate_smoke_ceiling(ceiling_usd)
    prompt, pricing, settings, prompt_path, pricing_path = _components(
        project_root,
        sdk_version,
    )
    trials = []
    total = 0.0
    for item in development_smoke_inputs():
        usage = conservative_usage_estimate(
            _request_input_bytes(prompt, item.observation),
            settings.maximum_output_tokens,
        )
        estimate = pricing.cost(usage)
        total += estimate
        trials.append(
            {
                "world_id": item.world_id,
                "observation_sha256": _sha256(item.observation),
                "input_token_upper_bound": usage.input_tokens,
                "output_token_ceiling": usage.output_tokens,
                "request_cost_upper_bound_usd": estimate,
            }
        )
    return {
        "schema_version": "janus-openai-smoke-preflight-v1",
        "network_calls": 0,
        "paid_api_calls": 0,
        "development_only": True,
        "trial_count": len(trials),
        "ceiling_usd": ceiling_usd,
        "planned_primary_request_upper_bound_usd": total,
        "fits_ceiling": total <= ceiling_usd,
        "repair_policy": (
            "at most one syntax-only repair when remaining local budget passes preflight; "
            "a repair may displace a later primary request"
        ),
        "settings": settings.model_dump(mode="json"),
        "prompt_sha256": _sha256(prompt_path.read_bytes()),
        "schema_sha256": _sha256(decision_schema_bytes()),
        "pricing_sha256": _sha256(pricing_path.read_bytes()),
        "trials": trials,
    }


def _existing_spend(cache_directory: Path) -> float:
    total = 0.0
    for path in sorted(cache_directory.glob("*.json")):
        payload = _json_object(path)
        if payload.get("stage") == SMOKE_STAGE:
            actual_cost = payload.get("actual_cost_usd", 0.0)
            if not isinstance(actual_cost, int | float):
                raise ValueError(f"invalid cached cost in {path}")
            total += float(actual_cost)
    return total


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


def run_openai_smoke(
    project_root: Path,
    output_directory: Path,
    *,
    transport: OpenAITransport,
    sdk_version: str,
    ceiling_usd: float = SMOKE_MAXIMUM_CEILING_USD,
) -> tuple[bool, Path]:
    """Run or safely resume the five development trials using an injected transport."""

    validate_smoke_ceiling(ceiling_usd)
    started_at = _utc_now()
    prompt, pricing, settings, prompt_path, pricing_path = _components(
        project_root,
        sdk_version,
    )
    preflight = smoke_preflight(
        project_root,
        ceiling_usd=ceiling_usd,
        sdk_version=sdk_version,
    )
    if not preflight["fits_ceiling"]:
        raise BudgetExceeded("five primary smoke requests do not fit the local ceiling")

    cache_directory = output_directory / "cache"
    prior_spend = _existing_spend(cache_directory)
    if prior_spend > ceiling_usd + 1e-12:
        raise BudgetExceeded("existing smoke artifacts already exceed the requested ceiling")
    budget = BudgetLedger(
        pricing=pricing,
        total_ceiling_usd=ceiling_usd,
        stage_ceilings_usd={SMOKE_STAGE: ceiling_usd},
        total_spent_usd=prior_spend,
        stage_spent_usd={SMOKE_STAGE: prior_spend},
    )
    counted_transport = CountingTransport(transport)
    adapter = OpenAIResponsesAdapter(
        counted_transport,
        sdk_version=sdk_version,
        timestamp=_utc_now,
    )
    runner = EvaluationRunner(
        adapter=adapter,
        settings=settings,
        prompt=prompt,
        cache=FileTrialCache(cache_directory),
        budget=budget,
    )

    artifacts: list[dict[str, Any]] = []
    budget_stop: str | None = None
    fatal_error_type: str | None = None
    for item in development_smoke_inputs():
        try:
            artifact = runner.run(
                item.observation,
                TrialContext(
                    world_id=item.world_id,
                    replicate_id="smoke-replicate-00",
                    benchmark_version="janus-v1",
                    stage=SMOKE_STAGE,
                ),
            )
        except BudgetExceeded as error:
            budget_stop = str(error)
            break
        except Exception as error:  # provider errors stop the stage; no blind retries
            fatal_error_type = type(error).__name__
            break
        artifacts.append(dict(artifact))

    completed = len(artifacts) == SMOKE_TRIAL_COUNT
    accepted_statuses = {ResponseStatus.VALID.value, ResponseStatus.REPAIRED.value}
    passed = (
        completed
        and all(item["validation_status"] in accepted_statuses for item in artifacts)
        and budget_stop is None
        and fatal_error_type is None
    )
    cache_hashes = {
        path.name: _sha256(path.read_bytes()) for path in sorted(cache_directory.glob("*.json"))
    }
    manifest = {
        "schema_version": "janus-openai-smoke-run-v1",
        "stage": SMOKE_STAGE,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "completed": completed,
        "passed": passed,
        "development_only": True,
        "hidden_artifacts_accessed": False,
        "trial_count_planned": SMOKE_TRIAL_COUNT,
        "trial_count_completed": len(artifacts),
        "provider_calls_this_invocation": counted_transport.call_count,
        "prior_cumulative_cost_usd": prior_spend,
        "invocation_cost_usd": budget.total_spent_usd - prior_spend,
        "cumulative_cost_usd": budget.total_spent_usd,
        "local_hard_ceiling_usd": ceiling_usd,
        "budget_stop": budget_stop,
        "fatal_error_type": fatal_error_type,
        "settings": settings.model_dump(mode="json"),
        "prompt_sha256": _sha256(prompt_path.read_bytes()),
        "schema_sha256": _sha256(decision_schema_bytes()),
        "pricing_sha256": _sha256(pricing_path.read_bytes()),
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
                "repair_blocked_by_budget": item["repair_blocked_by_budget"],
                "actual_cost_usd": item["actual_cost_usd"],
                "estimated_cost_usd": item["estimated_cost_usd"],
                "usage": item["usage"],
                "provider_request_ids": item["provider_request_ids"],
            }
            for item in artifacts
        ],
    }
    return passed, _write_manifest(output_directory, manifest)
