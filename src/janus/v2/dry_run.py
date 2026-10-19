"""Four-provider, zero-network Janus v2 harness dry run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from janus.evaluation.budget import BudgetLedger, PricingConfig
from janus.evaluation.harness import PromptRenderer
from janus.models.base import ModelSettings, TrialContext
from janus.scenarios.generator import CandidateGeneratorConfig, generate_candidates
from janus.storage.cache import FileTrialCache
from janus.v2 import BENCHMARK_VERSION, SCHEMA_VERSION
from janus.v2.contracts import decision_schema_bytes
from janus.v2.harness import V2EvaluationRunner
from janus.v2.providers import AnthropicV2Adapter, GoogleV2Adapter, OpenAIV2Adapter


def _decision() -> str:
    return json.dumps(
        {
            "diagnosis": "seller_collusion",
            "diagnosis_probabilities": {
                "supply_shortage": 0.1,
                "demand_increase": 0.1,
                "seller_collusion": 0.7,
                "momentum_demand": 0.1,
            },
            "action": "collusion_investigation",
            "confidence": 0.72,
        },
        separators=(",", ":"),
    )


def _transports() -> dict[str, Any]:
    raw = _decision()
    return {
        "openai": lambda payload: {
            "id": "openai-mock-1",
            "model": payload["model"],
            "status": "completed",
            "output_text": raw,
            "usage": {"input_tokens": 800, "output_tokens": 90},
        },
        "anthropic": lambda payload: {
            "id": "anthropic-mock-1",
            "model": payload["model"],
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": raw}],
            "usage": {"input_tokens": 800, "output_tokens": 90},
        },
        "google": lambda payload: {
            "id": "google-mock-1",
            "model": payload["model"],
            "status": "completed",
            "output_text": raw,
            "usage_metadata": {"prompt_token_count": 800, "candidates_token_count": 90},
        },
    }


def run_v2_dry_run(output_directory: Path, project_root: Path) -> bool:
    prompt_path = project_root / "configs/prompts/primary-v2.txt"
    prompt = PromptRenderer(prompt_path.read_text(), version="primary-v2")
    world = generate_candidates(
        "v2-dry-run", CandidateGeneratorConfig(candidates_per_family=1, root_seed=20261020)
    )[0]
    pricing = PricingConfig(
        provider="mock",
        model_id="janus-v2-mock",
        effective_date="2026-10-19",
        source_url="local://janus/v2/mock",
        service_tier="local",
        input_per_million_usd=0,
        cached_input_per_million_usd=0,
        cache_write_per_million_usd=0,
        output_per_million_usd=0,
        reasoning_per_million_usd=0,
    )
    transports = _transports()
    timestamp = lambda: "2026-10-19T00:00:00+00:00"  # noqa: E731
    adapters = {
        "openai_luna": OpenAIV2Adapter(
            transports["openai"], sdk_version="mock", timestamp=timestamp
        ),
        "openai_strong": OpenAIV2Adapter(
            transports["openai"], sdk_version="mock", timestamp=timestamp
        ),
        "anthropic": AnthropicV2Adapter(
            transports["anthropic"], sdk_version="mock", timestamp=timestamp
        ),
        "google": GoogleV2Adapter(transports["google"], sdk_version="mock", timestamp=timestamp),
    }
    artifacts = []
    for role, adapter in adapters.items():
        provider = "openai" if role.startswith("openai_") else role
        settings = ModelSettings(
            provider=provider,
            requested_model_id=f"{role}-offline-mock",
            endpoint="local://mock",
            temperature=0,
            top_p=1,
            reasoning_effort=None,
            maximum_output_tokens=512,
            sdk_version="mock",
            prompt_version=prompt.version,
            schema_version=SCHEMA_VERSION,
        )
        runner = V2EvaluationRunner(
            adapter=adapter,
            settings=settings,
            prompt=prompt,
            cache=FileTrialCache(output_directory / "cache" / role),
            budget=BudgetLedger(pricing, total_ceiling_usd=0, stage_ceilings_usd={"v2-dry-run": 0}),
        )
        context = TrialContext(
            world_id=world.world_id,
            replicate_id="replicate-00",
            benchmark_version=BENCHMARK_VERSION,
            stage="v2-dry-run",
        )
        first = runner.run(world.serialized_observation, context)
        replay = runner.run(world.serialized_observation, context)
        artifacts.append(
            {
                "model_role": role,
                "provider": provider,
                "status": first["validation_status"],
                "semantic_input_sha256": first["semantic_input_sha256"],
                "cache_replay_identical": first == replay,
                "paid_api_calls": 0,
            }
        )
    semantic_hashes = {row["semantic_input_sha256"] for row in artifacts}
    passed = (
        all(row["status"] == "valid" for row in artifacts)
        and all(row["cache_replay_identical"] for row in artifacts)
        and len(semantic_hashes) == 1
    )
    manifest = {
        "schema_version": "janus-v2-dry-run-v1",
        "passed": passed,
        "network_calls": 0,
        "paid_api_calls": 0,
        "prompt_sha256": prompt.hash,
        "schema_sha256": hashlib.sha256(decision_schema_bytes()).hexdigest(),
        "provider_results": artifacts,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    return passed
