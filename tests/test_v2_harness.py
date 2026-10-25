from __future__ import annotations

import json
from pathlib import Path

from janus.evaluation.budget import BudgetLedger, PricingConfig
from janus.evaluation.harness import PromptRenderer
from janus.models.base import ModelSettings, TrialContext
from janus.storage.cache import FileTrialCache
from janus.v2 import SCHEMA_VERSION
from janus.v2.harness import V2EvaluationRunner
from janus.v2.providers import OpenAIV2Adapter


def pricing(rate: float = 0) -> PricingConfig:
    return PricingConfig(
        provider="mock",
        model_id="mock",
        effective_date="2026-10-19",
        source_url="local://mock",
        service_tier="local",
        input_per_million_usd=rate,
        cached_input_per_million_usd=rate,
        cache_write_per_million_usd=rate,
        output_per_million_usd=rate,
        reasoning_per_million_usd=rate,
    )


def settings() -> ModelSettings:
    return ModelSettings(
        provider="openai",
        requested_model_id="mock",
        endpoint="local://mock",
        temperature=0,
        top_p=1,
        reasoning_effort=None,
        maximum_output_tokens=128,
        sdk_version="mock",
        prompt_version="v2",
        schema_version=SCHEMA_VERSION,
    )


def context() -> TrialContext:
    return TrialContext(
        world_id="world",
        replicate_id="replicate",
        benchmark_version="janus-v2",
        stage="development",
    )


def test_incomplete_is_recorded_without_retry(tmp_path: Path) -> None:
    calls = 0

    def transport(payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "model": "mock",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output_text": "{",
            "usage": {"input_tokens": 4, "output_tokens": 128},
        }

    runner = V2EvaluationRunner(
        adapter=OpenAIV2Adapter(transport, sdk_version="x", timestamp=lambda: "now"),
        settings=settings(),
        prompt=PromptRenderer("{{OBSERVATION_JSON}}", "v2"),
        cache=FileTrialCache(tmp_path / "cache"),
        budget=BudgetLedger(pricing(), 0, {"development": 0}),
    )
    artifact = runner.run(b"{}", context())
    assert artifact["validation_status"] == "incomplete"
    assert artifact["incomplete_details"] == {"reason": "max_output_tokens"}
    assert artifact["retry_count"] == 0
    assert calls == 1


def test_budget_stop_does_not_call_provider(tmp_path: Path) -> None:
    calls = 0

    def transport(payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {}

    runner = V2EvaluationRunner(
        adapter=OpenAIV2Adapter(transport, sdk_version="x", timestamp=lambda: "now"),
        settings=settings(),
        prompt=PromptRenderer("{{OBSERVATION_JSON}}", "v2"),
        cache=FileTrialCache(tmp_path / "cache"),
        budget=BudgetLedger(pricing(100), 0, {"development": 0}),
    )
    artifact = runner.run(b"{}", context())
    assert artifact["validation_status"] == "budget_stop"
    assert artifact["budget_stopped"]
    assert calls == 0


def test_valid_artifact_is_cache_replayed(tmp_path: Path) -> None:
    calls = 0
    raw = json.dumps(
        {
            "diagnosis": "supply_shortage",
            "diagnosis_probabilities": {
                "supply_shortage": 0.7,
                "demand_increase": 0.1,
                "seller_collusion": 0.1,
                "momentum_demand": 0.1,
            },
            "action": "do_nothing",
            "confidence": 0.5,
        }
    )

    def transport(payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"model": "mock", "status": "completed", "output_text": raw, "usage": {}}

    runner = V2EvaluationRunner(
        adapter=OpenAIV2Adapter(transport, sdk_version="x", timestamp=lambda: "now"),
        settings=settings(),
        prompt=PromptRenderer("{{OBSERVATION_JSON}}", "v2"),
        cache=FileTrialCache(tmp_path / "cache"),
        budget=BudgetLedger(pricing(), 0, {"development": 0}),
    )
    first = runner.run(b"{}", context())
    second = runner.run(b"{}", context())
    assert first == second
    assert first["validation_status"] == "valid"
    assert calls == 1


def test_malformed_structured_output_is_not_repaired_or_retried(tmp_path: Path) -> None:
    calls = 0

    def transport(payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "model": "mock",
            "status": "completed",
            "output_text": '{"action":"do_nothing"}',
            "usage": {},
        }

    runner = V2EvaluationRunner(
        adapter=OpenAIV2Adapter(transport, sdk_version="x", timestamp=lambda: "now"),
        settings=settings(),
        prompt=PromptRenderer("{{OBSERVATION_JSON}}", "v2"),
        cache=FileTrialCache(tmp_path / "cache"),
        budget=BudgetLedger(pricing(), 0, {"development": 0}),
    )
    artifact = runner.run(b"{}", context())
    assert artifact["validation_status"] == "schema_failure"
    assert artifact["parsed_decision"] is None
    assert artifact["retry_count"] == 0
    assert calls == 1
