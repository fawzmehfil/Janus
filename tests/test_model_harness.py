import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from janus.evaluation.budget import BudgetExceeded, BudgetLedger, PricingConfig
from janus.evaluation.harness import (
    SCHEMA_VERSION,
    EvaluationRunner,
    PromptRenderer,
    decision_schema_bytes,
    parse_decision,
)
from janus.evaluation.hidden_release import prepare_hidden_seed
from janus.evaluation.phase0e_runner import run_phase0e_dry_run
from janus.models.base import (
    ModelRequest,
    ModelSettings,
    ResponseStatus,
    TrialContext,
    UsageRecord,
)
from janus.models.mock import MockAdapter
from janus.models.providers.openai import OpenAIResponsesAdapter
from janus.storage.cache import FileTrialCache, request_cache_key


def valid_decision(**updates) -> str:
    payload = {
        "diagnosis": "supply_shortage",
        "diagnosis_probabilities": {
            "supply_shortage": 0.7,
            "demand_increase": 0.1,
            "seller_collusion": 0.1,
            "momentum_demand": 0.1,
        },
        "action": "do_nothing",
        "confidence": 0.6,
        "rationale": "A concise rationale.",
    }
    payload.update(updates)
    return json.dumps(payload)


def pricing(nonzero: bool = False) -> PricingConfig:
    rate = 1.0 if nonzero else 0.0
    return PricingConfig(
        provider="test",
        model_id="test-model",
        effective_date="2026-10-17",
        source_url="local://test",
        service_tier="test",
        input_per_million_usd=rate,
        cached_input_per_million_usd=rate,
        cache_write_per_million_usd=rate,
        output_per_million_usd=rate,
        reasoning_per_million_usd=rate,
    )


def settings() -> ModelSettings:
    return ModelSettings(
        provider="mock",
        requested_model_id="mock",
        endpoint="local://mock",
        temperature=0,
        top_p=1,
        reasoning_effort="none",
        maximum_output_tokens=100,
        sdk_version="test",
        prompt_version="test-v1",
        schema_version=SCHEMA_VERSION,
    )


def context(replicate: str = "r0") -> TrialContext:
    return TrialContext(
        world_id="world-1",
        replicate_id=replicate,
        benchmark_version="v1",
        stage="test",
    )


def runner(tmp_path: Path, adapter: MockAdapter, budget: BudgetLedger | None = None):
    return EvaluationRunner(
        adapter=adapter,
        settings=settings(),
        prompt=PromptRenderer("Observation: {{OBSERVATION_JSON}}", "test-v1"),
        cache=FileTrialCache(tmp_path / "cache"),
        budget=budget or BudgetLedger(pricing(), 0.0, {"test": 0.0}),
    )


def test_strict_decision_schema_accepts_only_valid_argmax_probabilities() -> None:
    assert parse_decision(valid_decision()).action.value == "do_nothing"
    with pytest.raises(ValidationError, match="sum to one"):
        parse_decision(
            valid_decision(
                diagnosis_probabilities={
                    "supply_shortage": 0.8,
                    "demand_increase": 0.1,
                    "seller_collusion": 0.1,
                    "momentum_demand": 0.1,
                }
            )
        )
    with pytest.raises(ValidationError, match="argmax"):
        parse_decision(valid_decision(diagnosis="demand_increase"))
    with pytest.raises(ValidationError, match="at most 100"):
        parse_decision(valid_decision(rationale="word " * 101))
    with pytest.raises(ValidationError, match="Extra inputs"):
        parse_decision(valid_decision(extra="forbidden"))


def test_valid_trial_is_cached_and_planned_replicates_have_distinct_keys(tmp_path) -> None:
    adapter = MockAdapter((valid_decision(),))
    evaluation = runner(tmp_path, adapter)
    first = evaluation.run(b'{"rows":[]}', context("r0"))
    again = evaluation.run(b'{"rows":[]}', context("r0"))
    assert first == again
    assert adapter.call_count == 1
    request_base = ModelRequest(
        settings=settings(),
        context=context("r0"),
        system_input=b"",
        developer_input=b"",
        user_input=b"x",
        output_schema=decision_schema_bytes(),
        logical_trial_id="world-1:r0",
    )
    request_other = request_base.model_copy(
        update={"context": context("r1"), "logical_trial_id": "world-1:r1"}
    )
    assert request_cache_key(request_base) != request_cache_key(request_other)


def test_one_fixed_repair_and_invalid_fallback_are_explicit(tmp_path) -> None:
    repaired_adapter = MockAdapter(("not json", valid_decision()))
    repaired = runner(tmp_path / "repaired", repaired_adapter).run(b"{}", context())
    assert repaired["validation_status"] == ResponseStatus.REPAIRED.value
    assert repaired["repair_attempted"] is True
    assert repaired_adapter.call_count == 2

    invalid_adapter = MockAdapter(("not json", "still not json"))
    invalid = runner(tmp_path / "invalid", invalid_adapter).run(b"{}", context())
    assert invalid["validation_status"] == ResponseStatus.SCHEMA_FAILURE.value
    assert invalid["fallback_action"] == "do_nothing"
    assert invalid_adapter.call_count == 2

    refusal_adapter = MockAdapter((None,), refusal_indices=frozenset({0}))
    refusal = runner(tmp_path / "refusal", refusal_adapter).run(b"{}", context())
    assert refusal["validation_status"] == ResponseStatus.REFUSAL.value
    assert refusal_adapter.call_count == 1

    timeout_adapter = MockAdapter((None,), timeout_indices=frozenset({0}))
    timeout = runner(tmp_path / "timeout", timeout_adapter).run(b"{}", context())
    assert timeout["validation_status"] == ResponseStatus.TIMEOUT.value
    assert timeout_adapter.call_count == 1


def test_budget_preflight_stops_before_adapter_call(tmp_path) -> None:
    adapter = MockAdapter((valid_decision(),))
    budget = BudgetLedger(pricing(nonzero=True), 0.000001, {"test": 0.000001})
    with pytest.raises(BudgetExceeded, match="would exceed"):
        runner(tmp_path, adapter, budget).run(b'{"rows":[]}', context())
    assert adapter.call_count == 0
    assert budget.total_spent_usd == 0


def test_cost_reconciliation_tracks_all_exposed_usage_categories() -> None:
    budget = BudgetLedger(pricing(nonzero=True), 1.0, {"test": 1.0})
    usage = UsageRecord(
        input_tokens=10,
        cached_input_tokens=20,
        cache_write_tokens=30,
        output_tokens=40,
        reasoning_tokens=50,
    )
    assert budget.preflight("test", usage) == pytest.approx(0.00015)
    assert budget.charge("test", usage) == pytest.approx(0.00015)
    assert budget.total_spent_usd == pytest.approx(0.00015)


def test_openai_adapter_builds_strict_payload_and_maps_usage_without_network() -> None:
    seen = []

    def transport(payload):
        seen.append(payload)
        return {
            "id": "response-1",
            "model": "gpt-6-luna-2026-10-01",
            "output_text": valid_decision(),
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 20},
                "output_tokens": 50,
                "output_tokens_details": {"reasoning_tokens": 15},
            },
        }

    adapter = OpenAIResponsesAdapter(
        transport,
        sdk_version="test-sdk",
        timestamp=lambda: "2026-10-17T00:00:00Z",
    )
    request = ModelRequest(
        settings=settings().model_copy(update={"requested_model_id": "gpt-6-luna"}),
        context=context(),
        system_input=b"system",
        developer_input=b"developer",
        user_input=b"user",
        output_schema=decision_schema_bytes(),
        logical_trial_id="world-1:r0",
    )
    reply = adapter.complete(request)
    assert len(seen) == 1
    assert seen[0]["text"]["format"]["strict"] is True
    assert seen[0]["metadata"]["replicate_id"] == "r0"
    assert reply.returned_model_id == "gpt-6-luna-2026-10-01"
    assert reply.usage.cached_input_tokens == 20
    assert reply.usage.reasoning_tokens == 15


def test_mock_phase0e_run_is_zero_cost_and_replayable(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "phase0e"
    assert run_phase0e_dry_run(output, project_root)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["trial_count"] == 5
    assert manifest["paid_api_calls"] == 0
    assert manifest["total_cost_usd"] == 0
    assert run_phase0e_dry_run(output, project_root)
    replay = json.loads((output / "run_manifest.json").read_text())
    assert replay["adapter_calls"] == 0


def test_hidden_seed_commitment_is_idempotent_and_revision_bound(tmp_path) -> None:
    private = tmp_path / "private"
    commitment = tmp_path / "commitment.json"
    first = prepare_hidden_seed(private, commitment, code_commit="abc123")
    second = prepare_hidden_seed(private, commitment, code_commit="abc123")
    assert first == second
    assert len(first) == 64
    assert (private / "seed_manifest.json").stat().st_mode & 0o777 == 0o600
    public = json.loads(commitment.read_text())
    assert public["sha256"] == first
    assert "root_seed" not in public
    with pytest.raises(ValueError, match="different code revision"):
        prepare_hidden_seed(private, commitment, code_commit="different")
