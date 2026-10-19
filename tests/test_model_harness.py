import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from janus.cli import main
from janus.evaluation.budget import BudgetExceeded, BudgetLedger, PricingConfig
from janus.evaluation.harness import (
    SCHEMA_VERSION,
    EvaluationRunner,
    PromptRenderer,
    decision_schema_bytes,
    parse_decision,
)
from janus.evaluation.hidden_release import prepare_hidden_seed
from janus.evaluation.openai_smoke import (
    DIAGNOSTIC_MAXIMUM_OUTPUT_TOKENS,
    SMOKE_MAXIMUM_CEILING_USD,
    SMOKE_STAGE,
    diagnostic_preflight,
    run_openai_diagnostic,
    run_openai_smoke,
    smoke_preflight,
    validate_smoke_ceiling,
)
from janus.evaluation.phase0e_runner import run_phase0e_dry_run
from janus.models.base import (
    ModelRequest,
    ModelSettings,
    ResponseStatus,
    TrialContext,
    UsageRecord,
)
from janus.models.mock import MockAdapter
from janus.models.providers.openai import OpenAIResponsesAdapter, make_openai_sdk_transport
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
    assert [attempt["kind"] for attempt in repaired["provider_attempts"]] == [
        "primary",
        "repair",
    ]
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


def test_incomplete_response_is_recorded_and_never_repaired(tmp_path) -> None:
    adapter = MockAdapter(
        (None, valid_decision()),
        incomplete_indices=frozenset({0}),
        usage=UsageRecord(input_tokens=10, output_tokens=100, reasoning_tokens=100),
    )
    artifact = runner(tmp_path, adapter).run(b"{}", context())
    assert artifact["validation_status"] == ResponseStatus.INCOMPLETE.value
    assert artifact["validation_errors"] == ["provider response incomplete: max_output_tokens"]
    assert artifact["repair_attempted"] is False
    assert artifact["fallback_action"] == "do_nothing"
    assert adapter.call_count == 1
    assert artifact["provider_attempts"] == [
        {
            "kind": "primary",
            "response_status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "cache_write_tokens": 0,
                "output_tokens": 100,
                "reasoning_tokens": 100,
            },
            "estimated_cost_usd": 0.0,
            "actual_cost_usd": 0.0,
            "provider_request_id": "mock-world-1:r0-0",
            "timestamp_utc": "2026-10-17T00:00:00Z",
        }
    ]


def test_budget_preflight_stops_before_adapter_call(tmp_path) -> None:
    adapter = MockAdapter((valid_decision(),))
    budget = BudgetLedger(pricing(nonzero=True), 0.000001, {"test": 0.000001})
    with pytest.raises(BudgetExceeded, match="would exceed"):
        runner(tmp_path, adapter, budget).run(b'{"rows":[]}', context())
    assert adapter.call_count == 0
    assert budget.total_spent_usd == 0


def test_syntax_repair_is_skipped_and_primary_artifact_saved_when_budget_is_low(
    tmp_path,
) -> None:
    adapter = MockAdapter(
        ("not json", valid_decision()),
        usage=UsageRecord(input_tokens=9_000),
    )
    budget = BudgetLedger(pricing(nonzero=True), 0.01, {"test": 0.01})
    artifact = runner(tmp_path, adapter, budget).run(b"{}", context())
    assert artifact["validation_status"] == ResponseStatus.SCHEMA_FAILURE.value
    assert artifact["repair_attempted"] is False
    assert artifact["repair_blocked_by_budget"] is True
    assert adapter.call_count == 1
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


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
            "status": "completed",
            "incomplete_details": None,
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 10},
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
    assert seen[0]["service_tier"] == "default"
    assert seen[0]["prompt_cache_options"] == {"mode": "explicit"}
    assert seen[0]["store"] is False
    assert reply.returned_model_id == "gpt-6-luna-2026-10-01"
    assert reply.response_status == "completed"
    assert reply.incomplete_details is None
    assert reply.usage.input_tokens == 70
    assert reply.usage.cached_input_tokens == 20
    assert reply.usage.cache_write_tokens == 10
    assert reply.usage.reasoning_tokens == 15


def test_official_sdk_transport_disables_retries_and_sets_idempotency_without_network(
    monkeypatch,
) -> None:
    import openai

    captured = {}

    class FakeResponse:
        output_text = None

        def model_dump(self, *, mode):
            assert mode == "json"
            return {
                "id": "response-fake",
                "model": "gpt-6-luna",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [{"content": [{"type": "refusal"}]}],
                "usage": {},
            }

    class FakeResponses:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.responses = FakeResponses()

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    transport, sdk_version = make_openai_sdk_transport(
        "test-only-secret",
        timeout_seconds=17,
    )
    response = transport(
        {
            "model": "gpt-6-luna",
            "metadata": {"logical_trial_id": "world-1:r0"},
            "input": [],
        }
    )
    assert sdk_version
    assert captured["client"] == {
        "api_key": "test-only-secret",
        "timeout": 17,
        "max_retries": 0,
    }
    assert captured["request"]["extra_headers"] == {"Idempotency-Key": "world-1:r0"}
    assert response["refusal"] is True
    assert response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}


def test_openai_smoke_preflight_and_fake_run_never_need_a_key(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    plan = smoke_preflight(project_root)
    assert plan["network_calls"] == 0
    assert plan["paid_api_calls"] == 0
    assert plan["development_only"] is True
    assert plan["trial_count"] == 5
    assert plan["fits_ceiling"] is True
    assert plan["planned_primary_request_upper_bound_usd"] <= 0.01

    calls = []

    def transport(payload):
        calls.append(payload)
        return {
            "id": f"response-{len(calls)}",
            "model": "gpt-6-luna",
            "output_text": valid_decision(),
            "status": "completed",
            "incomplete_details": None,
            "usage": {
                "input_tokens": 4_000,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 200,
                "output_tokens_details": {"reasoning_tokens": 50},
            },
        }

    passed, manifest_path = run_openai_smoke(
        project_root,
        tmp_path / "smoke",
        transport=transport,
        sdk_version="test-sdk",
    )
    assert passed
    assert len(calls) == 5
    manifest = json.loads(manifest_path.read_text())
    assert manifest["hidden_artifacts_accessed"] is False
    assert manifest["provider_calls_this_invocation"] == 5
    assert manifest["trial_count_completed"] == 5
    assert manifest["cumulative_cost_usd"] < SMOKE_MAXIMUM_CEILING_USD
    assert "OPENAI_API_KEY" not in manifest_path.read_text()

    replay_calls = []

    def replay_transport(payload):
        replay_calls.append(payload)
        raise AssertionError("cache replay must not call the provider")

    replay_passed, replay_manifest_path = run_openai_smoke(
        project_root,
        tmp_path / "smoke",
        transport=replay_transport,
        sdk_version="test-sdk",
    )
    assert replay_passed
    assert replay_calls == []
    replay_manifest = json.loads(replay_manifest_path.read_text())
    assert replay_manifest["provider_calls_this_invocation"] == 0
    assert replay_manifest["invocation_cost_usd"] == pytest.approx(0)


def test_one_world_diagnostic_accounts_for_prior_spend_and_disables_repair(
    tmp_path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    prior_output = tmp_path / "prior-smoke"
    prior_cache = prior_output / "cache"
    prior_cache.mkdir(parents=True)
    prior_spend = 0.0042324
    (prior_cache / "prior.json").write_text(
        json.dumps({"stage": SMOKE_STAGE, "actual_cost_usd": prior_spend})
    )
    output = tmp_path / "diagnostic"
    plan = diagnostic_preflight(project_root, prior_output, output)
    assert plan["network_calls"] == 0
    assert plan["paid_api_calls"] == 0
    assert plan["trial_count"] == 1
    assert plan["repairs_allowed"] is False
    assert plan["prior_smoke_spend_usd"] == pytest.approx(prior_spend)
    assert plan["settings"]["maximum_output_tokens"] == DIAGNOSTIC_MAXIMUM_OUTPUT_TOKENS
    assert plan["fits_ceiling"] is True

    calls = []

    def transport(payload):
        calls.append(payload)
        return {
            "id": "diagnostic-response-1",
            "model": "gpt-6-luna",
            "output_text": valid_decision(),
            "status": "completed",
            "incomplete_details": None,
            "usage": {
                "input_tokens": 4_000,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 800,
                "output_tokens_details": {"reasoning_tokens": 650},
            },
        }

    passed, manifest_path = run_openai_diagnostic(
        project_root,
        prior_output,
        output,
        transport=transport,
        sdk_version="test-sdk",
    )
    assert passed
    assert len(calls) == 1
    assert calls[0]["max_output_tokens"] == DIAGNOSTIC_MAXIMUM_OUTPUT_TOKENS
    manifest = json.loads(manifest_path.read_text())
    assert manifest["repairs_allowed"] is False
    assert manifest["provider_calls_this_invocation"] == 1
    assert manifest["trial"]["validation_status"] == ResponseStatus.VALID.value
    assert manifest["trial"]["repair_attempted"] is False
    assert manifest["trial"]["provider_attempts"][0]["usage"]["reasoning_tokens"] == 650
    assert manifest["cumulative_cost_usd"] > prior_spend
    assert manifest["cumulative_cost_usd"] < SMOKE_MAXIMUM_CEILING_USD


def test_one_world_diagnostic_preserves_incomplete_without_repair(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    calls = []

    def transport(payload):
        calls.append(payload)
        return {
            "id": "diagnostic-incomplete-1",
            "model": "gpt-6-luna",
            "output_text": "",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {
                "input_tokens": 4_000,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": DIAGNOSTIC_MAXIMUM_OUTPUT_TOKENS,
                "output_tokens_details": {"reasoning_tokens": DIAGNOSTIC_MAXIMUM_OUTPUT_TOKENS},
            },
        }

    passed, manifest_path = run_openai_diagnostic(
        project_root,
        tmp_path / "no-prior",
        tmp_path / "incomplete",
        transport=transport,
        sdk_version="test-sdk",
    )
    assert passed is False
    assert len(calls) == 1
    manifest = json.loads(manifest_path.read_text())
    assert manifest["provider_calls_this_invocation"] == 1
    assert manifest["status_counts"][ResponseStatus.INCOMPLETE.value] == 1
    assert manifest["trial"]["repair_attempted"] is False
    assert manifest["trial"]["provider_attempts"][0]["incomplete_details"] == {
        "reason": "max_output_tokens"
    }


def test_openai_smoke_rejects_any_ceiling_above_one_cent() -> None:
    with pytest.raises(ValueError, match="at most"):
        validate_smoke_ceiling(0.010001)
    with pytest.raises(ValueError, match="greater than zero"):
        validate_smoke_ceiling(0)


def test_openai_smoke_cli_refuses_wrong_confirmation_or_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="confirm-spend"):
        main(["openai-smoke", "--confirm-spend", "WRONG"])
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set"):
        main(["openai-smoke", "--confirm-spend", "ONE_CENT_SMOKE"])
    with pytest.raises(SystemExit, match="confirm-spend"):
        main(["openai-diagnostic", "--confirm-spend", "WRONG"])
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set"):
        main(["openai-diagnostic", "--confirm-spend", "ONE_CENT_DIAGNOSTIC"])


def test_mock_phase0e_run_is_zero_cost_and_replayable(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "phase0e"
    assert run_phase0e_dry_run(output, project_root)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["trial_count"] == 5
    assert manifest["paid_api_calls"] == 0
    assert manifest["total_cost_usd"] == 0
    assert manifest["adapter_calls"] == 5
    assert manifest["cache_replay_adapter_calls"] == 0
    assert manifest["cache_replay_passed"] is True
    assert run_phase0e_dry_run(output, project_root)
    replay = json.loads((output / "run_manifest.json").read_text())
    assert replay["adapter_calls"] == 0
    assert replay["cache_replay_adapter_calls"] == 0


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
