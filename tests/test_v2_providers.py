from __future__ import annotations

import json

import pytest

from janus.models.base import ModelRequest, ModelSettings, TrialContext
from janus.v2 import SCHEMA_VERSION
from janus.v2.contracts import decision_schema_bytes
from janus.v2.providers import (
    AnthropicV2Adapter,
    FireworksV2Adapter,
    GoogleV2Adapter,
    OpenAIV2Adapter,
)


def request(provider: str) -> ModelRequest:
    return ModelRequest(
        settings=ModelSettings(
            provider=provider,
            requested_model_id=f"{provider}-test",
            endpoint="local://test",
            temperature=0,
            top_p=1,
            reasoning_effort=None,
            maximum_output_tokens=512,
            sdk_version="test",
            prompt_version="primary-v2",
            schema_version=SCHEMA_VERSION,
        ),
        context=TrialContext(
            world_id="world-1",
            replicate_id="replicate-1",
            benchmark_version="janus-v2",
            stage="test",
        ),
        system_input=b"same system",
        developer_input=b"",
        user_input=b"same user",
        output_schema=decision_schema_bytes(),
        logical_trial_id="trial-1",
    )


def valid_text() -> str:
    return json.dumps(
        {
            "diagnosis": "demand_increase",
            "diagnosis_probabilities": {
                "supply_shortage": 0.1,
                "demand_increase": 0.7,
                "seller_collusion": 0.1,
                "momentum_demand": 0.1,
            },
            "action": "do_nothing",
            "confidence": 0.6,
        }
    )


def test_native_payloads_preserve_structured_schema() -> None:
    def noop(payload: dict[str, object]) -> dict[str, object]:
        return payload

    def timestamp() -> str:
        return "now"

    openai = OpenAIV2Adapter(noop, sdk_version="x", timestamp=timestamp)
    anthropic = AnthropicV2Adapter(noop, sdk_version="x", timestamp=timestamp)
    google = GoogleV2Adapter(noop, sdk_version="x", timestamp=timestamp)
    fireworks = FireworksV2Adapter(noop, sdk_version="x", timestamp=timestamp)
    assert openai.payload(request("openai"))["text"]["format"]["type"] == "json_schema"
    assert anthropic.payload(request("anthropic"))["output_config"]["format"]["type"] == (
        "json_schema"
    )
    assert google.payload(request("google"))["response_format"]["mime_type"] == ("application/json")
    assert fireworks.payload(request("fireworks"))["response_format"]["type"] == "json_schema"


def test_fireworks_reply_normalizes_usage_and_request_id() -> None:
    adapter = FireworksV2Adapter(
        lambda payload: {
            "id": "fw-request-1",
            "model": payload["model"],
            "choices": [{"finish_reason": "stop", "message": {"content": valid_text()}}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 5},
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        },
        sdk_version="x",
        timestamp=lambda: "now",
    )
    reply = adapter.complete(request("fireworks"))
    assert reply.provider_request_id == "fw-request-1"
    assert reply.usage.input_tokens == 15
    assert reply.usage.cached_input_tokens == 5
    assert reply.usage.output_tokens == 7
    assert reply.usage.reasoning_tokens == 3


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_native_replies_normalize_usage(provider: str) -> None:
    raw = valid_text()
    responses = {
        "openai": {
            "id": "o",
            "model": "openai-test",
            "status": "completed",
            "output_text": raw,
            "usage": {"input_tokens": 10, "output_tokens": 4},
        },
        "anthropic": {
            "id": "a",
            "model": "anthropic-test",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": raw}],
            "usage": {"input_tokens": 10, "output_tokens": 4},
        },
        "google": {
            "id": "g",
            "model": "google-test",
            "status": "completed",
            "output_text": raw,
            "usage_metadata": {"prompt_token_count": 10, "candidates_token_count": 4},
        },
    }
    classes = {
        "openai": OpenAIV2Adapter,
        "anthropic": AnthropicV2Adapter,
        "google": GoogleV2Adapter,
    }
    adapter = classes[provider](
        lambda payload: responses[provider], sdk_version="x", timestamp=lambda: "now"
    )
    reply = adapter.complete(request(provider))
    assert reply.raw_text == raw
    assert reply.usage.input_tokens == 10
    assert reply.usage.output_tokens == 4
    assert reply.latency_ms is not None


def test_transport_failures_and_rate_limits_are_preserved() -> None:
    def fail(payload: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("offline")

    failure = OpenAIV2Adapter(fail, sdk_version="x", timestamp=lambda: "now").complete(
        request("openai")
    )
    assert failure.transport_error == "RuntimeError: offline"
    limited = GoogleV2Adapter(
        lambda payload: {"status_code": 429, "_headers": {"retry-after": "2"}},
        sdk_version="x",
        timestamp=lambda: "now",
    ).complete(request("google"))
    assert limited.rate_limited
    assert limited.rate_limit_metadata == {"retry-after": "2"}


def test_returned_model_mismatch_and_missing_usage_are_preserved() -> None:
    adapter = OpenAIV2Adapter(
        lambda payload: {
            "id": "response-1",
            "model": "unexpected-snapshot",
            "status": "completed",
            "output_text": valid_text(),
        },
        sdk_version="x",
        timestamp=lambda: "now",
    )
    reply = adapter.complete(request("openai"))
    assert reply.returned_model_id == "unexpected-snapshot"
    assert reply.provider_request_id == "response-1"
    assert reply.usage.input_tokens == 0
    assert reply.usage.output_tokens == 0


def test_logical_idempotency_identifier_is_stable() -> None:
    adapter = OpenAIV2Adapter(lambda payload: payload, sdk_version="x", timestamp=lambda: "now")
    first = adapter.payload(request("openai"))
    second = adapter.payload(request("openai"))
    assert first["metadata"]["logical_trial_id"] == "trial-1"
    assert first == second
