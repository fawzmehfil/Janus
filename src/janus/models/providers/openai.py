"""OpenAI Responses adapter and explicitly constructed official-SDK transport.

The adapter remains transport-injected for no-network tests. The SDK transport
accepts a caller-supplied key only after CLI confirmation; this module never
reads environment variables or writes credentials to artifacts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from janus.models.base import ModelRequest, ProviderReply, UsageRecord

OpenAITransport = Callable[[dict[str, Any]], dict[str, Any]]


def _has_refusal(payload: dict[str, Any]) -> bool:
    output = payload.get("output", [])
    return any(
        isinstance(item, dict)
        and any(
            isinstance(content, dict) and content.get("type") == "refusal"
            for content in item.get("content", [])
        )
        for item in output
    )


def make_openai_sdk_transport(
    api_key: str,
    *,
    timeout_seconds: float = 120.0,
) -> tuple[OpenAITransport, str]:
    """Create a no-retry official-SDK transport for an explicitly authorized run."""

    if not api_key.strip():
        raise ValueError("OPENAI_API_KEY is empty")
    from importlib.metadata import version

    from openai import APITimeoutError, OpenAI

    client = OpenAI(
        api_key=api_key,
        timeout=timeout_seconds,
        max_retries=0,
    )

    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        logical_trial_id = str(payload.get("metadata", {}).get("logical_trial_id", ""))
        try:
            response = client.responses.create(
                **payload,
                extra_headers={"Idempotency-Key": logical_trial_id},
            )
        except APITimeoutError:
            return {
                "model": payload["model"],
                "output_text": None,
                "status": None,
                "incomplete_details": None,
                "timed_out": True,
                "usage": {},
            }
        data = cast(dict[str, Any], response.model_dump(mode="json"))
        data["output_text"] = response.output_text
        data["refusal"] = _has_refusal(data)
        return data

    return transport, version("openai")


class OpenAIResponsesAdapter:
    def __init__(
        self,
        transport: OpenAITransport,
        *,
        sdk_version: str,
        timestamp: Callable[[], str],
    ) -> None:
        self.transport = transport
        self.sdk_version = sdk_version
        self.timestamp = timestamp

    @staticmethod
    def payload(request: ModelRequest) -> dict[str, Any]:
        schema = __import__("json").loads(request.output_schema)
        messages = []
        for role, content in (
            ("system", request.system_input),
            ("developer", request.developer_input),
            ("user", request.user_input),
        ):
            if content:
                messages.append({"role": role, "content": content.decode()})
        payload: dict[str, Any] = {
            "model": request.settings.requested_model_id,
            "input": messages,
            "max_output_tokens": request.settings.maximum_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": request.settings.schema_version,
                    "schema": schema,
                    "strict": True,
                }
            },
            "metadata": {
                "logical_trial_id": request.logical_trial_id,
                "replicate_id": request.context.replicate_id,
            },
            "service_tier": "default",
            # Explicit mode without breakpoints disables the higher-priced cache
            # writes that implicit mode could create during this tiny smoke run.
            "prompt_cache_options": {"mode": "explicit"},
            "store": False,
        }
        if request.settings.reasoning_effort is not None:
            payload["reasoning"] = {"effort": request.settings.reasoning_effort}
        if request.settings.temperature is not None:
            payload["temperature"] = request.settings.temperature
        if request.settings.top_p is not None:
            payload["top_p"] = request.settings.top_p
        return payload

    def complete(self, request: ModelRequest) -> ProviderReply:
        response = self.transport(self.payload(request))
        usage = response.get("usage", {})
        input_details = usage.get("input_tokens_details", {})
        output_details = usage.get("output_tokens_details", {})
        raw_text = response.get("output_text")
        incomplete_details = response.get("incomplete_details")
        cached_tokens = int(input_details.get("cached_tokens", 0))
        cache_write_tokens = int(input_details.get("cache_write_tokens", 0))
        total_input_tokens = int(usage.get("input_tokens", 0))
        return ProviderReply(
            raw_text=None if raw_text is None else str(raw_text),
            returned_model_id=str(response.get("model", request.settings.requested_model_id)),
            provider_request_id=None if response.get("id") is None else str(response["id"]),
            response_status=(None if response.get("status") is None else str(response["status"])),
            incomplete_details=(
                incomplete_details if isinstance(incomplete_details, dict) else None
            ),
            usage=UsageRecord(
                input_tokens=max(0, total_input_tokens - cached_tokens - cache_write_tokens),
                cached_input_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
                output_tokens=int(usage.get("output_tokens", 0)),
                reasoning_tokens=int(output_details.get("reasoning_tokens", 0)),
            ),
            refusal=bool(response.get("refusal", False)),
            timed_out=bool(response.get("timed_out", False)),
            timestamp_utc=self.timestamp(),
        )
