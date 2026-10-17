"""OpenAI Responses payload adapter with an injected transport.

This module never reads an API key and never opens a network connection. A
deployment must explicitly inject an authenticated transport after local gates
and budget authorization pass.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from janus.models.base import ModelRequest, ProviderReply, UsageRecord

OpenAITransport = Callable[[dict[str, Any]], dict[str, Any]]


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
        return ProviderReply(
            raw_text=None if raw_text is None else str(raw_text),
            returned_model_id=str(response.get("model", request.settings.requested_model_id)),
            provider_request_id=None if response.get("id") is None else str(response["id"]),
            usage=UsageRecord(
                input_tokens=int(usage.get("input_tokens", 0)),
                cached_input_tokens=int(input_details.get("cached_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                reasoning_tokens=int(output_details.get("reasoning_tokens", 0)),
            ),
            refusal=bool(response.get("refusal", False)),
            timed_out=False,
            timestamp_utc=self.timestamp(),
        )
