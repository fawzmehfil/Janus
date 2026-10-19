"""Provider-neutral v2 adapters with injected, zero-network transports.

The payload builders mirror each provider's native structured-output surface,
while the scored prompt bytes and JSON schema remain identical. Live SDK
construction is intentionally deferred until exact model versions are frozen.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from janus.models.base import ModelRequest, ProviderReply, UsageRecord

Transport = Callable[[dict[str, Any]], dict[str, Any]]


def _http_json_transport(
    url: str,
    headers: dict[str, str],
    *,
    timeout_seconds: float,
) -> Transport:
    """Construct a no-retry JSON transport without reading ambient credentials."""

    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        import json

        request = Request(
            url,
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={"content-type": "application/json", **headers},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                result = json.loads(response.read())
                result["status_code"] = response.status
                result["_headers"] = dict(response.headers.items())
                return cast(dict[str, Any], result)
        except HTTPError as error:
            try:
                result = json.loads(error.read())
            except (json.JSONDecodeError, UnicodeDecodeError):
                result = {"error": {"type": "http_error", "message": str(error)}}
            result["status_code"] = error.code
            result["_headers"] = dict(error.headers.items())
            return cast(dict[str, Any], result)

    return transport


def make_anthropic_http_transport(api_key: str, *, timeout_seconds: float = 120.0) -> Transport:
    if not api_key.strip():
        raise ValueError("ANTHROPIC_API_KEY is empty")
    return _http_json_transport(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        timeout_seconds=timeout_seconds,
    )


def make_google_http_transport(api_key: str, *, timeout_seconds: float = 120.0) -> Transport:
    if not api_key.strip():
        raise ValueError("GEMINI_API_KEY is empty")
    return _http_json_transport(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        {"x-goog-api-key": api_key},
        timeout_seconds=timeout_seconds,
    )


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _rate_limits(response: dict[str, Any]) -> dict[str, str] | None:
    source = response.get("rate_limit_metadata", response.get("_headers"))
    if not isinstance(source, dict):
        return None
    result = {
        str(key): str(value)
        for key, value in source.items()
        if "rate" in str(key).lower() or "retry" in str(key).lower()
    }
    return result or None


def _is_rate_limited(response: dict[str, Any]) -> bool:
    error = response.get("error")
    error_type = error.get("type", "") if isinstance(error, dict) else ""
    return response.get("status_code") == 429 or "rate" in str(error_type).lower()


def _provider_error(response: dict[str, Any]) -> str | None:
    status_code = response.get("status_code")
    error = response.get("error")
    failed = isinstance(status_code, int) and status_code >= 400
    failed = failed or response.get("status") in {"failed", "cancelled"}
    if not failed and not isinstance(error, dict):
        return None
    if isinstance(error, dict):
        error_type = str(error.get("type", "provider_error"))
        message = str(error.get("message", "provider returned an error"))
        return f"{error_type}: {message}"
    return f"provider HTTP status {status_code}"


class NativeStructuredAdapter:
    """Base class that normalizes native provider replies without retrying."""

    provider = "unknown"

    def __init__(
        self,
        transport: Transport,
        *,
        sdk_version: str,
        timestamp: Callable[[], str],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.transport = transport
        self.sdk_version = sdk_version
        self.timestamp = timestamp
        self.monotonic = monotonic

    def payload(self, request: ModelRequest) -> dict[str, Any]:
        raise NotImplementedError

    def parse(
        self, response: dict[str, Any], request: ModelRequest, latency_ms: float
    ) -> ProviderReply:
        raise NotImplementedError

    def complete(self, request: ModelRequest) -> ProviderReply:
        started = self.monotonic()
        try:
            response = self.transport(self.payload(request))
        except TimeoutError:
            return self._failure(request, started, timed_out=True, error="timeout")
        except Exception as error:  # provider SDK exception type is deliberately transport-owned
            return self._failure(
                request,
                started,
                error=f"{type(error).__name__}: {error}",
            )
        latency_ms = max(0.0, (self.monotonic() - started) * 1_000.0)
        if _is_rate_limited(response):
            return ProviderReply(
                returned_model_id=_text(response.get("model"))
                or request.settings.requested_model_id,
                provider_request_id=_text(response.get("id")),
                response_status="rate_limited",
                rate_limited=True,
                rate_limit_metadata=_rate_limits(response),
                latency_ms=latency_ms,
                timestamp_utc=self.timestamp(),
            )
        provider_error = _provider_error(response)
        if provider_error is not None:
            return ProviderReply(
                returned_model_id=_text(response.get("model"))
                or request.settings.requested_model_id,
                provider_request_id=_text(response.get("id")),
                response_status="provider_error",
                transport_error=provider_error,
                rate_limit_metadata=_rate_limits(response),
                latency_ms=latency_ms,
                timestamp_utc=self.timestamp(),
            )
        return self.parse(response, request, latency_ms)

    def _failure(
        self,
        request: ModelRequest,
        started: float,
        *,
        timed_out: bool = False,
        error: str,
    ) -> ProviderReply:
        return ProviderReply(
            returned_model_id=request.settings.requested_model_id,
            response_status="timeout" if timed_out else "transport_error",
            timed_out=timed_out,
            transport_error=error,
            latency_ms=max(0.0, (self.monotonic() - started) * 1_000.0),
            timestamp_utc=self.timestamp(),
        )


class OpenAIV2Adapter(NativeStructuredAdapter):
    provider = "openai"

    def payload(self, request: ModelRequest) -> dict[str, Any]:
        import json

        inputs = [
            {"role": role, "content": content.decode()}
            for role, content in (
                ("system", request.system_input),
                ("developer", request.developer_input),
                ("user", request.user_input),
            )
            if content
        ]
        payload: dict[str, Any] = {
            "model": request.settings.requested_model_id,
            "input": inputs,
            "max_output_tokens": request.settings.maximum_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": request.settings.schema_version,
                    "schema": json.loads(request.output_schema),
                    "strict": True,
                }
            },
            "metadata": {"logical_trial_id": request.logical_trial_id},
            "store": False,
        }
        if request.settings.reasoning_effort is not None:
            payload["reasoning"] = {"effort": request.settings.reasoning_effort}
        if request.settings.temperature is not None:
            payload["temperature"] = request.settings.temperature
        if request.settings.top_p is not None:
            payload["top_p"] = request.settings.top_p
        return payload

    def parse(
        self, response: dict[str, Any], request: ModelRequest, latency_ms: float
    ) -> ProviderReply:
        usage = response.get("usage", {})
        usage = usage if isinstance(usage, dict) else {}
        input_details = usage.get("input_tokens_details", {})
        output_details = usage.get("output_tokens_details", {})
        input_details = input_details if isinstance(input_details, dict) else {}
        output_details = output_details if isinstance(output_details, dict) else {}
        cached = int(input_details.get("cached_tokens", 0))
        total_input = int(usage.get("input_tokens", 0))
        return ProviderReply(
            raw_text=_text(response.get("output_text")),
            returned_model_id=_text(response.get("model")) or request.settings.requested_model_id,
            provider_request_id=_text(response.get("id")),
            response_status=_text(response.get("status")),
            incomplete_details=(
                response.get("incomplete_details")
                if isinstance(response.get("incomplete_details"), dict)
                else None
            ),
            usage=UsageRecord(
                input_tokens=max(0, total_input - cached),
                cached_input_tokens=cached,
                output_tokens=int(usage.get("output_tokens", 0)),
                reasoning_tokens=int(output_details.get("reasoning_tokens", 0)),
            ),
            refusal=bool(response.get("refusal", False)),
            latency_ms=latency_ms,
            rate_limit_metadata=_rate_limits(response),
            timestamp_utc=self.timestamp(),
        )


class AnthropicV2Adapter(NativeStructuredAdapter):
    provider = "anthropic"

    def payload(self, request: ModelRequest) -> dict[str, Any]:
        import json

        system_parts = [
            value.decode() for value in (request.system_input, request.developer_input) if value
        ]
        payload: dict[str, Any] = {
            "model": request.settings.requested_model_id,
            "max_tokens": request.settings.maximum_output_tokens,
            "messages": [{"role": "user", "content": request.user_input.decode()}],
            "output_config": {
                "format": {"type": "json_schema", "schema": json.loads(request.output_schema)}
            },
            "metadata": {"user_id": request.logical_trial_id},
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.settings.temperature is not None:
            payload["temperature"] = request.settings.temperature
        if request.settings.top_p is not None:
            payload["top_p"] = request.settings.top_p
        if request.settings.reasoning_effort is not None:
            payload["output_config"]["effort"] = request.settings.reasoning_effort
        return payload

    def parse(
        self, response: dict[str, Any], request: ModelRequest, latency_ms: float
    ) -> ProviderReply:
        content = response.get("content", [])
        text_blocks = (
            [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if isinstance(content, list)
            else []
        )
        usage = response.get("usage", {})
        usage = usage if isinstance(usage, dict) else {}
        stop_reason = _text(response.get("stop_reason"))
        incomplete: dict[str, object] | None = (
            {"reason": "max_output_tokens"} if stop_reason == "max_tokens" else None
        )
        return ProviderReply(
            raw_text="".join(text_blocks) or None,
            returned_model_id=_text(response.get("model")) or request.settings.requested_model_id,
            provider_request_id=_text(response.get("id")),
            response_status="incomplete" if incomplete else "completed",
            incomplete_details=incomplete,
            usage=UsageRecord(
                input_tokens=int(usage.get("input_tokens", 0)),
                cached_input_tokens=int(usage.get("cache_read_input_tokens", 0)),
                cache_write_tokens=int(usage.get("cache_creation_input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                reasoning_tokens=int(
                    (usage.get("output_tokens_details") or {}).get("thinking_tokens", 0)
                ),
            ),
            refusal=stop_reason == "refusal",
            latency_ms=latency_ms,
            rate_limit_metadata=_rate_limits(response),
            timestamp_utc=self.timestamp(),
        )


class GoogleV2Adapter(NativeStructuredAdapter):
    provider = "google"

    def payload(self, request: ModelRequest) -> dict[str, Any]:
        import json

        system_parts = [
            value.decode() for value in (request.system_input, request.developer_input) if value
        ]
        prompt = request.user_input.decode()
        if system_parts:
            prompt = "\n\n".join((*system_parts, prompt))
        payload: dict[str, Any] = {
            "model": request.settings.requested_model_id,
            "input": prompt,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": json.loads(request.output_schema),
            },
            "generation_config": {
                "max_output_tokens": request.settings.maximum_output_tokens,
            },
            "metadata": {"logical_trial_id": request.logical_trial_id},
        }
        config = payload["generation_config"]
        if request.settings.temperature is not None:
            config["temperature"] = request.settings.temperature
        if request.settings.top_p is not None:
            config["top_p"] = request.settings.top_p
        return payload

    def parse(
        self, response: dict[str, Any], request: ModelRequest, latency_ms: float
    ) -> ProviderReply:
        raw_text = _text(response.get("output_text"))
        if raw_text is None:
            steps = response.get("steps", [])
            blocks = (
                [
                    block.get("text", "")
                    for step in steps
                    if isinstance(step, dict)
                    for block in step.get("content", [])
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if isinstance(steps, list)
                else []
            )
            raw_text = "".join(str(block) for block in blocks) or None
        usage = response.get("usage_metadata", response.get("usage", {}))
        usage = usage if isinstance(usage, dict) else {}
        status = _text(response.get("status")) or "completed"
        finish_reason = _text(response.get("finish_reason"))
        incomplete: dict[str, object] | None = None
        if status == "incomplete" or finish_reason in {"MAX_TOKENS", "MAX_OUTPUT_TOKENS"}:
            incomplete = {"reason": finish_reason or "unknown"}
            status = "incomplete"
        return ProviderReply(
            raw_text=raw_text,
            returned_model_id=_text(response.get("model")) or request.settings.requested_model_id,
            provider_request_id=_text(response.get("id")),
            response_status=status,
            incomplete_details=incomplete,
            usage=UsageRecord(
                input_tokens=int(
                    usage.get("prompt_token_count", usage.get("input_tokens", 0)) or 0
                ),
                cached_input_tokens=int(usage.get("cached_content_token_count", 0)),
                output_tokens=int(
                    usage.get("candidates_token_count", usage.get("output_tokens", 0)) or 0
                ),
                reasoning_tokens=int(usage.get("thoughts_token_count", 0)),
            ),
            refusal=finish_reason in {"SAFETY", "PROHIBITED_CONTENT"},
            latency_ms=latency_ms,
            rate_limit_metadata=_rate_limits(response),
            timestamp_utc=self.timestamp(),
        )
