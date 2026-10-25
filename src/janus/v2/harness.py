"""No-retry, provider-neutral Janus v2 evaluation harness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from janus.evaluation.budget import BudgetExceeded, BudgetLedger, conservative_usage_estimate
from janus.evaluation.harness import PromptRenderer
from janus.models.base import (
    ModelAdapter,
    ModelRequest,
    ModelSettings,
    ProviderReply,
    ResponseStatus,
    TrialContext,
)
from janus.storage.cache import FileTrialCache, request_cache_key
from janus.v2 import SCHEMA_VERSION
from janus.v2.contracts import DecisionV2, decision_schema_bytes


@dataclass(frozen=True, slots=True)
class V2EvaluationRunner:
    adapter: ModelAdapter
    settings: ModelSettings
    prompt: PromptRenderer
    cache: FileTrialCache
    budget: BudgetLedger

    def _request(self, observation: bytes, context: TrialContext) -> ModelRequest:
        return ModelRequest(
            settings=self.settings,
            context=context,
            system_input=self.prompt.system_input,
            developer_input=self.prompt.developer_input,
            user_input=self.prompt.render(observation),
            output_schema=decision_schema_bytes(),
            logical_trial_id=(
                f"{context.benchmark_version}:{self.settings.provider}:"
                f"{context.world_id}:{context.replicate_id}"
            ),
        )

    @staticmethod
    def _semantic_hash(request: ModelRequest) -> str:
        digest = hashlib.sha256()
        for role, content in (
            (b"system\0", request.system_input),
            (b"developer\0", request.developer_input),
            (b"user\0", request.user_input),
            (b"schema\0", request.output_schema),
        ):
            digest.update(role)
            digest.update(content)
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _status(reply: ProviderReply) -> ResponseStatus:
        if reply.timed_out:
            return ResponseStatus.TIMEOUT
        if reply.rate_limited:
            return ResponseStatus.RATE_LIMITED
        if reply.transport_error is not None:
            return ResponseStatus.TRANSPORT_ERROR
        if reply.refusal:
            return ResponseStatus.REFUSAL
        if reply.response_status == "incomplete":
            return ResponseStatus.INCOMPLETE
        if reply.raw_text is None or not reply.raw_text.strip():
            return ResponseStatus.EMPTY
        return ResponseStatus.SCHEMA_FAILURE

    def run(self, observation: bytes, context: TrialContext) -> dict[str, Any]:
        request = self._request(observation, context)
        key = request_cache_key(request)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        input_bytes = sum(
            len(value)
            for value in (
                request.system_input,
                request.developer_input,
                request.user_input,
                request.output_schema,
            )
        )
        estimated_usage = conservative_usage_estimate(
            input_bytes, self.settings.maximum_output_tokens
        )
        try:
            estimated_cost = self.budget.preflight(context.stage, estimated_usage)
        except BudgetExceeded as error:
            return self._artifact(
                request,
                key,
                status=ResponseStatus.BUDGET_STOP,
                reply=None,
                estimated_cost=0.0,
                actual_cost=0.0,
                decision=None,
                errors=(str(error),),
            )

        reply = self.adapter.complete(request)
        actual_cost = self.budget.charge(context.stage, reply.usage)
        status = self._status(reply)
        decision: DecisionV2 | None = None
        errors: list[str] = []
        if status is ResponseStatus.SCHEMA_FAILURE:
            try:
                decision = DecisionV2.model_validate_json(reply.raw_text or "")
                status = ResponseStatus.VALID
            except ValidationError as error:
                errors.append(str(error))
        elif status is ResponseStatus.INCOMPLETE:
            reason = (reply.incomplete_details or {}).get("reason", "unknown")
            errors.append(f"provider response incomplete: {reason}")
        elif reply.transport_error:
            errors.append(reply.transport_error)

        artifact = self._artifact(
            request,
            key,
            status=status,
            reply=reply,
            estimated_cost=estimated_cost,
            actual_cost=actual_cost,
            decision=decision,
            errors=tuple(errors),
        )
        self.cache.put(key, artifact)
        return artifact

    def _artifact(
        self,
        request: ModelRequest,
        key: str,
        *,
        status: ResponseStatus,
        reply: ProviderReply | None,
        estimated_cost: float,
        actual_cost: float,
        decision: DecisionV2 | None,
        errors: tuple[str, ...],
    ) -> dict[str, Any]:
        usage = {} if reply is None else reply.usage.model_dump(mode="json")
        return {
            "artifact_version": "janus-v2-trial-v1",
            "cache_key": key,
            "logical_trial_id": request.logical_trial_id,
            "world_id": request.context.world_id,
            "replicate_id": request.context.replicate_id,
            "stage": request.context.stage,
            "benchmark_version": request.context.benchmark_version,
            "provider": self.settings.provider,
            "requested_model_id": self.settings.requested_model_id,
            "returned_model_id": (
                self.settings.requested_model_id if reply is None else reply.returned_model_id
            ),
            "endpoint": self.settings.endpoint,
            "model_settings": self.settings.model_dump(mode="json"),
            "semantic_input_sha256": self._semantic_hash(request),
            "prompt_version": self.prompt.version,
            "prompt_sha256": self.prompt.hash,
            "schema_version": SCHEMA_VERSION,
            "schema_sha256": hashlib.sha256(decision_schema_bytes()).hexdigest(),
            "rendered_user_input": request.user_input.decode(),
            "raw_response": None if reply is None else reply.raw_text,
            "parsed_decision": None if decision is None else decision.model_dump(mode="json"),
            "validation_status": status.value,
            "validation_errors": list(errors),
            "response_status": None if reply is None else reply.response_status,
            "incomplete_details": None if reply is None else reply.incomplete_details,
            "refusal": False if reply is None else reply.refusal,
            "timed_out": False if reply is None else reply.timed_out,
            "rate_limited": False if reply is None else reply.rate_limited,
            "transport_error": None if reply is None else reply.transport_error,
            "budget_stopped": status is ResponseStatus.BUDGET_STOP,
            "usage": usage,
            "latency_ms": None if reply is None else reply.latency_ms,
            "estimated_cost_usd": estimated_cost,
            "actual_cost_usd": actual_cost,
            "provider_request_id": None if reply is None else reply.provider_request_id,
            "rate_limit_metadata": None if reply is None else reply.rate_limit_metadata,
            "timestamp_utc": None if reply is None else reply.timestamp_utc,
            "retry_count": 0,
        }


def artifact_sha256(artifact: dict[str, Any]) -> str:
    encoded = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
