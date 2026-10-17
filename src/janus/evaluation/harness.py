"""Strict, cached, budgeted model-evaluation harness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from janus.evaluation.budget import BudgetExceeded, BudgetLedger, conservative_usage_estimate
from janus.interventions.actions import ActionID
from janus.models.base import (
    Decision,
    ModelAdapter,
    ModelRequest,
    ModelSettings,
    ProviderReply,
    ResponseStatus,
    TrialContext,
    UsageRecord,
)
from janus.storage.cache import FileTrialCache, request_cache_key

SCHEMA_VERSION = "janus-decision-v1"
FIXED_REPAIR_INSTRUCTION = (
    "Return only one JSON object that conforms exactly to the supplied schema. "
    "Do not add markdown or commentary. Preserve your intended diagnosis and action; "
    "this is a syntax-only repair and contains no outcome feedback."
)


def decision_schema_bytes() -> bytes:
    return json.dumps(
        Decision.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def parse_decision(raw_text: str) -> Decision:
    return Decision.model_validate_json(raw_text)


@dataclass(frozen=True, slots=True)
class PromptRenderer:
    template: str
    version: str
    system_input: bytes = b""
    developer_input: bytes = b""

    def render(self, observation: bytes) -> bytes:
        marker = "{{OBSERVATION_JSON}}"
        if self.template.count(marker) != 1:
            raise ValueError("prompt template must contain exactly one observation marker")
        return self.template.replace(marker, observation.decode("utf-8")).encode()

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()


class TrialArtifact(dict[str, Any]):
    """JSON-compatible artifact alias used by cache and manifest code."""


class EvaluationRunner:
    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        settings: ModelSettings,
        prompt: PromptRenderer,
        cache: FileTrialCache,
        budget: BudgetLedger,
        allow_repair: bool = True,
    ) -> None:
        self.adapter = adapter
        self.settings = settings
        self.prompt = prompt
        self.cache = cache
        self.budget = budget
        self.allow_repair = allow_repair

    def _request(
        self,
        observation: bytes,
        context: TrialContext,
        *,
        repair_attempt: bool = False,
        raw_to_repair: str | None = None,
    ) -> ModelRequest:
        user_input = self.prompt.render(observation)
        if repair_attempt:
            user_input = (
                FIXED_REPAIR_INSTRUCTION.encode()
                + b"\n\nINVALID RESPONSE:\n"
                + (raw_to_repair or "").encode()
            )
        logical_id = f"{context.world_id}:{context.replicate_id}"
        return ModelRequest(
            settings=self.settings,
            context=context,
            system_input=self.prompt.system_input,
            developer_input=self.prompt.developer_input,
            user_input=user_input,
            output_schema=decision_schema_bytes(),
            logical_trial_id=logical_id,
            repair_attempt=repair_attempt,
        )

    def _call(self, request: ModelRequest) -> tuple[ProviderReply, float, float]:
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
            input_bytes,
            request.settings.maximum_output_tokens,
        )
        estimate = self.budget.preflight(request.context.stage, estimated_usage)
        reply = self.adapter.complete(request)
        actual = self.budget.charge(request.context.stage, reply.usage)
        return reply, estimate, actual

    @staticmethod
    def _status(reply: ProviderReply) -> ResponseStatus:
        if reply.timed_out:
            return ResponseStatus.TIMEOUT
        if reply.refusal:
            return ResponseStatus.REFUSAL
        if reply.raw_text is None or not reply.raw_text.strip():
            return ResponseStatus.EMPTY
        return ResponseStatus.SCHEMA_FAILURE

    def run(self, observation: bytes, context: TrialContext) -> TrialArtifact:
        request = self._request(observation, context)
        key = request_cache_key(request)
        cached = self.cache.get(key)
        if cached is not None:
            return TrialArtifact(cached)

        reply, estimate, actual = self._call(request)
        total_usage = reply.usage
        total_estimate = estimate
        total_actual = actual
        errors: list[str] = []
        decision: Decision | None = None
        status = self._status(reply)
        if status is ResponseStatus.SCHEMA_FAILURE:
            try:
                decision = parse_decision(reply.raw_text or "")
                status = ResponseStatus.VALID
            except ValidationError as error:
                errors.append(str(error))

        repair_reply: ProviderReply | None = None
        repair_blocked_by_budget = False
        if (
            decision is None
            and self.allow_repair
            and status in {ResponseStatus.SCHEMA_FAILURE, ResponseStatus.EMPTY}
        ):
            repair_request = self._request(
                observation,
                context,
                repair_attempt=True,
                raw_to_repair=reply.raw_text,
            )
            try:
                repair_reply, repair_estimate, repair_actual = self._call(repair_request)
            except BudgetExceeded as error:
                repair_blocked_by_budget = True
                errors.append(f"syntax repair not sent: {error}")
            else:
                total_estimate += repair_estimate
                total_actual += repair_actual
                total_usage = total_usage + repair_reply.usage
                repair_status = self._status(repair_reply)
                if repair_status is ResponseStatus.SCHEMA_FAILURE:
                    try:
                        decision = parse_decision(repair_reply.raw_text or "")
                        status = ResponseStatus.REPAIRED
                    except ValidationError as error:
                        errors.append(str(error))
                        status = ResponseStatus.SCHEMA_FAILURE
                else:
                    status = repair_status

        artifact = TrialArtifact(
            cache_key=key,
            logical_trial_id=request.logical_trial_id,
            world_id=context.world_id,
            replicate_id=context.replicate_id,
            stage=context.stage,
            benchmark_version=context.benchmark_version,
            provider=self.settings.provider,
            requested_model_id=self.settings.requested_model_id,
            returned_model_id=(repair_reply or reply).returned_model_id,
            endpoint=self.settings.endpoint,
            model_settings=self.settings.model_dump(mode="json"),
            prompt_version=self.prompt.version,
            prompt_sha256=self.prompt.hash,
            schema_version=SCHEMA_VERSION,
            schema_sha256=hashlib.sha256(decision_schema_bytes()).hexdigest(),
            rendered_user_input=request.user_input.decode(),
            raw_response=reply.raw_text,
            repair_raw_response=None if repair_reply is None else repair_reply.raw_text,
            parsed_decision=None if decision is None else decision.model_dump(mode="json"),
            validation_status=status.value,
            validation_errors=errors,
            repair_attempted=repair_reply is not None,
            repair_blocked_by_budget=repair_blocked_by_budget,
            fallback_action=(
                decision.action if decision is not None else ActionID.DO_NOTHING
            ).value,
            usage=total_usage.model_dump(mode="json"),
            estimated_cost_usd=total_estimate,
            actual_cost_usd=total_actual,
            provider_request_ids=[
                value
                for value in (
                    reply.provider_request_id,
                    None if repair_reply is None else repair_reply.provider_request_id,
                )
                if value is not None
            ],
            timestamps_utc=[
                reply.timestamp_utc,
                *([] if repair_reply is None else [repair_reply.timestamp_utc]),
            ],
        )
        self.cache.put(key, artifact)
        return artifact


def usage_from_artifact(artifact: TrialArtifact) -> UsageRecord:
    return UsageRecord.model_validate(artifact["usage"])
