"""Strict provider-independent request, response, and decision contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, Self

from pydantic import Field, model_validator

from janus.core.config import StrictFrozenModel
from janus.interventions.actions import ActionID
from janus.scenarios.families import LatentFamily


class DiagnosisProbabilities(StrictFrozenModel):
    supply_shortage: float = Field(ge=0, le=1)
    demand_increase: float = Field(ge=0, le=1)
    seller_collusion: float = Field(ge=0, le=1)
    momentum_demand: float = Field(ge=0, le=1)

    def as_dict(self) -> dict[LatentFamily, float]:
        return {
            LatentFamily.SUPPLY_SHORTAGE: self.supply_shortage,
            LatentFamily.DEMAND_INCREASE: self.demand_increase,
            LatentFamily.SELLER_COLLUSION: self.seller_collusion,
            LatentFamily.MOMENTUM_DEMAND: self.momentum_demand,
        }


class Decision(StrictFrozenModel):
    diagnosis: LatentFamily
    diagnosis_probabilities: DiagnosisProbabilities
    action: ActionID
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_probabilities_and_rationale(self) -> Self:
        probabilities = self.diagnosis_probabilities.as_dict()
        if abs(sum(probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("diagnosis probabilities must sum to one within 1e-6")
        maximum = max(probabilities.values())
        if maximum - probabilities[self.diagnosis] > 1e-12:
            raise ValueError("diagnosis must be an argmax diagnosis probability")
        if len(self.rationale.split()) > 100:
            raise ValueError("rationale must contain at most 100 words")
        return self


class ResponseStatus(StrEnum):
    VALID = "valid"
    REPAIRED = "repaired"
    INCOMPLETE = "incomplete"
    SCHEMA_FAILURE = "schema_failure"
    REFUSAL = "refusal"
    TIMEOUT = "timeout"
    EMPTY = "empty"
    RATE_LIMITED = "rate_limited"
    TRANSPORT_ERROR = "transport_error"
    BUDGET_STOP = "budget_stop"


class UsageRecord(StrictFrozenModel):
    input_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)

    def __add__(self, other: UsageRecord) -> UsageRecord:
        return UsageRecord(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


class ModelSettings(StrictFrozenModel):
    provider: str
    requested_model_id: str
    endpoint: str
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    reasoning_effort: str | None = None
    maximum_output_tokens: int = Field(gt=0)
    sdk_version: str
    prompt_version: str
    schema_version: str


class TrialContext(StrictFrozenModel):
    world_id: str
    replicate_id: str
    benchmark_version: str
    stage: str


class ModelRequest(StrictFrozenModel):
    settings: ModelSettings
    context: TrialContext
    system_input: bytes
    developer_input: bytes
    user_input: bytes
    output_schema: bytes
    logical_trial_id: str
    repair_attempt: bool = False


class ProviderReply(StrictFrozenModel):
    raw_text: str | None = None
    returned_model_id: str
    provider_request_id: str | None = None
    response_status: str | None = None
    incomplete_details: dict[str, object] | None = None
    usage: UsageRecord = Field(default_factory=UsageRecord)
    refusal: bool = False
    timed_out: bool = False
    rate_limited: bool = False
    transport_error: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    rate_limit_metadata: dict[str, str] | None = None
    timestamp_utc: str


class ModelAdapter(Protocol):
    def complete(self, request: ModelRequest) -> ProviderReply: ...
