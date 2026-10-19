"""Frozen scored-response and trial contracts for Janus v2."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from janus.core.config import StrictFrozenModel
from janus.interventions.actions import ActionID
from janus.models.base import DiagnosisProbabilities
from janus.scenarios.families import LatentFamily
from janus.v2 import SCHEMA_VERSION


class ActionRole(StrEnum):
    SUBSTANTIVE = "substantive"
    NEGATIVE_CONTROL = "negative_control"
    ABSTENTION_CONTROL = "abstention_control"


ACTION_ROLES: dict[ActionID, ActionRole] = {
    ActionID.DO_NOTHING: ActionRole.ABSTENTION_CONTROL,
    ActionID.PRICE_CEILING: ActionRole.NEGATIVE_CONTROL,
    ActionID.SUPPLY_SUBSIDY: ActionRole.NEGATIVE_CONTROL,
    ActionID.TRANSACTION_TAX: ActionRole.SUBSTANTIVE,
    ActionID.COLLUSION_INVESTIGATION: ActionRole.SUBSTANTIVE,
}


class DecisionV2(StrictFrozenModel):
    """Minimum scored response; rationale is deliberately not scored or required."""

    diagnosis: LatentFamily
    diagnosis_probabilities: DiagnosisProbabilities
    action: ActionID
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_probabilities(self) -> Self:
        probabilities = self.diagnosis_probabilities.as_dict()
        if abs(sum(probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("diagnosis probabilities must sum to one within 1e-6")
        maximum = max(probabilities.values())
        if maximum - probabilities[self.diagnosis] > 1e-12:
            raise ValueError("diagnosis must be an argmax diagnosis probability")
        return self


def decision_schema_bytes() -> bytes:
    schema = DecisionV2.model_json_schema()
    schema["title"] = SCHEMA_VERSION
    return json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()


class AdmissionThresholds(StrictFrozenModel):
    minimum_valid_rate: float = Field(default=0.98, ge=0, le=1)
    minimum_trials: int = Field(default=100, ge=50)
    maximum_budget_stop_rate: float = Field(default=0.0, ge=0, le=1)


class AdmissionResult(StrictFrozenModel):
    provider: str
    attempted: int
    valid: int
    valid_rate: float
    budget_stops: int
    admitted: bool
    reasons: tuple[str, ...]


def evaluate_admission(
    provider: str,
    statuses: tuple[str, ...],
    thresholds: AdmissionThresholds | None = None,
) -> AdmissionResult:
    thresholds = thresholds or AdmissionThresholds()
    attempted = len(statuses)
    valid = sum(status == "valid" for status in statuses)
    budget_stops = sum(status == "budget_stop" for status in statuses)
    valid_rate = 0.0 if attempted == 0 else valid / attempted
    reasons: list[str] = []
    if attempted < thresholds.minimum_trials:
        reasons.append(
            f"requires at least {thresholds.minimum_trials} representative development trials"
        )
    if valid_rate < thresholds.minimum_valid_rate:
        reasons.append(f"valid rate {valid_rate:.3f} is below {thresholds.minimum_valid_rate:.3f}")
    budget_stop_rate = 0.0 if attempted == 0 else budget_stops / attempted
    if budget_stop_rate > thresholds.maximum_budget_stop_rate:
        reasons.append("development run contains budget stops")
    return AdmissionResult(
        provider=provider,
        attempted=attempted,
        valid=valid,
        valid_rate=valid_rate,
        budget_stops=budget_stops,
        admitted=not reasons,
        reasons=tuple(reasons),
    )
