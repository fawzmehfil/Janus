from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from janus.interventions.actions import ActionID
from janus.v2.contracts import (
    ACTION_ROLES,
    ActionRole,
    AdmissionThresholds,
    DecisionV2,
    decision_schema_bytes,
    evaluate_admission,
)


def valid_decision() -> dict[str, object]:
    return {
        "diagnosis": "seller_collusion",
        "diagnosis_probabilities": {
            "supply_shortage": 0.1,
            "demand_increase": 0.1,
            "seller_collusion": 0.7,
            "momentum_demand": 0.1,
        },
        "action": "collusion_investigation",
        "confidence": 0.8,
    }


def test_v2_contract_is_minimal_and_strict() -> None:
    parsed = DecisionV2.model_validate(valid_decision())
    assert parsed.action is ActionID.COLLUSION_INVESTIGATION
    with pytest.raises(ValidationError):
        DecisionV2.model_validate({**valid_decision(), "rationale": "not scored"})


def test_v2_probabilities_must_be_calibratable_distribution() -> None:
    payload = valid_decision()
    payload["diagnosis_probabilities"] = {
        "supply_shortage": 0.2,
        "demand_increase": 0.2,
        "seller_collusion": 0.2,
        "momentum_demand": 0.2,
    }
    with pytest.raises(ValidationError, match="sum to one"):
        DecisionV2.model_validate_json(json.dumps(payload))


def test_provider_admission_requires_rate_and_sample_size() -> None:
    thresholds = AdmissionThresholds(minimum_trials=100, minimum_valid_rate=0.98)
    admitted = evaluate_admission("x", ("valid",) * 98 + ("incomplete",) * 2, thresholds)
    assert admitted.admitted
    too_small = evaluate_admission("x", ("valid",) * 99, thresholds)
    assert not too_small.admitted
    low_rate = evaluate_admission("x", ("valid",) * 97 + ("refusal",) * 3, thresholds)
    assert not low_rate.admitted


def test_primary_action_roles_are_explicit() -> None:
    assert ACTION_ROLES[ActionID.TRANSACTION_TAX] is ActionRole.SUBSTANTIVE
    assert ACTION_ROLES[ActionID.COLLUSION_INVESTIGATION] is ActionRole.SUBSTANTIVE
    assert ACTION_ROLES[ActionID.PRICE_CEILING] is ActionRole.NEGATIVE_CONTROL
    assert ACTION_ROLES[ActionID.SUPPLY_SUBSIDY] is ActionRole.NEGATIVE_CONTROL
    assert ACTION_ROLES[ActionID.DO_NOTHING] is ActionRole.ABSTENTION_CONTROL


def test_checked_in_schema_matches_runtime_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    checked_in = json.loads((root / "configs/schemas/decision-v2.json").read_text())
    assert checked_in == json.loads(decision_schema_bytes())
