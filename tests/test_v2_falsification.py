from __future__ import annotations

from janus.v2.unblind import _scored_action


def _actions() -> dict[str, dict[str, object]]:
    return {
        "do_nothing": {"means": {"primary": 10.0}},
        "transaction_tax": {"means": {"primary": 8.0}},
    }


def test_shuffling_diagnosis_does_not_mechanically_change_welfare_action() -> None:
    base = {
        "validation_status": "valid",
        "parsed_decision": {
            "diagnosis": "supply_shortage",
            "diagnosis_probabilities": {},
            "action": "do_nothing",
            "confidence": 0.5,
        },
    }
    shuffled = {
        **base,
        "parsed_decision": {
            **base["parsed_decision"],
            "diagnosis": "seller_collusion",
        },
    }
    assert _scored_action(base, _actions()) == _scored_action(shuffled, _actions())


def test_invalid_response_fallback_is_exactly_the_worst_action() -> None:
    artifact = {
        "validation_status": "incomplete",
        "parsed_decision": None,
    }
    assert _scored_action(artifact, _actions()) == "transaction_tax"
