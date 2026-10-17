import json

import pytest

from janus.core.clearing import clear_market
from janus.core.fixtures import analytic_market, competitive_orders
from janus.core.randomness import DisturbanceTape
from janus.evaluation.observations import (
    FORBIDDEN_PUBLIC_KEYS,
    PublicObservationHistory,
    build_public_row,
    serialize_public_history,
)
from janus.interventions.actions import ActionID


def test_public_serialization_is_deterministic_and_contains_no_hidden_fields() -> None:
    state = analytic_market()
    bids, asks = competitive_orders(state)
    tape = DisturbanceTape(4, "observation")
    ids = tuple(bid.order_id for bid in bids) + tuple(ask.order_id for ask in asks)
    clearing = clear_market(bids, asks, tie_priority=tape.tie_priority(ids, 0))
    row = build_public_row(
        state,
        clearing,
        price_history=(100.0,),
        intervention=ActionID.DO_NOTHING,
        reference_price=100,
        reference_volume=21,
        reference_inventory=40,
    )
    history = PublicObservationHistory(rows=(row,))
    first = serialize_public_history(history)
    second = serialize_public_history(history)
    assert first == second
    payload = json.loads(first)
    serialized = first.decode()
    assert payload["schema_version"] == "janus-observation-v1"
    assert not any(f'"{key}"' in serialized for key in FORBIDDEN_PUBLIC_KEYS)


def test_no_trade_marker_must_match_missing_price() -> None:
    from janus.evaluation.observations import PublicObservationRow

    with pytest.raises(ValueError):
        PublicObservationRow(
            period=0,
            transaction_price=100,
            no_trade=True,
            transaction_volume=0,
            aggregate_available_supply=1,
            unfilled_submitted_demand=1,
            aggregate_inventory=1,
            seller_hhi=1,
            active_firm_count=1,
            bankruptcies=0,
            realized_log_price_volatility=0,
            intervention=ActionID.DO_NOTHING,
            reference_price=100,
            reference_volume=1,
            reference_inventory=1,
        )
