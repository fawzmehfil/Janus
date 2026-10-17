import pytest
from hypothesis import given
from hypothesis import strategies as st

from janus.core.clearing import Ask, Bid, ClearingPriceRule, clear_market, slow_clear_market


@given(
    bid_values=st.lists(
        st.floats(min_value=0, max_value=1_000, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=12,
    ),
    ask_values=st.lists(
        st.floats(min_value=0, max_value=1_000, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=12,
    ),
    cap=st.one_of(
        st.none(), st.floats(min_value=0, max_value=1_000, allow_nan=False, allow_infinity=False)
    ),
)
@pytest.mark.parametrize("price_rule", ["midpoint", "marginal_ask"])
def test_clearing_feasibility_and_reference_agreement(
    bid_values: list[float],
    ask_values: list[float],
    cap: float | None,
    price_rule: ClearingPriceRule,
) -> None:
    bids = tuple(Bid(f"b{i}", f"buyer{i}", value, value) for i, value in enumerate(bid_values))
    asks = tuple(Ask(f"a{i}", f"seller{i}", value, value) for i, value in enumerate(ask_values))
    ids = [order.order_id for order in (*bids, *asks)]
    priority = {order_id: index for index, order_id in enumerate(reversed(ids))}

    result = clear_market(
        bids,
        asks,
        tie_priority=priority,
        price_cap=cap,
        price_rule=price_rule,
    )
    reference = slow_clear_market(
        bids,
        asks,
        tie_priority=priority,
        price_cap=cap,
        price_rule=price_rule,
    )

    assert result == reference
    assert result.quantity <= min(len(bids), len(asks))
    for trade in result.trades:
        assert (
            trade.seller_cost
            <= trade.price
            <= next(bid.amount for bid in bids if bid.order_id == trade.bid_order_id)
        )
        if cap is not None:
            assert trade.price <= cap


def test_no_trade_has_missing_transaction_price() -> None:
    bids = (Bid("b", "buyer", 10, 10),)
    asks = (Ask("a", "seller", 20, 20),)
    result = clear_market(bids, asks, tie_priority={"a": 0, "b": 1})
    assert result.quantity == 0
    assert result.transaction_price is None
    assert result.best_bid == 10
    assert result.best_ask == 20
