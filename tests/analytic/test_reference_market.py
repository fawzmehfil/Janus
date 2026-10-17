from janus.core.clearing import clear_market
from janus.core.fixtures import analytic_market, competitive_orders
from janus.core.randomness import DisturbanceTape


def test_reference_market_matches_hand_calculation() -> None:
    state = analytic_market()
    bids, asks = competitive_orders(state)
    tape = DisturbanceTape(7, "reference")
    priority = tape.tie_priority(tuple(order.order_id for order in (*bids, *asks)), 0)

    result = clear_market(bids, asks, tie_priority=priority)

    assert result.quantity == 21
    assert result.transaction_price == 100.0
    assert sum(trade.buyer_value - trade.seller_cost for trade in result.trades) == 840.0


def test_small_hand_solved_markets() -> None:
    from janus.core.clearing import Ask, Bid

    cases = [
        ([100], [60], 1, 80.0, 40.0),
        ([110, 90], [50, 80], 2, 85.0, 70.0),
        ([120, 100, 70], [40, 80, 90], 2, 90.0, 100.0),
    ]
    for bid_values, ask_costs, quantity, price, surplus in cases:
        bids = tuple(Bid(f"b{i}", f"buyer{i}", value, value) for i, value in enumerate(bid_values))
        asks = tuple(Ask(f"a{i}", f"seller{i}", cost, cost) for i, cost in enumerate(ask_costs))
        ids = [order.order_id for order in (*bids, *asks)]
        priority = {order_id: i for i, order_id in enumerate(ids)}
        result = clear_market(bids, asks, tie_priority=priority)
        assert result.quantity == quantity
        assert result.transaction_price == price
        assert sum(trade.buyer_value - trade.seller_cost for trade in result.trades) == surplus
