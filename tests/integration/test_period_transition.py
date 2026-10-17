from janus.core.config import AccountingConfig, MarketConfig
from janus.core.engine import run_period
from janus.core.fixtures import analytic_market, competitive_orders
from janus.core.randomness import DisturbanceTape


def test_period_transition_conserves_units_and_cash_transfers() -> None:
    config = MarketConfig(holding_cost_per_unit=0)
    state = analytic_market(config)
    bids, asks = competitive_orders(state)
    tape = DisturbanceTape(42, "period")
    priority = tape.tie_priority(tuple(order.order_id for order in (*bids, *asks)), 0)

    result = run_period(
        state,
        bids,
        asks,
        tie_priority=priority,
        market_config=config,
        accounting_config=AccountingConfig(marginal_cost_of_public_funds=0),
    )

    assert result.next_state.period == 1
    assert result.clearing.quantity == 21
    assert all(seller.inventory >= 0 for seller in result.next_state.sellers)
    old_cash = sum(buyer.cash for buyer in state.buyers) + sum(
        seller.cash for seller in state.sellers
    )
    new_cash = (
        sum(buyer.cash for buyer in result.next_state.buyers)
        + sum(seller.cash for seller in result.next_state.sellers)
        + result.next_state.government_balance
    )
    assert abs(old_cash - new_cash) < 1e-9
