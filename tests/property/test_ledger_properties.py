from hypothesis import given
from hypothesis import strategies as st

from janus.core.clearing import Trade
from janus.core.ledger import PolicyFlows, compute_period_ledger


def make_trade(value: float = 120, cost: float = 60, price: float = 90) -> Trade:
    return Trade("bid", "ask", "buyer", "seller", price, value, cost)


@given(
    tax_rate=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    subsidy=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    fine=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
)
def test_pure_transfers_cancel_when_behavior_is_fixed(
    tax_rate: float, subsidy: float, fine: float
) -> None:
    trade = make_trade()
    ledger = compute_period_ledger(
        (trade,),
        policy=PolicyFlows(
            buyer_tax_rate=tax_rate,
            seller_subsidy_per_unit=subsidy,
            seller_fines=(("seller", fine),),
        ),
        marginal_cost_of_public_funds=0,
    )
    assert abs(ledger.transfer_balance) < 1e-9
    assert abs(ledger.net_social_surplus - (trade.buyer_value - trade.seller_cost)) < 1e-8


def test_real_and_financing_costs_reduce_surplus_once() -> None:
    ledger = compute_period_ledger(
        (make_trade(),),
        policy=PolicyFlows(seller_subsidy_per_unit=10, administration_cost=3),
        marginal_cost_of_public_funds=0.2,
    )
    assert ledger.government_balance == -10
    assert ledger.financing_cost == 2
    assert ledger.net_social_surplus == 55
