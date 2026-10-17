"""Hand-checkable analytic market fixtures."""

from __future__ import annotations

from janus.core.clearing import Ask, Bid
from janus.core.config import MarketConfig
from janus.core.state import BuyerState, MarketState, SellerState


def analytic_market(config: MarketConfig | None = None) -> MarketState:
    """Return the 40-buyer/40-slot fixture with q*=21, p*=100, surplus=840."""

    config = config or MarketConfig()
    cash = config.initial_cash_multiple * config.reference_price
    buyers = tuple(
        BuyerState(
            buyer_id=f"buyer-{index:02d}",
            cash=cash,
            true_value=config.buyer_value_start - config.schedule_step * index,
        )
        for index in range(config.potential_buyers)
    )
    supply_slots = config.active_sellers * config.units_per_seller
    costs = tuple(
        config.seller_cost_start + config.schedule_step * index for index in range(supply_slots)
    )
    sellers = tuple(
        SellerState(
            seller_id=f"seller-{seller_index:02d}",
            cash=cash,
            inventory=config.units_per_seller,
            capacity=config.units_per_seller,
            marginal_costs=tuple(
                costs[seller_index + config.active_sellers * slot_index]
                for slot_index in range(config.units_per_seller)
            ),
        )
        for seller_index in range(config.active_sellers)
    )
    return MarketState(period=0, buyers=buyers, sellers=sellers)


def competitive_orders(state: MarketState) -> tuple[tuple[Bid, ...], tuple[Ask, ...]]:
    bids = tuple(
        Bid(
            order_id=f"bid:{state.period}:{buyer.buyer_id}",
            buyer_id=buyer.buyer_id,
            amount=buyer.true_value,
            true_value=buyer.true_value,
        )
        for buyer in state.buyers
        if buyer.active
    )
    asks: list[Ask] = []
    for seller in state.sellers:
        if not seller.active:
            continue
        for unit, cost in enumerate(seller.marginal_costs[: seller.inventory]):
            asks.append(
                Ask(
                    order_id=f"ask:{state.period}:{seller.seller_id}:{unit}",
                    seller_id=seller.seller_id,
                    amount=cost,
                    marginal_cost=cost,
                )
            )
    return bids, tuple(asks)
