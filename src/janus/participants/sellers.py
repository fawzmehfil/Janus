"""Seller order generation from costs, inventory, and explicit strategies."""

from __future__ import annotations

from janus.core.clearing import Ask
from janus.core.state import MarketState
from janus.scenarios.families import MarketPrimitives


def generate_asks(
    state: MarketState,
    primitives: MarketPrimitives,
    *,
    seller_subsidy_per_unit: float = 0.0,
) -> tuple[Ask, ...]:
    if seller_subsidy_per_unit < 0:
        raise ValueError("seller subsidy must be nonnegative")
    asks: list[Ask] = []
    for seller in state.sellers:
        if not seller.active:
            continue
        costs = primitives.seller_marginal_costs[seller.seller_id]
        offer_limit = min(
            seller.inventory,
            len(costs),
            primitives.seller_offer_limits[seller.seller_id],
        )
        markup = primitives.seller_markups.get(seller.seller_id, 0.0)
        for unit, cost in enumerate(costs[:offer_limit]):
            asks.append(
                Ask(
                    order_id=f"ask:{state.period}:{seller.seller_id}:{unit}",
                    seller_id=seller.seller_id,
                    amount=max(0.0, cost + markup - seller_subsidy_per_unit),
                    marginal_cost=cost,
                )
            )
    return tuple(asks)
