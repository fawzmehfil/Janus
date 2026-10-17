"""Buyer order generation from true values and explicit behavioral distortions."""

from __future__ import annotations

from janus.core.clearing import Bid
from janus.core.state import MarketState
from janus.scenarios.families import MarketPrimitives


def generate_bids(
    state: MarketState,
    primitives: MarketPrimitives,
    *,
    buyer_tax_rate: float = 0.0,
) -> tuple[Bid, ...]:
    """Generate one unit bid per active buyer.

    A buyer-side tax is incorporated into the submitted exchange bid so the
    gross payment cannot exceed the buyer's willingness to pay.
    """

    if buyer_tax_rate < 0:
        raise ValueError("buyer tax rate must be nonnegative")
    bids: list[Bid] = []
    for buyer in state.buyers:
        if not buyer.active:
            continue
        true_value = primitives.buyer_true_values[buyer.buyer_id]
        distortion = primitives.buyer_bid_adjustments.get(buyer.buyer_id, 0.0)
        gross_willingness = max(0.0, true_value + distortion)
        exchange_bid = gross_willingness / (1 + buyer_tax_rate)
        bids.append(
            Bid(
                order_id=f"bid:{state.period}:{buyer.buyer_id}",
                buyer_id=buyer.buyer_id,
                amount=exchange_bid,
                true_value=true_value,
            )
        )
    return tuple(bids)
