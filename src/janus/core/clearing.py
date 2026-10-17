"""Deterministic periodic sealed-bid double-auction clearing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Literal

ClearingPriceRule = Literal["midpoint", "marginal_ask"]


def _validate_amount(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class Bid:
    order_id: str
    buyer_id: str
    amount: float
    true_value: float

    def __post_init__(self) -> None:
        _validate_amount("bid", self.amount)
        _validate_amount("true value", self.true_value)


@dataclass(frozen=True, slots=True)
class Ask:
    order_id: str
    seller_id: str
    amount: float
    marginal_cost: float

    def __post_init__(self) -> None:
        _validate_amount("ask", self.amount)
        _validate_amount("marginal cost", self.marginal_cost)


@dataclass(frozen=True, slots=True)
class Trade:
    bid_order_id: str
    ask_order_id: str
    buyer_id: str
    seller_id: str
    price: float
    buyer_value: float
    seller_cost: float


@dataclass(frozen=True, slots=True)
class ClearingResult:
    trades: tuple[Trade, ...]
    transaction_price: float | None
    best_bid: float | None
    best_ask: float | None
    submitted_demand: int
    available_supply: int

    @property
    def quantity(self) -> int:
        return len(self.trades)

    @property
    def unfilled_demand(self) -> int:
        return self.submitted_demand - self.quantity


def _validate_orders(
    bids: Sequence[Bid], asks: Sequence[Ask], tie_priority: Mapping[str, int]
) -> None:
    ids = [bid.order_id for bid in bids] + [ask.order_id for ask in asks]
    if len(ids) != len(set(ids)):
        raise ValueError("order IDs must be unique within a period")
    missing = set(ids).difference(tie_priority)
    if missing:
        raise ValueError(f"tie-priority map is missing order IDs: {sorted(missing)}")


def _sorted_orders(
    bids: Sequence[Bid], asks: Sequence[Ask], tie_priority: Mapping[str, int]
) -> tuple[list[Bid], list[Ask]]:
    _validate_orders(bids, asks, tie_priority)
    sorted_bids = sorted(
        bids, key=lambda bid: (-bid.amount, tie_priority[bid.order_id], bid.order_id)
    )
    sorted_asks = sorted(
        asks, key=lambda ask: (ask.amount, tie_priority[ask.order_id], ask.order_id)
    )
    return sorted_bids, sorted_asks


def clear_market(
    bids: Sequence[Bid],
    asks: Sequence[Ask],
    *,
    tie_priority: Mapping[str, int],
    price_cap: float | None = None,
    price_rule: ClearingPriceRule = "midpoint",
) -> ClearingResult:
    """Clear a uniform-price auction, optionally subject to a price ceiling."""

    if price_cap is not None:
        _validate_amount("price cap", price_cap)
    if price_rule not in ("midpoint", "marginal_ask"):
        raise ValueError(f"unknown clearing price rule: {price_rule}")
    sorted_bids, sorted_asks = _sorted_orders(bids, asks, tie_priority)

    quantity = 0
    for bid, ask in zip(sorted_bids, sorted_asks, strict=False):
        maximum_price = bid.amount if price_cap is None else min(bid.amount, price_cap)
        if ask.amount > maximum_price:
            break
        quantity += 1

    if quantity == 0:
        return ClearingResult(
            trades=(),
            transaction_price=None,
            best_bid=sorted_bids[0].amount if sorted_bids else None,
            best_ask=sorted_asks[0].amount if sorted_asks else None,
            submitted_demand=len(bids),
            available_supply=len(asks),
        )

    marginal_bid = sorted_bids[quantity - 1].amount
    marginal_ask = sorted_asks[quantity - 1].amount
    price = marginal_ask if price_rule == "marginal_ask" else (marginal_bid + marginal_ask) / 2
    if price_cap is not None:
        price = min(price, price_cap)
    if price < marginal_ask or price > marginal_bid:
        raise AssertionError("clearing price violates marginal order feasibility")

    trades = tuple(
        Trade(
            bid_order_id=bid.order_id,
            ask_order_id=ask.order_id,
            buyer_id=bid.buyer_id,
            seller_id=ask.seller_id,
            price=price,
            buyer_value=bid.true_value,
            seller_cost=ask.marginal_cost,
        )
        for bid, ask in zip(sorted_bids[:quantity], sorted_asks[:quantity], strict=True)
    )
    return ClearingResult(
        trades=trades,
        transaction_price=price,
        best_bid=sorted_bids[0].amount,
        best_ask=sorted_asks[0].amount,
        submitted_demand=len(bids),
        available_supply=len(asks),
    )


def slow_clear_market(
    bids: Sequence[Bid],
    asks: Sequence[Ask],
    *,
    tie_priority: Mapping[str, int],
    price_cap: float | None = None,
    price_rule: ClearingPriceRule = "midpoint",
) -> ClearingResult:
    """Transparent reference implementation used for differential tests."""

    sorted_bids, sorted_asks = _sorted_orders(bids, asks, tie_priority)
    feasible_quantities = [0]
    for candidate in range(1, min(len(sorted_bids), len(sorted_asks)) + 1):
        bid = sorted_bids[candidate - 1]
        ask = sorted_asks[candidate - 1]
        limit = bid.amount if price_cap is None else min(bid.amount, price_cap)
        if ask.amount <= limit:
            feasible_quantities.append(candidate)
    quantity = max(feasible_quantities)
    result = clear_market(
        bids,
        asks,
        tie_priority=tie_priority,
        price_cap=price_cap,
        price_rule=price_rule,
    )
    if result.quantity != quantity:
        raise AssertionError("production and reference clearing quantities differ")
    return result


def clearing_results_equal(left: ClearingResult, right: ClearingResult) -> bool:
    if left.quantity != right.quantity or left.trades != right.trades:
        return False
    if left.transaction_price is None or right.transaction_price is None:
        return left.transaction_price is right.transaction_price
    return isclose(left.transaction_price, right.transaction_price, abs_tol=1e-12)
