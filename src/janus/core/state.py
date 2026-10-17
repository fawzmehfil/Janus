"""Immutable market state and order-domain types."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _require_finite_nonnegative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class BuyerState:
    buyer_id: str
    cash: float
    true_value: float
    active: bool = True

    def __post_init__(self) -> None:
        _require_finite_nonnegative("cash", self.cash)
        _require_finite_nonnegative("true_value", self.true_value)


@dataclass(frozen=True, slots=True)
class SellerState:
    seller_id: str
    cash: float
    inventory: int
    capacity: int
    marginal_costs: tuple[float, ...]
    active: bool = True

    def __post_init__(self) -> None:
        _require_finite_nonnegative("cash", self.cash)
        if self.inventory < 0 or self.capacity < 0:
            raise ValueError("inventory and capacity must be nonnegative")
        if not self.marginal_costs:
            raise ValueError("a seller requires at least one marginal-cost slot")
        for cost in self.marginal_costs:
            _require_finite_nonnegative("marginal cost", cost)


@dataclass(frozen=True, slots=True)
class MarketState:
    period: int
    buyers: tuple[BuyerState, ...]
    sellers: tuple[SellerState, ...]
    government_balance: float = 0.0
    last_price: float | None = None
    recent_transaction_prices: tuple[float, ...] = ()
    cumulative_real_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.period < 0:
            raise ValueError("period must be nonnegative")
        if len({buyer.buyer_id for buyer in self.buyers}) != len(self.buyers):
            raise ValueError("buyer IDs must be unique")
        if len({seller.seller_id for seller in self.sellers}) != len(self.sellers):
            raise ValueError("seller IDs must be unique")
        if self.last_price is not None:
            _require_finite_nonnegative("last_price", self.last_price)
        for price in self.recent_transaction_prices:
            _require_finite_nonnegative("recent transaction price", price)
        if len(self.recent_transaction_prices) > 4:
            raise ValueError("market state retains at most four recent transaction prices")
        _require_finite_nonnegative("cumulative_real_cost", self.cumulative_real_cost)

    def clone(self) -> MarketState:
        """Return a structurally independent value copy suitable for branching."""

        return MarketState(
            period=self.period,
            buyers=tuple(
                BuyerState(b.buyer_id, b.cash, b.true_value, b.active) for b in self.buyers
            ),
            sellers=tuple(
                SellerState(
                    s.seller_id,
                    s.cash,
                    s.inventory,
                    s.capacity,
                    tuple(s.marginal_costs),
                    s.active,
                )
                for s in self.sellers
            ),
            government_balance=self.government_balance,
            last_price=self.last_price,
            recent_transaction_prices=tuple(self.recent_transaction_prices),
            cumulative_real_cost=self.cumulative_real_cost,
        )
