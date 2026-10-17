"""Versioned public observation contract with hidden-field exclusion."""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import pairwise
from math import log, sqrt

from pydantic import Field, model_validator

from janus.core.clearing import ClearingResult
from janus.core.config import StrictFrozenModel
from janus.core.state import MarketState
from janus.interventions.actions import ActionID

OBSERVATION_SCHEMA_VERSION = "janus-observation-v1"
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "family",
        "hidden_family",
        "latent_family",
        "marginal_costs",
        "true_value",
        "strategy",
        "coalition",
        "future_disturbances",
        "counterfactual_values",
        "oracle_action",
    }
)


class PublicObservationRow(StrictFrozenModel):
    period: int = Field(ge=0)
    transaction_price: float | None = Field(default=None, ge=0)
    no_trade: bool
    transaction_volume: int = Field(ge=0)
    aggregate_available_supply: int = Field(ge=0)
    unfilled_submitted_demand: int = Field(ge=0)
    aggregate_inventory: int = Field(ge=0)
    best_bid: float | None = Field(default=None, ge=0)
    best_ask: float | None = Field(default=None, ge=0)
    spread: float | None = None
    seller_hhi: float = Field(ge=0, le=1)
    active_firm_count: int = Field(ge=0)
    bankruptcies: int = Field(ge=0)
    realized_log_price_volatility: float = Field(ge=0)
    intervention: ActionID
    reference_price: float = Field(gt=0)
    reference_volume: float = Field(gt=0)
    reference_inventory: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_trade_marker(self) -> PublicObservationRow:
        if self.no_trade != (self.transaction_price is None):
            raise ValueError("no_trade must exactly match a missing transaction price")
        return self


class PublicObservationHistory(StrictFrozenModel):
    schema_version: str = OBSERVATION_SCHEMA_VERSION
    rows: tuple[PublicObservationRow, ...]


def realized_log_volatility(prices: Sequence[float | None], *, window: int = 5) -> float:
    observed = [price for price in prices if price is not None]
    observed = observed[-(window + 1) :]
    if len(observed) < 3:
        return 0.0
    returns = [log(right / left) for left, right in pairwise(observed)]
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return sqrt(variance)


def seller_hhi(state: MarketState) -> float:
    capacities = [seller.capacity for seller in state.sellers if seller.active]
    total = sum(capacities)
    if total == 0:
        return 0.0
    return sum((capacity / total) ** 2 for capacity in capacities)


def build_public_row(
    state: MarketState,
    clearing: ClearingResult,
    *,
    price_history: Sequence[float | None],
    intervention: ActionID,
    reference_price: float,
    reference_volume: float,
    reference_inventory: float,
) -> PublicObservationRow:
    spread = (
        None
        if clearing.best_bid is None or clearing.best_ask is None
        else clearing.best_ask - clearing.best_bid
    )
    return PublicObservationRow(
        period=state.period,
        transaction_price=clearing.transaction_price,
        no_trade=clearing.transaction_price is None,
        transaction_volume=clearing.quantity,
        aggregate_available_supply=clearing.available_supply,
        unfilled_submitted_demand=clearing.unfilled_demand,
        aggregate_inventory=sum(seller.inventory for seller in state.sellers if seller.active),
        best_bid=clearing.best_bid,
        best_ask=clearing.best_ask,
        spread=spread,
        seller_hhi=seller_hhi(state),
        active_firm_count=sum(seller.active for seller in state.sellers),
        bankruptcies=sum(not seller.active for seller in state.sellers),
        realized_log_price_volatility=realized_log_volatility(price_history),
        intervention=intervention,
        reference_price=reference_price,
        reference_volume=reference_volume,
        reference_inventory=reference_inventory,
    )


def serialize_public_history(history: PublicObservationHistory) -> bytes:
    payload = history.model_dump(mode="json")
    leaked = FORBIDDEN_PUBLIC_KEYS.intersection(_all_keys(payload))
    if leaked:
        raise ValueError(f"public observation contains forbidden keys: {sorted(leaked)}")
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list | tuple):
        return set().union(*(_all_keys(item) for item in value)) if value else set()
    return set()
