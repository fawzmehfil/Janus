"""Four causal mechanisms that alter distinct economic primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import ceil
from typing import Self

from pydantic import Field, model_validator

from janus.core.config import StrictFrozenModel
from janus.core.state import MarketState


class LatentFamily(StrEnum):
    SUPPLY_SHORTAGE = "supply_shortage"
    DEMAND_INCREASE = "demand_increase"
    SELLER_COLLUSION = "seller_collusion"
    MOMENTUM_DEMAND = "momentum_demand"


class ScenarioConfig(StrictFrozenModel):
    scenario_id: str
    family: LatentFamily
    shock_start: int = Field(default=20, ge=0)
    shock_duration: int = Field(default=20, ge=1)
    reference_price: float = Field(default=100.0, gt=0)

    supply_cost_shift_fraction: float = Field(default=0.20, ge=0, le=1)
    supply_capacity_loss_fraction: float = Field(default=0.20, ge=0, lt=1)
    affected_seller_fraction: float = Field(default=0.50, gt=0, le=1)

    demand_value_shift_fraction: float = Field(default=0.20, ge=0, le=1)
    affected_buyer_fraction: float = Field(default=0.50, gt=0, le=1)

    colluding_seller_fraction: float = Field(default=0.50, gt=0, le=1)
    collusive_markup_fraction: float = Field(default=0.20, ge=0, le=1)
    collusive_withholding_fraction: float = Field(default=0.20, ge=0, lt=1)

    momentum_buyer_fraction: float = Field(default=0.30, gt=0, le=1)
    momentum_feedback: float = Field(default=2.0, ge=0)
    momentum_distortion_cap_fraction: float = Field(default=0.30, ge=0, le=1)
    momentum_decay: float = Field(default=0.85, gt=0, le=1)

    affected_seller_ids: tuple[str, ...] = ()
    affected_buyer_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_named_actors(self) -> Self:
        if len(set(self.affected_seller_ids)) != len(self.affected_seller_ids):
            raise ValueError("affected seller IDs must be unique")
        if len(set(self.affected_buyer_ids)) != len(self.affected_buyer_ids):
            raise ValueError("affected buyer IDs must be unique")
        return self

    def shock_active(self, period: int) -> bool:
        return self.shock_start <= period < self.shock_start + self.shock_duration


@dataclass(frozen=True, slots=True)
class MarketPrimitives:
    """Complete economic/behavioral primitives used to form submitted orders."""

    buyer_true_values: dict[str, float]
    buyer_bid_adjustments: dict[str, float]
    seller_marginal_costs: dict[str, tuple[float, ...]]
    seller_offer_limits: dict[str, int]
    seller_markups: dict[str, float]
    colluding_seller_ids: tuple[str, ...] = ()
    momentum_buyer_ids: tuple[str, ...] = ()


def _select_ids(
    all_ids: tuple[str, ...], explicit: tuple[str, ...], fraction: float
) -> tuple[str, ...]:
    if explicit:
        unknown = set(explicit).difference(all_ids)
        if unknown:
            raise ValueError(f"scenario references unknown actor IDs: {sorted(unknown)}")
        return explicit
    count = max(1, ceil(len(all_ids) * fraction))
    return tuple(sorted(all_ids)[:count])


def primitives_for_period(
    state: MarketState,
    scenario: ScenarioConfig,
    *,
    recent_log_returns: tuple[float, ...] = (),
    collusion_disabled: bool = False,
) -> MarketPrimitives:
    buyer_values = {buyer.buyer_id: buyer.true_value for buyer in state.buyers}
    buyer_adjustments = {buyer.buyer_id: 0.0 for buyer in state.buyers}
    seller_costs = {seller.seller_id: tuple(seller.marginal_costs) for seller in state.sellers}
    seller_limits = {seller.seller_id: seller.inventory for seller in state.sellers}
    seller_markups = {seller.seller_id: 0.0 for seller in state.sellers}
    colluders: tuple[str, ...] = ()
    momentum_buyers: tuple[str, ...] = ()

    if not scenario.shock_active(state.period):
        return MarketPrimitives(
            buyer_values,
            buyer_adjustments,
            seller_costs,
            seller_limits,
            seller_markups,
        )

    seller_ids = tuple(seller.seller_id for seller in state.sellers if seller.active)
    buyer_ids = tuple(buyer.buyer_id for buyer in state.buyers if buyer.active)

    if scenario.family is LatentFamily.SUPPLY_SHORTAGE:
        affected = _select_ids(
            seller_ids, scenario.affected_seller_ids, scenario.affected_seller_fraction
        )
        cost_shift = scenario.reference_price * scenario.supply_cost_shift_fraction
        for seller_id in affected:
            seller_costs[seller_id] = tuple(cost + cost_shift for cost in seller_costs[seller_id])
            seller_limits[seller_id] = max(
                0,
                int(seller_limits[seller_id] * (1 - scenario.supply_capacity_loss_fraction)),
            )

    elif scenario.family is LatentFamily.DEMAND_INCREASE:
        affected = _select_ids(
            buyer_ids, scenario.affected_buyer_ids, scenario.affected_buyer_fraction
        )
        value_shift = scenario.reference_price * scenario.demand_value_shift_fraction
        for buyer_id in affected:
            buyer_values[buyer_id] += value_shift

    elif scenario.family is LatentFamily.SELLER_COLLUSION:
        colluders = _select_ids(
            seller_ids, scenario.affected_seller_ids, scenario.colluding_seller_fraction
        )
        if not collusion_disabled:
            markup = scenario.reference_price * scenario.collusive_markup_fraction
            for seller_id in colluders:
                seller_markups[seller_id] = markup
                seller_limits[seller_id] = max(
                    0,
                    int(seller_limits[seller_id] * (1 - scenario.collusive_withholding_fraction)),
                )

    elif scenario.family is LatentFamily.MOMENTUM_DEMAND:
        momentum_buyers = _select_ids(
            buyer_ids, scenario.affected_buyer_ids, scenario.momentum_buyer_fraction
        )
        weighted_return = sum(
            value * scenario.momentum_decay**lag
            for lag, value in enumerate(reversed(recent_log_returns))
        )
        distortion = min(
            scenario.reference_price * scenario.momentum_distortion_cap_fraction,
            max(0.0, scenario.reference_price * scenario.momentum_feedback * weighted_return),
        )
        for buyer_id in momentum_buyers:
            buyer_adjustments[buyer_id] = distortion

    return MarketPrimitives(
        buyer_true_values=buyer_values,
        buyer_bid_adjustments=buyer_adjustments,
        seller_marginal_costs=seller_costs,
        seller_offer_limits=seller_limits,
        seller_markups=seller_markups,
        colluding_seller_ids=colluders,
        momentum_buyer_ids=momentum_buyers,
    )
