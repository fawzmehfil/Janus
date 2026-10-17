"""Strict, immutable configuration schemas for the local simulator."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictFrozenModel(BaseModel):
    """Base model that rejects unknown fields and cannot mutate after validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MarketConfig(StrictFrozenModel):
    """Phase 0 market mechanics that are independent of a hidden scenario."""

    potential_buyers: int = Field(default=40, ge=1)
    active_sellers: int = Field(default=8, ge=1)
    units_per_seller: int = Field(default=5, ge=1)
    initial_cash_multiple: float = Field(default=10.0, gt=0)
    reference_price: float = Field(default=100.0, gt=0)
    buyer_value_start: float = Field(default=140.0, ge=0)
    seller_cost_start: float = Field(default=60.0, ge=0)
    schedule_step: float = Field(default=2.0, gt=0)
    holding_cost_per_unit: float = Field(default=0.0, ge=0)
    allow_credit: bool = False

    @model_validator(mode="after")
    def validate_supply_slots(self) -> MarketConfig:
        if self.active_sellers * self.units_per_seller < 1:
            raise ValueError("the market must contain at least one supply slot")
        return self


class AccountingConfig(StrictFrozenModel):
    """Normative parameters used by primary surplus accounting."""

    discount_factor: float = Field(default=1.0, gt=0, le=1)
    marginal_cost_of_public_funds: float = Field(default=0.20, ge=0)


class SimulationConfig(StrictFrozenModel):
    """Timing and root randomness configuration."""

    pre_shock_periods: int = Field(default=20, ge=1)
    observation_periods: int = Field(default=10, ge=1)
    recovery_periods: int = Field(default=20, ge=1)
    root_seed: int = Field(default=20261016, ge=0, le=2**63 - 1)
    market: MarketConfig = Field(default_factory=MarketConfig)
    accounting: AccountingConfig = Field(default_factory=AccountingConfig)

    @property
    def decision_period(self) -> int:
        return self.pre_shock_periods + self.observation_periods
