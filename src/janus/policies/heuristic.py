"""A deliberately simple public-observation-only policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from janus.interventions.actions import ActionID


@dataclass(frozen=True, slots=True)
class PublicHeuristicPolicy:
    inventory_ratio_threshold: float = 0.75
    concentration_threshold: float = 0.20
    price_ratio_threshold: float = 1.12
    excess_demand_ratio_threshold: float = 0.25

    def act(self, observation: bytes) -> ActionID:
        payload: dict[str, Any] = json.loads(observation)
        rows = payload["rows"]
        if not rows:
            return ActionID.DO_NOTHING
        row = rows[-1]
        inventory_ratio = row["aggregate_inventory"] / row["reference_inventory"]
        price = row["transaction_price"]
        price_ratio = 1.0 if price is None else price / row["reference_price"]
        demand_scale = max(1.0, row["transaction_volume"] + row["unfilled_submitted_demand"])
        excess_demand_ratio = row["unfilled_submitted_demand"] / demand_scale

        if row["seller_hhi"] >= self.concentration_threshold and price_ratio > 1.05:
            return ActionID.COLLUSION_INVESTIGATION
        if inventory_ratio < self.inventory_ratio_threshold:
            return ActionID.SUPPLY_SUBSIDY
        if price_ratio >= self.price_ratio_threshold and excess_demand_ratio < 0.10:
            return ActionID.TRANSACTION_TAX
        if excess_demand_ratio >= self.excess_demand_ratio_threshold:
            return ActionID.PRICE_CEILING
        return ActionID.DO_NOTHING
