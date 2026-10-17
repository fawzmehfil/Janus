"""The five mutually exclusive Phase 0 interventions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field

from janus.core.config import StrictFrozenModel
from janus.core.ledger import PolicyFlows
from janus.core.randomness import DisturbanceTape
from janus.scenarios.families import LatentFamily, ScenarioConfig


class ActionID(StrEnum):
    DO_NOTHING = "do_nothing"
    PRICE_CEILING = "price_ceiling"
    TRANSACTION_TAX = "transaction_tax"
    SUPPLY_SUBSIDY = "supply_subsidy"
    COLLUSION_INVESTIGATION = "collusion_investigation"


class InterventionConfig(StrictFrozenModel):
    price_ceiling_multiplier: float = Field(default=1.10, gt=0)
    transaction_tax_rate: float = Field(default=0.10, ge=0)
    supply_subsidy_fraction: float = Field(default=0.15, ge=0)
    investigation_sensitivity: float = Field(default=0.85, ge=0, le=1)
    investigation_false_positive_rate: float = Field(default=0.05, ge=0, le=1)
    investigation_fine_fraction: float = Field(default=0.25, ge=0)
    price_ceiling_admin_cost: float = Field(default=1.0, ge=0)
    tax_admin_cost: float = Field(default=0.5, ge=0)
    subsidy_admin_cost: float = Field(default=0.5, ge=0)
    investigation_admin_cost: float = Field(default=5.0, ge=0)
    investigation_compliance_cost: float = Field(default=2.0, ge=0)
    false_positive_disruption_cost: float = Field(default=10.0, ge=0)


@dataclass(frozen=True, slots=True)
class ResolvedIntervention:
    action: ActionID
    price_cap: float | None
    policy_flows: PolicyFlows
    collusion_disabled: bool = False
    audit_detected: bool | None = None
    false_positive: bool = False


def resolve_intervention(
    action: ActionID,
    *,
    reference_price: float,
    scenario: ScenarioConfig,
    colluding_seller_ids: tuple[str, ...],
    tape: DisturbanceTape,
    decision_period: int,
    config: InterventionConfig | None = None,
) -> ResolvedIntervention:
    config = config or InterventionConfig()

    if action is ActionID.DO_NOTHING:
        return ResolvedIntervention(action, None, PolicyFlows())

    if action is ActionID.PRICE_CEILING:
        return ResolvedIntervention(
            action,
            config.price_ceiling_multiplier * reference_price,
            PolicyFlows(administration_cost=config.price_ceiling_admin_cost),
        )

    if action is ActionID.TRANSACTION_TAX:
        return ResolvedIntervention(
            action,
            None,
            PolicyFlows(
                buyer_tax_rate=config.transaction_tax_rate,
                administration_cost=config.tax_admin_cost,
            ),
        )

    if action is ActionID.SUPPLY_SUBSIDY:
        return ResolvedIntervention(
            action,
            None,
            PolicyFlows(
                seller_subsidy_per_unit=config.supply_subsidy_fraction * reference_price,
                administration_cost=config.subsidy_admin_cost,
            ),
        )

    is_collusion = scenario.family is LatentFamily.SELLER_COLLUSION
    probability = (
        config.investigation_sensitivity
        if is_collusion
        else config.investigation_false_positive_rate
    )
    detected = tape.bernoulli(probability, "investigation", decision_period, scenario.scenario_id)
    false_positive = detected and not is_collusion
    fines = (
        tuple(
            (seller_id, config.investigation_fine_fraction * reference_price)
            for seller_id in colluding_seller_ids
        )
        if detected and is_collusion
        else ()
    )
    return ResolvedIntervention(
        action,
        None,
        PolicyFlows(
            seller_fines=fines,
            administration_cost=config.investigation_admin_cost,
            compliance_cost=config.investigation_compliance_cost,
            false_positive_cost=config.false_positive_disruption_cost if false_positive else 0.0,
        ),
        collusion_disabled=detected and is_collusion,
        audit_detected=detected,
        false_positive=false_positive,
    )
