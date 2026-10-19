from __future__ import annotations

import pytest

from janus.core.clearing import Trade
from janus.core.ledger import compute_period_ledger
from janus.core.randomness import DisturbanceTape
from janus.interventions.actions import ActionID, resolve_intervention
from janus.scenarios.families import LatentFamily, ScenarioConfig
from janus.v2.contracts import ACTION_ROLES, ActionRole


def _scenario() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="negative-control-microcase",
        family=LatentFamily.SUPPLY_SHORTAGE,
    )


def test_negative_control_roles_are_frozen_and_semantically_distinct() -> None:
    assert ACTION_ROLES[ActionID.PRICE_CEILING] is ActionRole.NEGATIVE_CONTROL
    assert ACTION_ROLES[ActionID.SUPPLY_SUBSIDY] is ActionRole.NEGATIVE_CONTROL
    assert ACTION_ROLES[ActionID.TRANSACTION_TAX] is ActionRole.SUBSTANTIVE
    assert ACTION_ROLES[ActionID.COLLUSION_INVESTIGATION] is ActionRole.SUBSTANTIVE
    assert ACTION_ROLES[ActionID.DO_NOTHING] is ActionRole.ABSTENTION_CONTROL


def test_price_ceiling_negative_control_has_exact_cap_and_real_cost() -> None:
    intervention = resolve_intervention(
        ActionID.PRICE_CEILING,
        reference_price=100,
        scenario=_scenario(),
        colluding_seller_ids=(),
        tape=DisturbanceTape(4, "negative-control-ceiling"),
        decision_period=20,
    )
    assert intervention.price_cap == pytest.approx(110)
    assert intervention.policy_flows.administration_cost == 5
    ledger = compute_period_ledger((), policy=intervention.policy_flows)
    assert ledger.real_cost == 5
    assert ledger.net_social_surplus == -5


def test_supply_subsidy_is_a_balanced_transfer_with_financing_burden() -> None:
    intervention = resolve_intervention(
        ActionID.SUPPLY_SUBSIDY,
        reference_price=100,
        scenario=_scenario(),
        colluding_seller_ids=(),
        tape=DisturbanceTape(5, "negative-control-subsidy"),
        decision_period=20,
    )
    assert intervention.policy_flows.seller_subsidy_per_unit == 15
    trade = Trade("bid", "ask", "buyer", "seller", 100, 130, 80)
    ledger = compute_period_ledger((trade,), policy=intervention.policy_flows)
    assert ledger.transfer_balance == 0
    assert ledger.government_balance == -15
    assert ledger.financing_cost == 3
    assert ledger.real_cost == 0.5
