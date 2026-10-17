from dataclasses import replace

from janus.core.clearing import clear_market
from janus.core.fixtures import analytic_market
from janus.core.randomness import DisturbanceTape
from janus.interventions.actions import (
    ActionID,
    InterventionConfig,
    resolve_intervention,
)
from janus.participants.buyers import generate_bids
from janus.participants.sellers import generate_asks
from janus.scenarios.families import LatentFamily, ScenarioConfig, primitives_for_period


def config_for(family: LatentFamily) -> ScenarioConfig:
    return ScenarioConfig(scenario_id=f"action-{family}", family=family)


def test_all_action_ids_are_stable() -> None:
    assert [action.value for action in ActionID] == [
        "do_nothing",
        "price_ceiling",
        "transaction_tax",
        "supply_subsidy",
        "collusion_investigation",
    ]


def test_price_ceiling_never_clears_above_cap() -> None:
    state = replace(analytic_market(), period=20)
    scenario = config_for(LatentFamily.DEMAND_INCREASE)
    primitives = primitives_for_period(state, scenario)
    tape = DisturbanceTape(1, "ceiling")
    action = resolve_intervention(
        ActionID.PRICE_CEILING,
        reference_price=100,
        scenario=scenario,
        colluding_seller_ids=(),
        tape=tape,
        decision_period=20,
    )
    bids = generate_bids(state, primitives)
    asks = generate_asks(state, primitives)
    ids = tuple(bid.order_id for bid in bids) + tuple(ask.order_id for ask in asks)
    result = clear_market(
        bids,
        asks,
        tie_priority=tape.tie_priority(ids, 20),
        price_cap=action.price_cap,
    )
    assert result.transaction_price is not None
    assert action.price_cap is not None
    assert result.transaction_price <= action.price_cap


def test_tax_reduces_exchange_bid_and_subsidy_reduces_ask() -> None:
    state = replace(analytic_market(), period=20)
    scenario = config_for(LatentFamily.DEMAND_INCREASE)
    primitives = primitives_for_period(state, scenario)
    tape = DisturbanceTape(1, "tax-subsidy")
    tax = resolve_intervention(
        ActionID.TRANSACTION_TAX,
        reference_price=100,
        scenario=scenario,
        colluding_seller_ids=(),
        tape=tape,
        decision_period=20,
    )
    subsidy = resolve_intervention(
        ActionID.SUPPLY_SUBSIDY,
        reference_price=100,
        scenario=scenario,
        colluding_seller_ids=(),
        tape=tape,
        decision_period=20,
    )
    baseline_bid = generate_bids(state, primitives)[0]
    taxed_bid = generate_bids(state, primitives, buyer_tax_rate=tax.policy_flows.buyer_tax_rate)[0]
    baseline_ask = generate_asks(state, primitives)[0]
    subsidized_ask = generate_asks(
        state,
        primitives,
        seller_subsidy_per_unit=subsidy.policy_flows.seller_subsidy_per_unit,
    )[0]
    assert taxed_bid.amount < baseline_bid.amount
    assert subsidized_ask.amount < baseline_ask.amount


def test_detected_investigation_disables_collusion_and_levies_transfers() -> None:
    state = replace(analytic_market(), period=20)
    scenario = config_for(LatentFamily.SELLER_COLLUSION)
    primitives = primitives_for_period(state, scenario)
    action = resolve_intervention(
        ActionID.COLLUSION_INVESTIGATION,
        reference_price=100,
        scenario=scenario,
        colluding_seller_ids=primitives.colluding_seller_ids,
        tape=DisturbanceTape(2, "investigation"),
        decision_period=20,
        config=InterventionConfig(investigation_sensitivity=1),
    )
    assert action.audit_detected is True
    assert action.collusion_disabled is True
    assert {seller for seller, _ in action.policy_flows.seller_fines} == set(
        primitives.colluding_seller_ids
    )


def test_false_positive_is_explicit_real_cost() -> None:
    scenario = config_for(LatentFamily.SUPPLY_SHORTAGE)
    action = resolve_intervention(
        ActionID.COLLUSION_INVESTIGATION,
        reference_price=100,
        scenario=scenario,
        colluding_seller_ids=(),
        tape=DisturbanceTape(3, "false-positive"),
        decision_period=20,
        config=InterventionConfig(investigation_false_positive_rate=1),
    )
    assert action.false_positive
    assert action.policy_flows.false_positive_cost > 0


def test_zero_magnitude_interventions_match_do_nothing_behavior() -> None:
    scenario = config_for(LatentFamily.DEMAND_INCREASE)
    tape = DisturbanceTape(1, "zero")
    zero = InterventionConfig(
        price_ceiling_multiplier=10,
        transaction_tax_rate=0,
        supply_subsidy_fraction=0,
        price_ceiling_admin_cost=0,
        tax_admin_cost=0,
        subsidy_admin_cost=0,
    )
    do_nothing = resolve_intervention(
        ActionID.DO_NOTHING,
        reference_price=100,
        scenario=scenario,
        colluding_seller_ids=(),
        tape=tape,
        decision_period=20,
        config=zero,
    )
    tax = resolve_intervention(
        ActionID.TRANSACTION_TAX,
        reference_price=100,
        scenario=scenario,
        colluding_seller_ids=(),
        tape=tape,
        decision_period=20,
        config=zero,
    )
    subsidy = resolve_intervention(
        ActionID.SUPPLY_SUBSIDY,
        reference_price=100,
        scenario=scenario,
        colluding_seller_ids=(),
        tape=tape,
        decision_period=20,
        config=zero,
    )
    assert tax.policy_flows == do_nothing.policy_flows
    assert subsidy.policy_flows == do_nothing.policy_flows
