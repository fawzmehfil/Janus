from dataclasses import replace

from janus.core.fixtures import analytic_market
from janus.scenarios.families import LatentFamily, ScenarioConfig, primitives_for_period


def shocked_state():
    return replace(analytic_market(), period=20)


def scenario(family: LatentFamily) -> ScenarioConfig:
    return ScenarioConfig(scenario_id=f"test-{family}", family=family)


def test_supply_shortage_changes_cost_or_capacity_not_demand() -> None:
    state = shocked_state()
    baseline_config = scenario(LatentFamily.DEMAND_INCREASE).model_copy(update={"shock_start": 99})
    baseline = primitives_for_period(state, baseline_config)
    shocked = primitives_for_period(state, scenario(LatentFamily.SUPPLY_SHORTAGE))
    assert shocked.buyer_true_values == baseline.buyer_true_values
    assert shocked.buyer_bid_adjustments == baseline.buyer_bid_adjustments
    assert shocked.seller_marginal_costs != baseline.seller_marginal_costs
    assert shocked.seller_offer_limits != baseline.seller_offer_limits
    assert shocked.seller_markups == baseline.seller_markups


def test_demand_increase_changes_true_values_not_seller_side() -> None:
    state = shocked_state()
    baseline_config = scenario(LatentFamily.DEMAND_INCREASE).model_copy(update={"shock_start": 99})
    baseline = primitives_for_period(state, baseline_config)
    shocked = primitives_for_period(state, scenario(LatentFamily.DEMAND_INCREASE))
    assert shocked.buyer_true_values != baseline.buyer_true_values
    assert shocked.buyer_bid_adjustments == baseline.buyer_bid_adjustments
    assert shocked.seller_marginal_costs == baseline.seller_marginal_costs
    assert shocked.seller_offer_limits == baseline.seller_offer_limits


def test_collusion_changes_strategy_not_cost_or_value() -> None:
    state = shocked_state()
    config = scenario(LatentFamily.SELLER_COLLUSION)
    active = primitives_for_period(state, config)
    disabled = primitives_for_period(state, config, collusion_disabled=True)
    assert active.buyer_true_values == disabled.buyer_true_values
    assert active.seller_marginal_costs == disabled.seller_marginal_costs
    assert active.seller_markups != disabled.seller_markups
    assert active.seller_offer_limits != disabled.seller_offer_limits
    assert active.colluding_seller_ids == disabled.colluding_seller_ids


def test_momentum_changes_bids_without_creating_fundamental_value() -> None:
    state = shocked_state()
    config = scenario(LatentFamily.MOMENTUM_DEMAND)
    flat = primitives_for_period(state, config, recent_log_returns=(0.0, 0.0))
    rising = primitives_for_period(state, config, recent_log_returns=(0.03, 0.04))
    assert rising.buyer_true_values == flat.buyer_true_values
    assert rising.seller_marginal_costs == flat.seller_marginal_costs
    assert rising.buyer_bid_adjustments != flat.buyer_bid_adjustments
