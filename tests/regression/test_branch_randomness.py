from dataclasses import replace

from janus.core.fixtures import analytic_market
from janus.core.randomness import DisturbanceTape
from janus.interventions.actions import ActionID
from janus.scenarios.families import LatentFamily, ScenarioConfig
from janus.simulation.runner import DynamicsConfig, run_recovery


def test_action_branches_share_exogenous_tape() -> None:
    state = replace(analytic_market(), period=30, last_price=100.0)
    scenario = ScenarioConfig(
        scenario_id="paired-branches",
        family=LatentFamily.MOMENTUM_DEMAND,
        shock_duration=20,
    )
    tape = DisturbanceTape(44, "paired")
    dynamics = DynamicsConfig(recovery_periods=4)
    do_nothing = run_recovery(state, scenario, ActionID.DO_NOTHING, tape, dynamics=dynamics)
    taxed = run_recovery(state, scenario, ActionID.TRANSACTION_TAX, tape, dynamics=dynamics)

    for left, right in zip(do_nothing.steps, taxed.steps, strict=True):
        assert left.exogenous.buyer_value_noise == right.exogenous.buyer_value_noise
        assert left.exogenous.seller_cost_noise == right.exogenous.seller_cost_noise
        left_priorities = dict(left.exogenous.tie_priorities)
        right_priorities = dict(right.exogenous.tie_priorities)
        common_orders = set(left_priorities).intersection(right_priorities)
        assert common_orders
        assert all(
            left_priorities[order_id] == right_priorities[order_id] for order_id in common_orders
        )


def test_initial_state_is_not_mutated_by_recovery() -> None:
    state = replace(analytic_market(), period=30, last_price=100.0)
    original = state.clone()
    scenario = ScenarioConfig(
        scenario_id="immutable",
        family=LatentFamily.DEMAND_INCREASE,
        shock_duration=20,
    )
    run_recovery(
        state,
        scenario,
        ActionID.SUPPLY_SUBSIDY,
        DisturbanceTape(1, "immutable"),
        dynamics=DynamicsConfig(recovery_periods=2),
    )
    assert state == original


def test_recovery_preserves_recent_price_memory_across_branches() -> None:
    state = replace(
        analytic_market(),
        period=30,
        last_price=103.0,
        recent_transaction_prices=(99.0, 100.0, 101.0, 103.0),
    )
    scenario = ScenarioConfig(
        scenario_id="price-memory",
        family=LatentFamily.MOMENTUM_DEMAND,
        shock_duration=20,
    )
    result = run_recovery(
        state,
        scenario,
        ActionID.DO_NOTHING,
        DisturbanceTape(1, "price-memory"),
        dynamics=DynamicsConfig(recovery_periods=1),
    )
    assert result.initial_state.recent_transaction_prices == (99.0, 100.0, 101.0, 103.0)
    assert len(result.final_state.recent_transaction_prices) == 4
