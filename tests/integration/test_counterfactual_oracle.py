from dataclasses import replace

from janus.core.fixtures import analytic_market
from janus.interventions.actions import ActionID
from janus.oracle.counterfactual import check_oracle_convergence, evaluate_oracle
from janus.scenarios.families import LatentFamily, ScenarioConfig
from janus.simulation.runner import DynamicsConfig


def decision_state():
    return replace(analytic_market(), period=30, last_price=100.0)


def test_oracle_uses_paired_samples_and_dominates_actions_by_mean() -> None:
    scenario = ScenarioConfig(
        scenario_id="oracle-supply",
        family=LatentFamily.SUPPLY_SHORTAGE,
        shock_duration=20,
    )
    result = evaluate_oracle(
        decision_state(),
        scenario,
        root_seed=12,
        rollout_count=8,
        dynamics=DynamicsConfig(recovery_periods=3),
    )
    assert all(len(value.samples) == 8 for value in result.action_values)
    best = result.value_for(result.best_action).mean
    assert all(best >= value.mean for value in result.action_values)
    assert result.best_action in result.epsilon_optimal_actions


def test_oracle_is_bit_reproducible() -> None:
    scenario = ScenarioConfig(
        scenario_id="oracle-collusion",
        family=LatentFamily.SELLER_COLLUSION,
        shock_duration=20,
    )
    kwargs = {
        "root_seed": 9,
        "rollout_count": 4,
        "dynamics": DynamicsConfig(recovery_periods=2),
    }
    assert evaluate_oracle(decision_state(), scenario, **kwargs) == evaluate_oracle(
        decision_state(), scenario, **kwargs
    )


def test_small_convergence_audit_reports_all_scales() -> None:
    scenario = ScenarioConfig(
        scenario_id="convergence-demand",
        family=LatentFamily.DEMAND_INCREASE,
        shock_duration=20,
    )
    result = check_oracle_convergence(
        decision_state(),
        scenario,
        root_seed=3,
        base_rollouts=2,
        dynamics=DynamicsConfig(recovery_periods=2),
    )
    assert result.rollout_counts == (2, 4, 8)
    assert len(result.best_actions) == 3
    assert result.maximum_mean_delta >= 0
    assert all(action in ActionID for action in result.best_actions)
