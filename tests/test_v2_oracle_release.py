from __future__ import annotations

import numpy as np

from janus.interventions.actions import ActionID
from janus.v2.oracle import METRICS, primary_oracle
from janus.v2.release import _replicate_schedule


def test_v2_oracle_round_trip_preserves_primary_samples() -> None:
    outcomes = []
    for index, action in enumerate(ActionID):
        samples = {metric: [float(index), float(index + 1)] for metric in METRICS}
        outcomes.append(
            {
                "action": action.value,
                "samples": samples,
                "means": {metric: float(index + 0.5) for metric in METRICS},
                "standard_errors": {metric: 0.5 for metric in METRICS},
            }
        )
    payload = {
        "action_outcomes": outcomes,
        "best_action": ActionID.COLLUSION_INVESTIGATION.value,
        "epsilon_optimal_actions": [ActionID.COLLUSION_INVESTIGATION.value],
        "epsilon_by_action": [[action.value, 4.2] for action in ActionID],
        "rollout_count": 2,
    }
    oracle = primary_oracle(payload)
    assert oracle.rollout_count == 2
    assert oracle.value_for(ActionID.DO_NOTHING).samples == (0.0, 1.0)


def test_split_replicate_schedule_has_300_trials() -> None:
    schedule = _replicate_schedule(40, np.random.default_rng(44))
    assert len(schedule) == 160
    assert sum(len(replicates) for replicates in schedule.values()) == 300
    assert all(len(replicates) in {1, 2} for replicates in schedule.values())
