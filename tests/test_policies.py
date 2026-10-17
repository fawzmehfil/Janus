import json

from janus.interventions.actions import ActionID
from janus.policies.constant import ConstantPolicy
from janus.policies.heuristic import PublicHeuristicPolicy
from janus.policies.random import UniformRandomPolicy


def public_payload(**updates: object) -> bytes:
    row = {
        "aggregate_inventory": 40,
        "reference_inventory": 40,
        "transaction_price": 100,
        "reference_price": 100,
        "transaction_volume": 20,
        "unfilled_submitted_demand": 0,
        "seller_hhi": 0.125,
    }
    row.update(updates)
    return json.dumps({"rows": [row]}).encode()


def test_constant_and_random_policies_are_stable() -> None:
    assert ConstantPolicy(ActionID.PRICE_CEILING).act(b"ignored") is ActionID.PRICE_CEILING
    policy = UniformRandomPolicy(123)
    assert policy.act(b"ignored", trial_id="a") == policy.act(b"ignored", trial_id="a")


def test_heuristic_uses_only_public_payload() -> None:
    policy = PublicHeuristicPolicy()
    assert policy.act(public_payload()) is ActionID.DO_NOTHING
    assert policy.act(public_payload(aggregate_inventory=10)) is ActionID.SUPPLY_SUBSIDY
    assert (
        policy.act(public_payload(seller_hhi=0.4, transaction_price=115))
        is ActionID.COLLUSION_INVESTIGATION
    )
