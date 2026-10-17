"""Representative alternate-clearing and normative-sensitivity audit."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from janus.core.clearing import ClearingPriceRule
from janus.core.config import MarketConfig
from janus.oracle.counterfactual import OracleObjective, evaluate_oracle
from janus.scenarios.generator import CandidateGeneratorConfig, CandidateWorld, generate_candidates
from janus.scenarios.matching import match_ambiguity_bundles
from janus.simulation.runner import DynamicsConfig


def _job(
    item: tuple[CandidateWorld, OracleObjective, ClearingPriceRule, int],
) -> tuple[str, str, str, str]:
    world, objective, price_rule, rollouts = item
    result = evaluate_oracle(
        world.decision_state,
        world.scenario,
        root_seed=20261017,
        rollout_count=rollouts,
        market_config=MarketConfig(initial_cash_multiple=100.0),
        dynamics=DynamicsConfig(recovery_periods=20),
        objective=objective,
        price_rule=price_rule,
    )
    return world.family.value, objective, price_rule, result.best_action.value


def run_robustness_audit(
    output_path: Path,
    *,
    rollouts: int = 64,
    workers: int = 4,
) -> bool:
    config = CandidateGeneratorConfig(candidates_per_family=80)
    candidates = generate_candidates("calibration", config)
    bundle = match_ambiguity_bundles(
        candidates,
        bundle_count=15,
        split="calibration",
    )[0]
    objectives: tuple[OracleObjective, ...] = (
        "primary",
        "consumer_protection",
        "producer_continuity",
        "stability_aware",
        "fiscal_neutral",
    )
    jobs: list[tuple[CandidateWorld, OracleObjective, ClearingPriceRule, int]] = []
    for world in bundle.worlds:
        jobs.append((world, "primary", "marginal_ask", rollouts))
        jobs.extend((world, objective, "midpoint", rollouts) for objective in objectives)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = tuple(executor.map(_job, jobs))
    nested: dict[str, dict[str, str]] = {}
    for family, objective, price_rule, action in rows:
        nested.setdefault(family, {})[f"{objective}:{price_rule}"] = action
    clearing_stable = all(
        values["primary:midpoint"] == values["primary:marginal_ask"] for values in nested.values()
    )
    normative_stable = all(
        len({action for key, action in values.items() if key.endswith(":midpoint")}) == 1
        for values in nested.values()
    )
    payload = {
        "schema_version": "janus-robustness-v1",
        "rollouts": rollouts,
        "representative_worlds": len(bundle.worlds),
        "clearing_stable": clearing_stable,
        "normative_stable": normative_stable,
        "completed": True,
        "results": nested,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return clearing_stable
