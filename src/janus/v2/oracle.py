"""V2 counterfactual oracle preserving decomposed economic outcome samples."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from janus.core.config import MarketConfig
from janus.core.randomness import DisturbanceTape
from janus.core.replay import content_hash
from janus.evaluation.observations import realized_log_volatility, seller_hhi
from janus.interventions.actions import ActionID, InterventionConfig
from janus.oracle.counterfactual import ActionValue, OracleResult
from janus.scenarios.generator import CandidateWorld
from janus.simulation.runner import DynamicsConfig, RecoveryResult, run_recovery

METRICS = (
    "primary",
    "net_total_surplus",
    "consumer_protection",
    "producer_continuity",
    "stability_aware",
    "fiscal_neutral",
    "consumer_surplus",
    "producer_surplus",
    "government_balance",
    "real_policy_cost",
    "financing_burden",
    "exit_count",
    "volatility",
    "price_level",
    "efficient_demand_left_unfilled",
    "transaction_volume",
    "inventory",
    "stockout_periods",
    "concentration",
    "recovery_time",
    "unnecessary_intervention_harm",
)


def _recovery_time(
    result: RecoveryResult, *, reference_price: float, reference_volume: float
) -> float:
    recovered = []
    for step in result.steps:
        clearing = step.period_result.clearing
        price_ok = (
            clearing.transaction_price is not None
            and abs(clearing.transaction_price - reference_price) / reference_price <= 0.05
        )
        volume_ok = clearing.quantity >= 0.90 * reference_volume
        recovered.append(price_ok and volume_ok)
    for start in range(max(0, len(recovered) - 2)):
        if all(recovered[start : start + 3]):
            return float(start + 1)
    return float(len(recovered) + 1)


def _outcome(
    result: RecoveryResult, *, reference_price: float, reference_volume: float
) -> dict[str, float]:
    ledgers = [step.period_result.ledger for step in result.steps]
    prices = [step.period_result.clearing.transaction_price for step in result.steps]
    observed_prices = [price for price in prices if price is not None]
    return {
        "primary": result.utilities.primary,
        "net_total_surplus": result.utilities.primary,
        "consumer_protection": result.utilities.consumer_protection,
        "producer_continuity": result.utilities.producer_continuity,
        "stability_aware": result.utilities.stability_aware,
        "fiscal_neutral": result.utilities.fiscal_neutral,
        "consumer_surplus": sum(row.consumer_surplus for row in ledgers),
        "producer_surplus": sum(row.producer_surplus for row in ledgers),
        "government_balance": sum(row.government_balance for row in ledgers),
        "real_policy_cost": sum(row.real_cost for row in ledgers),
        "financing_burden": sum(row.financing_cost for row in ledgers),
        "exit_count": float(sum(not seller.active for seller in result.final_state.sellers)),
        "volatility": realized_log_volatility(prices, window=len(prices)),
        "price_level": fmean(observed_prices) if observed_prices else 0.0,
        "efficient_demand_left_unfilled": float(
            sum(step.efficient_demand_left_unfilled for step in result.steps)
        ),
        "transaction_volume": float(
            sum(step.period_result.clearing.quantity for step in result.steps)
        ),
        "inventory": float(
            fmean(
                sum(seller.inventory for seller in step.period_result.next_state.sellers)
                for step in result.steps
            )
        ),
        "stockout_periods": float(
            sum(
                step.period_result.clearing.unfilled_demand > 0
                and step.period_result.clearing.quantity
                >= step.period_result.clearing.available_supply
                for step in result.steps
            )
        ),
        "concentration": float(
            fmean(seller_hhi(step.period_result.next_state) for step in result.steps)
        ),
        "recovery_time": _recovery_time(
            result,
            reference_price=reference_price,
            reference_volume=reference_volume,
        ),
    }


def _standard_error(values: list[float]) -> float:
    return 0.0 if len(values) < 2 else stdev(values) / sqrt(len(values))


def evaluate_v2_oracle(
    world: CandidateWorld,
    *,
    root_seed: int,
    rollout_count: int,
    market: MarketConfig,
    intervention: InterventionConfig,
    dynamics: DynamicsConfig,
) -> dict[str, Any]:
    samples: dict[ActionID, dict[str, list[float]]] = {
        action: {metric: [] for metric in METRICS} for action in ActionID
    }
    final_public_row = world.observation.rows[-1]
    for rollout in range(rollout_count):
        tape = DisturbanceTape(root_seed, f"{world.scenario.scenario_id}:future:{rollout}")
        for action in ActionID:
            outcome = _outcome(
                run_recovery(
                    world.decision_state,
                    world.scenario,
                    action,
                    tape,
                    market_config=market,
                    intervention_config=intervention,
                    dynamics=dynamics,
                ),
                reference_price=final_public_row.reference_price,
                reference_volume=final_public_row.reference_volume,
            )
            for metric, value in outcome.items():
                samples[action][metric].append(value)
    do_nothing = samples[ActionID.DO_NOTHING]["primary"]
    for action in ActionID:
        samples[action]["unnecessary_intervention_harm"] = [
            float(reference) - float(selected)
            for reference, selected in zip(
                do_nothing,
                samples[action]["primary"],
                strict=True,
            )
        ]
    action_outcomes: list[dict[str, Any]] = []
    for action in ActionID:
        action_outcomes.append(
            {
                "action": action.value,
                "samples": samples[action],
                "means": {metric: fmean(samples[action][metric]) for metric in METRICS},
                "standard_errors": {
                    metric: _standard_error(samples[action][metric]) for metric in METRICS
                },
            }
        )
    best = max(
        action_outcomes,
        key=lambda row: (float(row["means"]["primary"]), str(row["action"])),
    )
    epsilon_by_action = []
    epsilon_optimal = []
    for row in action_outcomes:
        differences = [
            float(left) - float(right)
            for left, right in zip(
                best["samples"]["primary"], row["samples"]["primary"], strict=True
            )
        ]
        epsilon = max(4.2, 2 * _standard_error(differences))
        epsilon_by_action.append([row["action"], epsilon])
        if float(best["means"]["primary"]) - float(row["means"]["primary"]) <= epsilon:
            epsilon_optimal.append(row["action"])
    return {
        "schema_version": "janus-v2-oracle-v2",
        "world_id": world.world_id,
        "objective": "primary",
        "rollout_count": rollout_count,
        "action_outcomes": action_outcomes,
        "best_action": best["action"],
        "epsilon_optimal_actions": sorted(epsilon_optimal),
        "epsilon_by_action": epsilon_by_action,
    }


def primary_oracle(payload: dict[str, Any]) -> OracleResult:
    values = tuple(
        ActionValue(
            action=ActionID(row["action"]),
            samples=tuple(float(value) for value in row["samples"]["primary"]),
            mean=float(row["means"]["primary"]),
            standard_error=float(row["standard_errors"]["primary"]),
        )
        for row in payload["action_outcomes"]
    )
    return OracleResult(
        action_values=values,
        best_action=ActionID(payload["best_action"]),
        epsilon_optimal_actions=frozenset(
            ActionID(value) for value in payload["epsilon_optimal_actions"]
        ),
        epsilon_by_action=tuple(
            (ActionID(action), float(epsilon)) for action, epsilon in payload["epsilon_by_action"]
        ),
        rollout_count=int(payload["rollout_count"]),
    )


def _job(
    job: tuple[
        CandidateWorld,
        int,
        int,
        MarketConfig,
        InterventionConfig,
        DynamicsConfig,
    ],
) -> tuple[str, dict[str, Any]]:
    world, seed, rollouts, market, intervention, dynamics = job
    return world.world_id, evaluate_v2_oracle(
        world,
        root_seed=seed,
        rollout_count=rollouts,
        market=market,
        intervention=intervention,
        dynamics=dynamics,
    )


def load_or_evaluate_v2_oracles(
    worlds: tuple[CandidateWorld, ...],
    *,
    output_directory: Path,
    root_seed: int,
    rollout_count: int,
    workers: int,
    market: MarketConfig,
    intervention: InterventionConfig,
    dynamics: DynamicsConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    oracle_directory = output_directory / "oracles"
    oracle_directory.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    missing = []
    keys = {}
    for world in worlds:
        key = content_hash(
            {
                "format": "janus-v2-oracle-v2",
                "state": asdict(world.decision_state),
                "scenario": world.scenario.model_dump(mode="json"),
                "root_seed": root_seed,
                "rollout_count": rollout_count,
                "market": market.model_dump(mode="json"),
                "intervention": intervention.model_dump(mode="json"),
                "dynamics": dynamics.model_dump(mode="json"),
            }
        )
        keys[world.world_id] = key
        path = oracle_directory / f"{world.world_id}-v2-k{rollout_count}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("cache_key") == key:
                results[world.world_id] = payload["oracle"]
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                continue
        missing.append(world)
    jobs = tuple(
        (world, root_seed, rollout_count, market, intervention, dynamics) for world in missing
    )
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for world_id, oracle in executor.map(_job, jobs):
                results[world_id] = oracle
                path = oracle_directory / f"{world_id}-v2-k{rollout_count}.json"
                path.write_text(
                    json.dumps(
                        {"cache_key": keys[world_id], "oracle": oracle},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return results, hashes
