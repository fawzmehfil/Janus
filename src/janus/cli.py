"""Command-line entry point for deterministic local workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from janus.core.clearing import clear_market
from janus.core.fixtures import analytic_market, competitive_orders
from janus.core.randomness import DisturbanceTape
from janus.evaluation.phase0d_runner import run_phase0d


def analytic_fixture_command() -> int:
    state = analytic_market()
    bids, asks = competitive_orders(state)
    tape = DisturbanceTape(20261016, "analytic-fixture")
    order_ids = tuple(bid.order_id for bid in bids) + tuple(ask.order_id for ask in asks)
    priorities = tape.tie_priority(order_ids, state.period)
    result = clear_market(bids, asks, tie_priority=priorities)
    surplus = sum(trade.buyer_value - trade.seller_cost for trade in result.trades)
    payload = {
        "quantity": result.quantity,
        "price": result.transaction_price,
        "surplus": surplus,
        "expected": {"quantity": 21, "price": 100.0, "surplus": 840.0},
        "passed": result.quantity == 21 and result.transaction_price == 100.0 and surplus == 840.0,
        "clearing": asdict(result),
    }
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0 if payload["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="janus")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("analytic-fixture", help="verify the hand-solvable reference market")
    phase0d = subparsers.add_parser("phase0d", help="run the local benchmark-validity gate")
    phase0d.add_argument("--output", type=Path, default=Path("artifacts/phase0d"))
    phase0d.add_argument("--rollouts", type=int, default=256)
    phase0d.add_argument("--workers", type=int, default=4)
    phase0d.add_argument(
        "--skip-convergence",
        action="store_true",
        help="diagnostic only: skip the required 2x/4x oracle audit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analytic-fixture":
        return analytic_fixture_command()
    if args.command == "phase0d":
        passed = run_phase0d(
            args.output,
            rollout_count=args.rollouts,
            workers=args.workers,
            run_convergence=not args.skip_convergence,
        )
        print(json.dumps({"output": str(args.output), "passed": passed}, sort_keys=True))
        return 0 if passed else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
