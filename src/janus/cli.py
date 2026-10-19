"""Command-line entry point for deterministic local workflows."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from janus.core.clearing import clear_market
from janus.core.fixtures import analytic_market, competitive_orders
from janus.core.randomness import DisturbanceTape
from janus.evaluation.hidden_release import generate_hidden_release, prepare_hidden_seed
from janus.evaluation.openai_smoke import (
    DIAGNOSTIC_CONFIRMATION,
    SMOKE_MAXIMUM_CEILING_USD,
    SPEND_CONFIRMATION,
    diagnostic_preflight,
    run_openai_diagnostic,
    run_openai_smoke,
    smoke_preflight,
    validate_smoke_ceiling,
)
from janus.evaluation.phase0d_runner import run_phase0d
from janus.evaluation.phase0e_runner import run_phase0e_dry_run
from janus.evaluation.robustness import run_robustness_audit
from janus.models.providers.openai import make_openai_sdk_transport


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
    phase0e = subparsers.add_parser("phase0e-dry-run", help="run the zero-cost mock harness")
    phase0e.add_argument("--output", type=Path, default=Path("artifacts/phase0e-dry-run"))
    smoke_preflight_parser = subparsers.add_parser(
        "openai-smoke-preflight",
        help="print the no-network five-world Luna cost and input plan",
    )
    smoke_preflight_parser.add_argument(
        "--max-usd",
        type=float,
        default=SMOKE_MAXIMUM_CEILING_USD,
    )
    diagnostic_preflight_parser = subparsers.add_parser(
        "openai-diagnostic-preflight",
        help="print the no-network one-world 2,048-token diagnostic plan",
    )
    diagnostic_preflight_parser.add_argument(
        "--prior-output",
        type=Path,
        default=Path("artifacts/openai-smoke-v1"),
    )
    diagnostic_preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/openai-diagnostic-2048-v1"),
    )
    diagnostic_preflight_parser.add_argument(
        "--max-usd",
        type=float,
        default=SMOKE_MAXIMUM_CEILING_USD,
    )
    smoke = subparsers.add_parser(
        "openai-smoke",
        help="run the explicitly confirmed five-world paid Luna smoke test",
    )
    smoke.add_argument("--output", type=Path, default=Path("artifacts/openai-smoke-v1"))
    smoke.add_argument("--max-usd", type=float, default=SMOKE_MAXIMUM_CEILING_USD)
    smoke.add_argument("--timeout-seconds", type=float, default=120.0)
    smoke.add_argument(
        "--confirm-spend",
        required=True,
        help=f"must equal {SPEND_CONFIRMATION!r}; no request is sent otherwise",
    )
    diagnostic = subparsers.add_parser(
        "openai-diagnostic",
        help="run one explicitly confirmed 2,048-token Luna development diagnostic",
    )
    diagnostic.add_argument(
        "--prior-output",
        type=Path,
        default=Path("artifacts/openai-smoke-v1"),
    )
    diagnostic.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/openai-diagnostic-2048-v1"),
    )
    diagnostic.add_argument("--max-usd", type=float, default=SMOKE_MAXIMUM_CEILING_USD)
    diagnostic.add_argument("--timeout-seconds", type=float, default=120.0)
    diagnostic.add_argument(
        "--confirm-spend",
        required=True,
        help=f"must equal {DIAGNOSTIC_CONFIRMATION!r}; no request is sent otherwise",
    )
    hidden_seed = subparsers.add_parser(
        "hidden-seed", help="create a private seed and public commitment before generation"
    )
    hidden_seed.add_argument("--code-commit", required=True)
    hidden_seed.add_argument("--private-output", type=Path, default=Path("artifacts/hidden-v1"))
    hidden_seed.add_argument(
        "--commitment",
        type=Path,
        default=Path("manifests/hidden-v1-commitment.json"),
    )
    hidden_generate = subparsers.add_parser(
        "hidden-generate", help="generate the committed hidden release locally"
    )
    hidden_generate.add_argument("--code-commit", required=True)
    hidden_generate.add_argument("--private-output", type=Path, default=Path("artifacts/hidden-v1"))
    hidden_generate.add_argument(
        "--public-manifest",
        type=Path,
        default=Path("manifests/hidden-v1-public.json"),
    )
    hidden_generate.add_argument(
        "--commitment",
        type=Path,
        default=Path("manifests/hidden-v1-commitment.json"),
    )
    hidden_generate.add_argument("--workers", type=int, default=4)
    robustness = subparsers.add_parser(
        "robustness", help="run alternate-clearing and normative sensitivity audits"
    )
    robustness.add_argument("--output", type=Path, default=Path("artifacts/robustness/report.json"))
    robustness.add_argument("--rollouts", type=int, default=64)
    robustness.add_argument("--workers", type=int, default=4)
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
    if args.command == "phase0e-dry-run":
        passed = run_phase0e_dry_run(args.output, Path.cwd())
        print(json.dumps({"output": str(args.output), "passed": passed}, sort_keys=True))
        return 0 if passed else 2
    if args.command == "openai-smoke-preflight":
        plan = smoke_preflight(Path.cwd(), ceiling_usd=args.max_usd)
        print(json.dumps(plan, sort_keys=True, indent=2))
        return 0 if plan["fits_ceiling"] else 2
    if args.command == "openai-diagnostic-preflight":
        plan = diagnostic_preflight(
            Path.cwd(),
            args.prior_output,
            args.output,
            ceiling_usd=args.max_usd,
        )
        print(json.dumps(plan, sort_keys=True, indent=2))
        return 0 if plan["fits_ceiling"] else 2
    if args.command == "openai-smoke":
        validate_smoke_ceiling(args.max_usd)
        if args.confirm_spend != SPEND_CONFIRMATION:
            raise SystemExit(
                f"refusing paid run: --confirm-spend must equal {SPEND_CONFIRMATION!r}"
            )
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise SystemExit("refusing paid run: OPENAI_API_KEY is not set")
        transport, sdk_version = make_openai_sdk_transport(
            api_key,
            timeout_seconds=args.timeout_seconds,
        )
        passed, manifest = run_openai_smoke(
            Path.cwd(),
            args.output,
            transport=transport,
            sdk_version=sdk_version,
            ceiling_usd=args.max_usd,
        )
        print(json.dumps({"manifest": str(manifest), "passed": passed}, sort_keys=True))
        return 0 if passed else 2
    if args.command == "openai-diagnostic":
        validate_smoke_ceiling(args.max_usd)
        if args.confirm_spend != DIAGNOSTIC_CONFIRMATION:
            raise SystemExit(
                f"refusing paid run: --confirm-spend must equal {DIAGNOSTIC_CONFIRMATION!r}"
            )
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise SystemExit("refusing paid run: OPENAI_API_KEY is not set")
        transport, sdk_version = make_openai_sdk_transport(
            api_key,
            timeout_seconds=args.timeout_seconds,
        )
        passed, manifest = run_openai_diagnostic(
            Path.cwd(),
            args.prior_output,
            args.output,
            transport=transport,
            sdk_version=sdk_version,
            ceiling_usd=args.max_usd,
        )
        print(json.dumps({"manifest": str(manifest), "passed": passed}, sort_keys=True))
        return 0 if passed else 2
    if args.command == "hidden-seed":
        commitment = prepare_hidden_seed(
            args.private_output,
            args.commitment,
            code_commit=args.code_commit,
        )
        print(json.dumps({"commitment_sha256": commitment}, sort_keys=True))
        return 0
    if args.command == "hidden-generate":
        passed = generate_hidden_release(
            args.private_output,
            args.public_manifest,
            args.commitment,
            Path("configs/experiments/hidden-v1.json"),
            Path("configs/prompts/primary-v1.txt"),
            code_commit=args.code_commit,
            workers=args.workers,
        )
        print(json.dumps({"public_manifest": str(args.public_manifest), "passed": passed}))
        return 0 if passed else 2
    if args.command == "robustness":
        passed = run_robustness_audit(
            args.output,
            rollouts=args.rollouts,
            workers=args.workers,
        )
        print(json.dumps({"output": str(args.output), "passed": passed}, sort_keys=True))
        return 0 if passed else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
