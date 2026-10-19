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
from janus.evaluation.hidden_analysis import run_hidden_analysis
from janus.evaluation.hidden_release import generate_hidden_release, prepare_hidden_seed
from janus.evaluation.hidden_sensitivity import (
    generate_normative_oracles,
    run_hidden_sensitivity,
)
from janus.evaluation.openai_hidden import (
    HIDDEN_MAXIMUM_CUMULATIVE_CEILING_USD,
    RECOMMENDED_CUMULATIVE_CEILING_USD,
    hidden_confirmation,
    hidden_preflight,
    run_openai_hidden,
    validate_hidden_ceiling,
)
from janus.evaluation.openai_smoke import (
    CORRECTED_SMOKE_CONFIRMATION,
    CORRECTED_SMOKE_MAXIMUM_CEILING_USD,
    DIAGNOSTIC_CONFIRMATION,
    SMOKE_MAXIMUM_CEILING_USD,
    SPEND_CONFIRMATION,
    corrected_smoke_preflight,
    diagnostic_preflight,
    run_corrected_openai_smoke,
    run_openai_diagnostic,
    run_openai_smoke,
    smoke_preflight,
    validate_smoke_ceiling,
)
from janus.evaluation.phase0d_runner import run_phase0d
from janus.evaluation.phase0e_runner import run_phase0e_dry_run
from janus.evaluation.publication import generate_publication_outputs
from janus.evaluation.robustness import run_robustness_audit
from janus.models.providers.openai import make_openai_sdk_transport
from janus.v2.admission import build_provider_admission
from janus.v2.archive import build_reproduction_archive
from janus.v2.audit import audit_reproduction_archive, development_gate_report
from janus.v2.development import run_v2_local_development
from janus.v2.dry_run import run_v2_dry_run
from janus.v2.ood_pilot import run_structural_ood_pilot
from janus.v2.precision import run_precision_analysis
from janus.v2.publication import generate_v2_publication_outputs
from janus.v2.release import generate_v2_hidden_release, prepare_v2_seed
from janus.v2.unblind import run_v2_unblinding


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
    corrected_preflight = subparsers.add_parser(
        "openai-corrected-smoke-preflight",
        help="print the no-network five-world 2,048-token corrected smoke plan",
    )
    corrected_preflight.add_argument(
        "--prior-output", type=Path, default=Path("artifacts/openai-smoke-v1")
    )
    corrected_preflight.add_argument(
        "--diagnostic-output",
        type=Path,
        default=Path("artifacts/openai-diagnostic-2048-v1"),
    )
    corrected_preflight.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/openai-smoke-2048-v2"),
    )
    corrected_preflight.add_argument(
        "--max-usd", type=float, default=CORRECTED_SMOKE_MAXIMUM_CEILING_USD
    )
    corrected = subparsers.add_parser(
        "openai-corrected-smoke",
        help="run the explicitly confirmed five-world 2,048-token corrected smoke",
    )
    corrected.add_argument("--prior-output", type=Path, default=Path("artifacts/openai-smoke-v1"))
    corrected.add_argument(
        "--diagnostic-output",
        type=Path,
        default=Path("artifacts/openai-diagnostic-2048-v1"),
    )
    corrected.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/openai-smoke-2048-v2"),
    )
    corrected.add_argument("--max-usd", type=float, default=CORRECTED_SMOKE_MAXIMUM_CEILING_USD)
    corrected.add_argument("--timeout-seconds", type=float, default=120.0)
    corrected.add_argument(
        "--confirm-spend",
        required=True,
        help=f"must equal {CORRECTED_SMOKE_CONFIRMATION!r}; no request is sent otherwise",
    )
    hidden_preflight_parser = subparsers.add_parser(
        "openai-hidden-preflight",
        help="validate and price the no-network 300-request hidden collection",
    )
    hidden_preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/openai-hidden-v1"),
    )
    hidden_preflight_parser.add_argument(
        "--max-usd",
        type=float,
        default=RECOMMENDED_CUMULATIVE_CEILING_USD,
        help="cumulative project ceiling, including the recorded live smoke spend",
    )
    hidden_run = subparsers.add_parser(
        "openai-hidden",
        help="collect or resume the explicitly confirmed 300 hidden Luna responses",
    )
    hidden_run.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/openai-hidden-v1"),
    )
    hidden_run.add_argument(
        "--max-usd",
        type=float,
        default=RECOMMENDED_CUMULATIVE_CEILING_USD,
        help=(
            "cumulative project ceiling; must be at most "
            f"${HIDDEN_MAXIMUM_CUMULATIVE_CEILING_USD:.2f}"
        ),
    )
    hidden_run.add_argument("--timeout-seconds", type=float, default=120.0)
    hidden_run.add_argument(
        "--confirm-spend",
        required=True,
        help="must exactly match the ceiling-specific phrase printed by preflight",
    )
    hidden_analysis = subparsers.add_parser(
        "hidden-analyze",
        help="verify, unblind, and score a response set committed before analysis",
    )
    hidden_analysis.add_argument(
        "--response-manifest",
        type=Path,
        default=Path("artifacts/openai-hidden-v1/manifests/run-20261019T083004.963054Z.json"),
    )
    hidden_analysis.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/hidden-v1-analysis/report.json"),
    )
    hidden_analysis.add_argument("--response-commit", required=True)
    hidden_sensitivity = subparsers.add_parser(
        "hidden-sensitivity",
        help="run local-only invalid-response and normative sensitivity analyses",
    )
    hidden_sensitivity.add_argument(
        "--response-manifest",
        type=Path,
        default=Path("artifacts/openai-hidden-v1/manifests/run-20261019T083004.963054Z.json"),
    )
    hidden_sensitivity.add_argument(
        "--normative-output",
        type=Path,
        default=Path("artifacts/hidden-v1-normative"),
    )
    hidden_sensitivity.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/hidden-v1-analysis/sensitivity.json"),
    )
    hidden_sensitivity.add_argument("--workers", type=int, default=4)
    hidden_publication = subparsers.add_parser(
        "hidden-publication",
        help="generate final hidden-v1 tables and vector figures",
    )
    hidden_publication.add_argument(
        "--primary-report",
        type=Path,
        default=Path("artifacts/hidden-v1-analysis/report.json"),
    )
    hidden_publication.add_argument(
        "--sensitivity-report",
        type=Path,
        default=Path("artifacts/hidden-v1-analysis/sensitivity.json"),
    )
    hidden_publication.add_argument(
        "--output",
        type=Path,
        default=Path("reports/hidden-v1-publication"),
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
    v2_development = subparsers.add_parser(
        "v2-local-development",
        help="run the local-only v2 baseline, margin, contrast, and precision gates",
    )
    v2_development.add_argument("--output", type=Path, default=Path("artifacts/v2-development"))
    v2_development.add_argument("--objective-rollouts", type=int, default=64)
    v2_development.add_argument("--workers", type=int, default=4)
    v2_precision = subparsers.add_parser(
        "v2-precision", help="recompute the local-only v1-informed allocation analysis"
    )
    v2_precision.add_argument("--output", type=Path, default=Path("reports/v2/precision.json"))
    v2_dry_run = subparsers.add_parser(
        "v2-dry-run", help="exercise all four model roles with injected offline transports"
    )
    v2_dry_run.add_argument("--output", type=Path, default=Path("artifacts/v2-dry-run"))
    subparsers.add_parser(
        "v2-gates", help="report all prerequisites that still block hidden-v2 generation"
    )
    v2_ood = subparsers.add_parser(
        "v2-ood-pilot", help="audit structural cue/action reversals using revealed v1 pilot data"
    )
    v2_ood.add_argument("--output", type=Path, default=Path("reports/v2/structural-ood.json"))
    v2_audit = subparsers.add_parser(
        "v2-audit", help="audit a completed reproduction archive without self-certifying it"
    )
    v2_audit.add_argument("archive", type=Path)
    v2_seed = subparsers.add_parser(
        "v2-hidden-seed", help="create the gated fresh hidden-v2 seed commitment"
    )
    v2_seed.add_argument("--code-commit", required=True)
    v2_seed.add_argument("--private-output", type=Path, default=Path("artifacts/hidden-v2"))
    v2_seed.add_argument(
        "--commitment", type=Path, default=Path("manifests/hidden-v2-commitment.json")
    )
    v2_generate = subparsers.add_parser(
        "v2-hidden-generate", help="generate hidden-v2 after every development gate passes"
    )
    v2_generate.add_argument("--code-commit", required=True)
    v2_generate.add_argument("--private-output", type=Path, default=Path("artifacts/hidden-v2"))
    v2_generate.add_argument(
        "--public-manifest", type=Path, default=Path("manifests/hidden-v2-public.json")
    )
    v2_generate.add_argument(
        "--commitment", type=Path, default=Path("manifests/hidden-v2-commitment.json")
    )
    v2_generate.add_argument("--workers", type=int, default=4)
    v2_admission = subparsers.add_parser(
        "v2-admission", help="evaluate four exact-model development response manifests"
    )
    v2_admission.add_argument("manifests", nargs=4, type=Path)
    v2_admission.add_argument(
        "--output", type=Path, default=Path("manifests/v2/provider-admission.json")
    )
    v2_archive = subparsers.add_parser(
        "v2-archive", help="build the complete post-unblinding reproduction directory"
    )
    v2_archive.add_argument("--response-commit", required=True)
    v2_archive.add_argument("--response-manifest", action="append", type=Path, required=True)
    v2_archive.add_argument("--output", type=Path, required=True)
    v2_unblind = subparsers.add_parser(
        "v2-unblind", help="verify four committed response manifests and score hidden-v2"
    )
    v2_unblind.add_argument("--response-commit", required=True)
    v2_unblind.add_argument("--response-manifest", action="append", type=Path, required=True)
    v2_unblind.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/hidden-v2-analysis/comparative-report.json"),
    )
    v2_publication = subparsers.add_parser(
        "v2-publication", help="regenerate deterministic v2 tables and vector figures"
    )
    v2_publication.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/hidden-v2-analysis/comparative-report.json"),
    )
    v2_publication.add_argument("--output", type=Path, default=Path("reports/v2-publication"))
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
    if args.command == "openai-corrected-smoke-preflight":
        plan = corrected_smoke_preflight(
            Path.cwd(),
            args.prior_output,
            args.diagnostic_output,
            args.output,
            ceiling_usd=args.max_usd,
        )
        print(json.dumps(plan, sort_keys=True, indent=2))
        return 0 if plan["fits_ceiling"] else 2
    if args.command == "openai-hidden-preflight":
        plan = hidden_preflight(
            Path.cwd(),
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
    if args.command == "openai-corrected-smoke":
        if not 0 < args.max_usd <= CORRECTED_SMOKE_MAXIMUM_CEILING_USD:
            raise SystemExit(
                "refusing paid run: corrected smoke --max-usd must be greater than zero "
                f"and at most {CORRECTED_SMOKE_MAXIMUM_CEILING_USD:.2f}"
            )
        if args.confirm_spend != CORRECTED_SMOKE_CONFIRMATION:
            raise SystemExit(
                f"refusing paid run: --confirm-spend must equal {CORRECTED_SMOKE_CONFIRMATION!r}"
            )
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise SystemExit("refusing paid run: OPENAI_API_KEY is not set")
        transport, sdk_version = make_openai_sdk_transport(
            api_key,
            timeout_seconds=args.timeout_seconds,
        )
        passed, manifest = run_corrected_openai_smoke(
            Path.cwd(),
            args.prior_output,
            args.diagnostic_output,
            args.output,
            transport=transport,
            sdk_version=sdk_version,
            ceiling_usd=args.max_usd,
        )
        print(json.dumps({"manifest": str(manifest), "passed": passed}, sort_keys=True))
        return 0 if passed else 2
    if args.command == "openai-hidden":
        validate_hidden_ceiling(args.max_usd)
        required_confirmation = hidden_confirmation(args.max_usd)
        if args.confirm_spend != required_confirmation:
            raise SystemExit(
                "refusing paid run: --confirm-spend must equal "
                f"{required_confirmation!r} for this cumulative ceiling"
            )
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise SystemExit("refusing paid run: OPENAI_API_KEY is not set")
        transport, sdk_version = make_openai_sdk_transport(
            api_key,
            timeout_seconds=args.timeout_seconds,
        )
        passed, manifest = run_openai_hidden(
            Path.cwd(),
            args.output,
            transport=transport,
            sdk_version=sdk_version,
            ceiling_usd=args.max_usd,
        )
        print(json.dumps({"manifest": str(manifest), "passed": passed}, sort_keys=True))
        return 0 if passed else 2
    if args.command == "hidden-analyze":
        report = run_hidden_analysis(
            Path.cwd(),
            args.response_manifest,
            args.output,
            response_commit=args.response_commit,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "hypotheses": report["hypotheses"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "hidden-sensitivity":
        normative_manifest = generate_normative_oracles(
            Path.cwd(),
            args.normative_output,
            workers=args.workers,
            progress=lambda completed, total: print(
                f"local replay progress: {completed}/{total} worlds",
                flush=True,
            ),
        )
        report = run_hidden_sensitivity(
            Path.cwd(),
            args.response_manifest,
            args.normative_output,
            args.output,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "provider_calls": report["provider_calls"],
                    "normative_worlds": normative_manifest["world_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "hidden-publication":
        publication_manifest = generate_publication_outputs(
            Path.cwd(),
            args.primary_report,
            args.sensitivity_report,
            args.output,
        )
        print(json.dumps(publication_manifest, sort_keys=True))
        return 0
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
    if args.command == "v2-local-development":
        report = run_v2_local_development(
            Path.cwd(),
            args.output,
            objective_rollouts=args.objective_rollouts,
            workers=args.workers,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["passed"] else 2
    if args.command == "v2-precision":
        report = run_precision_analysis(Path.cwd(), args.output)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "v2-dry-run":
        passed = run_v2_dry_run(args.output, Path.cwd())
        print(json.dumps({"output": str(args.output), "passed": passed}, sort_keys=True))
        return 0 if passed else 2
    if args.command == "v2-gates":
        report = development_gate_report(Path.cwd())
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0 if report["passed"] else 2
    if args.command == "v2-ood-pilot":
        report = run_structural_ood_pilot(Path.cwd(), args.output)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["passed"] else 2
    if args.command == "v2-audit":
        report = audit_reproduction_archive(args.archive)
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0 if report["passed"] else 2
    if args.command == "v2-hidden-seed":
        try:
            commitment = prepare_v2_seed(
                Path.cwd(),
                args.private_output,
                args.commitment,
                code_commit=args.code_commit,
            )
        except ValueError as error:
            raise SystemExit(f"refusing hidden-v2 seed: {error}") from error
        print(json.dumps({"commitment_sha256": commitment}, sort_keys=True))
        return 0
    if args.command == "v2-hidden-generate":
        try:
            passed = generate_v2_hidden_release(
                Path.cwd(),
                args.private_output,
                args.public_manifest,
                args.commitment,
                code_commit=args.code_commit,
                workers=args.workers,
            )
        except ValueError as error:
            raise SystemExit(f"refusing hidden-v2 generation: {error}") from error
        print(json.dumps({"public_manifest": str(args.public_manifest), "passed": passed}))
        return 0 if passed else 2
    if args.command == "v2-admission":
        report = build_provider_admission(tuple(args.manifests), args.output)
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0 if report["passed"] else 2
    if args.command == "v2-archive":
        archive_manifest = build_reproduction_archive(
            Path.cwd(),
            args.output,
            tuple(args.response_manifest),
            response_commit=args.response_commit,
        )
        print(
            json.dumps(
                {"output": str(args.output), "files": len(archive_manifest["sha256"])},
                sort_keys=True,
            )
        )
        return 0
    if args.command == "v2-unblind":
        try:
            report = run_v2_unblinding(
                Path.cwd(),
                tuple(args.response_manifest),
                args.output,
                response_commit=args.response_commit,
            )
        except ValueError as error:
            raise SystemExit(f"refusing hidden-v2 unblinding: {error}") from error
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "models": sorted(report["model_summaries"]),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "v2-publication":
        publication_manifest = generate_v2_publication_outputs(args.report, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "files": len(publication_manifest["sha256"]),
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
