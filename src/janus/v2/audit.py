"""Independent artifact audit and pre-release gate checks for Janus v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from janus.interventions.actions import ActionID
from janus.v2 import PROMPT_VERSION, SCHEMA_VERSION
from janus.v2.contracts import decision_schema_bytes


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def development_gate_report(project_root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append({"name": name, "passed": passed, "evidence": evidence})

    local_manifest = project_root / "artifacts/v2-development/run_manifest.json"
    check(
        "local_development",
        local_manifest.exists() and bool(_read(local_manifest).get("passed")),
        str(local_manifest),
    )
    precision = project_root / "reports/v2/precision.json"
    check("precision_analysis", precision.exists(), str(precision))
    coverage = project_root / "reports/v2/action-coverage.json"
    check(
        "action_margin_coverage",
        coverage.exists() and bool(_read(coverage).get("passed")),
        str(coverage),
    )
    contrasts = project_root / "reports/v2/matched-contrasts.json"
    check(
        "matched_causal_contrasts",
        contrasts.exists() and bool(_read(contrasts).get("passed")),
        str(contrasts),
    )
    baseline_dir = project_root / "manifests/v2/baselines"
    objectives = (
        "primary",
        "consumer_protection",
        "producer_continuity",
        "stability_aware",
        "fiscal_neutral",
    )
    baseline_ok = True
    for objective in objectives:
        manifest_path = baseline_dir / f"{objective}.json"
        if not manifest_path.exists():
            baseline_ok = False
            continue
        manifest = _read(manifest_path)
        model_path = baseline_dir / str(manifest.get("model_file", ""))
        baseline_ok = (
            baseline_ok
            and model_path.exists()
            and (_sha256(model_path) == manifest.get("model_sha256"))
        )
        baseline_ok = baseline_ok and bool(
            manifest.get("learning_curve_stability", {}).get("stabilized")
        )
    check("five_frozen_public_baselines", baseline_ok, str(baseline_dir))
    validity_path = project_root / "artifacts/phase0d-release/validity_report.json"
    validity = _read(validity_path) if validity_path.exists() else {}
    selection = (
        _read(baseline_dir / "primary.json").get("selected_candidate")
        if (baseline_dir / "primary.json").exists()
        else None
    )
    above_heuristic = any(
        row.get("name") == "partial_above_heuristic" and row.get("passed") is True
        for row in validity.get("gates", [])
    )
    check(
        "primary_public_policy_above_heuristic",
        selection == "ridge:alpha=100.0" and above_heuristic,
        str(validity_path),
    )

    model_config = _read(project_root / "configs/models/v2-candidates.json")
    exact_models_frozen = False
    admission_path = project_root / "manifests/v2/model-admission.json"
    admissions = _read(admission_path) if admission_path.exists() else {}
    configured_ids = {
        str(row.get("model_id"))
        for row in model_config["candidates"]
        if row.get("enabled") is True and isinstance(row.get("model_id"), str)
    }
    admitted_ids = {
        str(row.get("requested_model_id"))
        for row in admissions.get("models", [])
        if row.get("admitted") is True
    }
    exact_models_frozen = (
        model_config.get("status") == "exact_candidate_settings_frozen_for_admission"
        and len(configured_ids) == 4
        and configured_ids == admitted_ids
    )
    check(
        "exact_models_admitted_and_frozen",
        exact_models_frozen,
        str(admission_path),
    )
    check(
        "four_model_admission_gate",
        admission_path.exists()
        and len(admissions.get("models", [])) == 4
        and all(
            row.get("admitted") and row.get("model_identity_match")
            for row in admissions.get("models", [])
        ),
        str(admission_path),
    )
    ood_path = project_root / "reports/v2/structural-ood.json"
    check(
        "structural_ood_reversal",
        ood_path.exists() and bool(_read(ood_path).get("passed")),
        str(ood_path),
    )
    robustness_path = project_root / "artifacts/robustness/report.json"
    convergence = validity.get("oracle_convergence", {})
    robustness_ok = (
        robustness_path.exists()
        and bool(_read(robustness_path).get("clearing_stable"))
        and bool(convergence.get("passed"))
        and (project_root / "docs/v2/ROBUSTNESS_AND_FALSIFICATION.md").exists()
    )
    check(
        "robustness_and_falsification_matrix",
        robustness_ok,
        "docs/v2/ROBUSTNESS_AND_FALSIFICATION.md",
    )
    prompt_review = project_root / "manifests/v2/prompt-review-approval.json"
    prompt_review_payload = _read(prompt_review) if prompt_review.exists() else {}
    prompt_path = project_root / "configs/prompts/primary-v2.txt"
    reviewed_prompt_current = (
        prompt_path.exists()
        and prompt_review_payload.get("prompt_sha256") == _sha256(prompt_path)
        and prompt_review_payload.get("prompt_version") == PROMPT_VERSION
    )
    reviewed_schema_current = (
        prompt_review_payload.get("decision_schema_sha256")
        == hashlib.sha256(decision_schema_bytes()).hexdigest()
        and prompt_review_payload.get("decision_schema_version") == SCHEMA_VERSION
    )
    criteria = prompt_review_payload.get("criteria", {})
    required_criteria = (
        "intervention_semantics_match_simulator",
        "welfare_language_matches_ledger",
        "confidence_is_action_optimality_probability",
        "provider_neutral_wording",
    )
    check(
        "evidence_based_prompt_rationale_review",
        prompt_review.exists()
        and prompt_review_payload.get("approved") is True
        and prompt_review_payload.get("review_type") == "ai_assisted_internal_desk_review"
        and prompt_review_payload.get("independent_reviewer") is False
        and reviewed_prompt_current
        and reviewed_schema_current
        and isinstance(criteria, dict)
        and all(criteria.get(name) is True for name in required_criteria),
        str(prompt_review),
    )
    return {
        "schema_version": "janus-v2-development-gates-v1",
        "checks": checks,
        "passed": all(row["passed"] for row in checks),
        "hidden_v2_generation_authorized": False,
        "note": "passing checks is necessary but does not itself authorize a paid or hidden run",
    }


def audit_reproduction_archive(archive: Path) -> dict[str, Any]:
    """Verify a completed archive without claiming auditor independence."""

    required = (
        "archive-manifest.json",
        "LICENSE",
        "CITATION.cff",
        "configs/experiments/v2-analysis-preregistration.json",
        "manifests/hidden-v2-commitment.json",
        "manifests/hidden-v2-public.json",
        "private/hidden-v2/private_manifest.json",
        "private/hidden-v2/seed_manifest.json",
        "analysis/comparative-report.json",
        "reports/tables",
        "reports/figures",
    )
    missing = [name for name in required if not (archive / name).exists()]
    hash_failures: list[str] = []
    manifest_path = archive / "archive-manifest.json"
    if manifest_path.exists():
        manifest = _read(manifest_path)
        for name, expected in manifest.get("sha256", {}).items():
            path = archive / str(name)
            if not path.is_file() or _sha256(path) != expected:
                hash_failures.append(str(name))
    response_manifests = tuple((archive / "responses").glob("*/manifests/run-manifest.json"))
    response_manifest_ok = len(response_manifests) == 4
    if response_manifest_ok:
        response_manifest_ok = all(
            not _read(path).get("unblinded", True) for path in response_manifests
        )
    return {
        "schema_version": "janus-v2-independent-audit-v1",
        "archive": str(archive),
        "missing_required_paths": missing,
        "hash_failures": hash_failures,
        "four_blinded_response_manifests_present": response_manifest_ok,
        "passed": not missing and not hash_failures and response_manifest_ok,
        "self_certification": False,
        "auditor_instruction": (
            "Run this command from an independently controlled checkout and record auditor "
            "identity, date, repository revision, and the JSON output. The project authors "
            "must not represent their own run as the independent audit."
        ),
        "action_ids_verified": [action.value for action in ActionID],
    }
