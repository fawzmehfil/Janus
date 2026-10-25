"""Explicitly authorized, budget-bounded Fireworks collection commands."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from janus.evaluation.budget import BudgetLedger, PricingConfig
from janus.models.base import ModelSettings
from janus.v2 import BENCHMARK_VERSION, PROMPT_VERSION, SCHEMA_VERSION
from janus.v2.audit import development_gate_report
from janus.v2.collection import run_v2_collection
from janus.v2.contracts import decision_schema_bytes
from janus.v2.development import _selected_bundles
from janus.v2.providers import FireworksV2Adapter, make_fireworks_http_transport

FIREWORKS_PROJECT_CEILING_USD = 5.0
FIREWORKS_MODEL_CONFIG = Path("configs/models/v2-candidates.json")
FIREWORKS_PRICING_CONFIG = Path("configs/pricing/v2-fireworks-2026-10-21.json")
FIREWORKS_DEVELOPMENT_MANIFEST = Path("manifests/v2/development-public.json")
FIREWORKS_SMOKE_MANIFEST = Path("manifests/v2/smoke-public.json")
FIREWORKS_MODEL_ROLES = (
    "gpt_oss_120b",
    "deepseek_v4p1_flash",
    "glm_5p3_flash",
    "nemotron_lightning_3p5_30b_a3b",
)

CONFIRMATIONS = {
    "smoke-v2": "I CONFIRM FIREWORKS V2 SMOKE SPEND",
    "development-v2": "I CONFIRM FIREWORKS V2 DEVELOPMENT SPEND",
    "hidden-v2": "I CONFIRM FIREWORKS V2 HIDDEN SPEND",
}


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _verify_public_manifest(project_root: Path, path: Path, stage: str) -> bool:
    if not path.exists():
        return False
    config = _read(project_root / FIREWORKS_MODEL_CONFIG)
    expected = str(config["public_manifest_sha256"][stage])
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{stage} public manifest differs from its frozen SHA-256")
    return True


def build_development_manifest(project_root: Path) -> Path:
    """Materialize the predeclared 100 public admission trials."""

    output = project_root / FIREWORKS_DEVELOPMENT_MANIFEST
    if _verify_public_manifest(project_root, output, "development-v2"):
        if int(_read(output)["trials_per_model"]) != 100:
            raise ValueError("development admission manifest must contain exactly 100 trials")
        return output
    split_path = project_root / "artifacts/phase0d-release/split_manifest.json"
    split = _read(split_path)
    bundles = (
        *_selected_bundles("development", split["development"]["bundles"]),
        *_selected_bundles("calibration", split["calibration"]["bundles"]),
    )
    payload = {
        "schema_version": "janus-v2-public-collection-manifest-v1",
        "benchmark_version": BENCHMARK_VERSION,
        "stage": "development-v2",
        "source_split_manifest": str(split_path.relative_to(project_root)),
        "bundle_count": len(bundles),
        "trials_per_model": sum(len(bundle.worlds) for bundle in bundles),
        "decision_schema_sha256": hashlib.sha256(decision_schema_bytes()).hexdigest(),
        "bundles": [
            {
                "bundle_id": bundle.bundle_id,
                "worlds": [
                    {
                        "world_id": world.world_id,
                        "observation": json.loads(world.serialized_observation),
                        "replicate_ids": ["replicate-00"],
                    }
                    for world in bundle.worlds
                ],
            }
            for bundle in bundles
        ],
    }
    if payload["trials_per_model"] != 100:
        raise ValueError("development admission manifest must contain exactly 100 trials")
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if output.exists() and output.read_text() != encoded:
        raise ValueError("development manifest differs from its frozen on-disk version")
    _write(output, payload)
    _verify_public_manifest(project_root, output, "development-v2")
    return output


def build_smoke_manifest(project_root: Path) -> Path:
    """Freeze five deterministic development trials for transport smoke testing."""

    output = project_root / FIREWORKS_SMOKE_MANIFEST
    if _verify_public_manifest(project_root, output, "smoke-v2"):
        if int(_read(output)["trials_per_model"]) != 5:
            raise ValueError("smoke manifest must contain exactly five trials")
        return output
    development_path = build_development_manifest(project_root)
    development = _read(development_path)
    selected_worlds = [
        world
        for bundle in development["bundles"]
        for world in bundle["worlds"]
    ][:5]
    payload = {
        "schema_version": "janus-v2-public-collection-manifest-v1",
        "benchmark_version": BENCHMARK_VERSION,
        "stage": "smoke-v2",
        "source_manifest": str(development_path.relative_to(project_root)),
        "bundle_count": 5,
        "trials_per_model": 5,
        "decision_schema_sha256": development["decision_schema_sha256"],
        "bundles": [
            {"bundle_id": f"smoke-{index:02d}", "worlds": [world]}
            for index, world in enumerate(selected_worlds)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if output.exists() and output.read_text() != encoded:
        raise ValueError("smoke manifest differs from its frozen on-disk version")
    _write(output, payload)
    _verify_public_manifest(project_root, output, "smoke-v2")
    return output


def _frozen_model(project_root: Path, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    models = _read(project_root / FIREWORKS_MODEL_CONFIG)
    pricing = _read(project_root / FIREWORKS_PRICING_CONFIG)
    if models.get("status") != "exact_candidate_settings_frozen_for_admission":
        raise ValueError("Fireworks candidate settings are not frozen")
    if pricing.get("status") != "frozen_for_paid_admission":
        raise ValueError("Fireworks pricing snapshot is not frozen")
    model_rows = [row for row in models["candidates"] if row.get("role") == role]
    price_rows = [row for row in pricing["models"] if row.get("role") == role]
    if len(model_rows) != 1 or len(price_rows) != 1:
        raise ValueError(f"unknown or duplicate frozen model role: {role}")
    model, price = model_rows[0], price_rows[0]
    if model["model_id"] != price["model_id"] or model.get("enabled") is not True:
        raise ValueError(f"model/pricing mismatch for role {role}")
    return model, price


def _spent_in_cache(output_directory: Path) -> float:
    total = 0.0
    for path in (output_directory / "cache").glob("*.json"):
        payload = _read(path)
        total += float(payload.get("actual_cost_usd", 0.0))
    return total


def _other_project_spend(project_root: Path, output_directory: Path) -> float:
    total = 0.0
    root = project_root / "artifacts/v2-fireworks"
    for path in root.glob("**/cache/*.json"):
        if path.parent.parent.resolve() == output_directory.resolve():
            continue
        total += float(_read(path).get("actual_cost_usd", 0.0))
    return total


def run_fireworks_collection(
    project_root: Path,
    *,
    stage: str,
    role: str,
    output_directory: Path,
    api_key: str,
    confirmation: str,
    ceiling_usd: float,
    base_url: str,
    timeout_seconds: float,
) -> tuple[bool, Path]:
    """Run or resume one model/stage after all spend guards pass."""

    required = CONFIRMATIONS.get(stage)
    if required is None:
        raise ValueError(f"unsupported Fireworks collection stage: {stage}")
    if confirmation != required:
        raise ValueError(f"--confirm-spend must equal {required!r}")
    if not api_key.strip():
        raise ValueError("FIREWORKS_API_KEY is not set")
    if not 0 < ceiling_usd <= FIREWORKS_PROJECT_CEILING_USD:
        raise ValueError("--max-usd must be greater than zero and at most $5.00")
    model, price = _frozen_model(project_root, role)
    model_ceiling = float(price["maximum_project_allocation_usd"])
    if ceiling_usd > model_ceiling + 1e-12:
        raise ValueError(
            f"--max-usd exceeds the frozen ${model_ceiling:.2f} allocation for {role}"
        )
    frozen_base_url = str(model["serving_configuration"]["base_url"])
    if base_url.rstrip("/") != frozen_base_url.rstrip("/"):
        raise ValueError("base URL differs from the frozen serving configuration")
    prior_model_spend = _spent_in_cache(output_directory)
    if prior_model_spend > ceiling_usd + 1e-12:
        raise ValueError("existing model spend already exceeds the requested local ceiling")
    other_spend = _other_project_spend(project_root, output_directory)
    if other_spend + ceiling_usd > FIREWORKS_PROJECT_CEILING_USD + 1e-12:
        raise ValueError("local ceiling would exceed the $5.00 cumulative project ceiling")

    if stage == "smoke-v2":
        public_manifest = build_smoke_manifest(project_root)
    elif stage == "development-v2":
        public_manifest = build_development_manifest(project_root)
    else:
        gates = development_gate_report(project_root)
        if not gates["passed"]:
            failed = [row["name"] for row in gates["checks"] if not row["passed"]]
            raise ValueError(f"hidden collection is blocked by development gates: {failed}")
        public_manifest = project_root / "manifests/hidden-v2-public.json"
        if not public_manifest.exists():
            raise ValueError("hidden-v2 public manifest does not exist")

    pricing = PricingConfig(
        provider="fireworks",
        model_id=str(price["model_id"]),
        effective_date=str(price["retrieval_date"]),
        source_url=str(price["source_url"]),
        service_tier=str(price["service_tier"]),
        input_per_million_usd=float(price["input_per_million_usd"]),
        cached_input_per_million_usd=float(price["cached_input_per_million_usd"]),
        cache_write_per_million_usd=0,
        output_per_million_usd=float(price["output_per_million_usd"]),
        reasoning_per_million_usd=0,
    )
    settings = ModelSettings(
        provider="fireworks",
        requested_model_id=str(model["model_id"]),
        endpoint=f"{base_url.rstrip('/')}/chat/completions",
        temperature=model["serving_configuration"].get("temperature"),
        top_p=model["serving_configuration"].get("top_p"),
        reasoning_effort=model["reasoning_mode"].get("normalized_effort"),
        maximum_output_tokens=int(model["serving_configuration"]["maximum_output_tokens"]),
        sdk_version="stdlib-urllib-no-retry",
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
    )
    budget = BudgetLedger(
        pricing,
        total_ceiling_usd=ceiling_usd,
        stage_ceilings_usd={stage: ceiling_usd},
        total_spent_usd=prior_model_spend,
        stage_spent_usd={stage: prior_model_spend},
    )
    transport = make_fireworks_http_transport(
        api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    calls_this_invocation = 0

    def counted_transport(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls_this_invocation
        calls_this_invocation += 1
        return transport(payload)

    adapter = FireworksV2Adapter(
        counted_transport,
        sdk_version=settings.sdk_version,
        timestamp=lambda: datetime.now(UTC).isoformat(),
    )
    completed, manifest_path = run_v2_collection(
        project_root,
        public_manifest,
        output_directory,
        model_role=role,
        adapter=adapter,
        settings=settings,
        budget=budget,
        stage=stage,
    )
    manifest = _read(manifest_path)
    manifest.update(
        {
            "provider_calls_this_invocation": calls_this_invocation,
            "project_cumulative_cost_usd": (
                _other_project_spend(project_root, output_directory)
                + _spent_in_cache(output_directory)
            ),
            "project_hard_ceiling_usd": FIREWORKS_PROJECT_CEILING_USD,
        }
    )
    _write(manifest_path, manifest)
    return completed, manifest_path
