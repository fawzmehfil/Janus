"""Zero-cost end-to-end Phase 0E dry run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from janus.evaluation.budget import BudgetLedger, PricingConfig
from janus.evaluation.harness import (
    SCHEMA_VERSION,
    EvaluationRunner,
    PromptRenderer,
    decision_schema_bytes,
)
from janus.models.base import ModelSettings, ResponseStatus, TrialContext
from janus.models.mock import MockAdapter
from janus.scenarios.generator import CandidateGeneratorConfig, generate_candidates
from janus.storage.cache import FileTrialCache


def _json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


def _valid_mock_reply(family: str) -> str:
    probabilities = {
        "supply_shortage": 0.1,
        "demand_increase": 0.1,
        "seller_collusion": 0.1,
        "momentum_demand": 0.1,
    }
    probabilities[family] = 0.7
    action = {
        "supply_shortage": "do_nothing",
        "demand_increase": "do_nothing",
        "seller_collusion": "collusion_investigation",
        "momentum_demand": "transaction_tax",
    }[family]
    return json.dumps(
        {
            "diagnosis": family,
            "diagnosis_probabilities": probabilities,
            "action": action,
            "confidence": 0.7,
            "rationale": "Synthetic mock decision used only to validate the local harness.",
        },
        separators=(",", ":"),
    )


def run_phase0e_dry_run(output_directory: Path, project_root: Path) -> bool:
    prompt_path = project_root / "configs/prompts/primary-v1.txt"
    pricing_path = project_root / "configs/pricing/mock-zero.json"
    prompt = PromptRenderer(prompt_path.read_text(), version="primary-v1")
    pricing = PricingConfig.model_validate(_json(pricing_path))
    settings = ModelSettings(
        provider="mock",
        requested_model_id="janus-mock-v1",
        endpoint="local://mock",
        temperature=0.0,
        top_p=1.0,
        reasoning_effort="none",
        maximum_output_tokens=512,
        sdk_version="janus-local-v1",
        prompt_version=prompt.version,
        schema_version=SCHEMA_VERSION,
    )
    generator = CandidateGeneratorConfig(candidates_per_family=2, root_seed=20261017)
    candidates = generate_candidates("phase0e-dry-run", generator)
    selected = (candidates[0], candidates[2], candidates[4], candidates[6], candidates[1])
    adapter = MockAdapter(tuple(_valid_mock_reply(world.family.value) for world in selected))
    budget = BudgetLedger(pricing, total_ceiling_usd=0.0, stage_ceilings_usd={"dry-run": 0.0})
    cache = FileTrialCache(output_directory / "cache")
    runner = EvaluationRunner(
        adapter=adapter,
        settings=settings,
        prompt=prompt,
        cache=cache,
        budget=budget,
    )
    artifacts = []
    for index, world in enumerate(selected):
        artifact = runner.run(
            world.serialized_observation,
            TrialContext(
                world_id=world.world_id,
                replicate_id=f"replicate-{index:02d}",
                benchmark_version="janus-v1",
                stage="dry-run",
            ),
        )
        artifacts.append(artifact)

    adapter_calls = adapter.call_count
    replayed = []
    for index, world in enumerate(selected):
        replayed.append(
            runner.run(
                world.serialized_observation,
                TrialContext(
                    world_id=world.world_id,
                    replicate_id=f"replicate-{index:02d}",
                    benchmark_version="janus-v1",
                    stage="dry-run",
                ),
            )
        )
    cache_replay_adapter_calls = adapter.call_count - adapter_calls

    cache_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((output_directory / "cache").glob("*.json"))
    }
    passed = (
        len(artifacts) == 5
        and all(item["validation_status"] == ResponseStatus.VALID.value for item in artifacts)
        and replayed == artifacts
        and cache_replay_adapter_calls == 0
        and budget.total_spent_usd == 0
    )
    manifest = {
        "phase": "0E",
        "mode": "mock-dry-run",
        "passed": passed,
        "paid_api_calls": 0,
        "adapter_calls": adapter_calls,
        "cache_replay_adapter_calls": cache_replay_adapter_calls,
        "cache_replay_passed": replayed == artifacts and cache_replay_adapter_calls == 0,
        "trial_count": len(artifacts),
        "total_cost_usd": budget.total_spent_usd,
        "prompt_sha256": prompt.hash,
        "schema_sha256": hashlib.sha256(decision_schema_bytes()).hexdigest(),
        "pricing_sha256": hashlib.sha256(pricing_path.read_bytes()).hexdigest(),
        "cache_artifact_sha256": cache_hashes,
        "trials": [
            {
                "logical_trial_id": item["logical_trial_id"],
                "cache_key": item["cache_key"],
                "status": item["validation_status"],
                "fallback_action": item["fallback_action"],
            }
            for item in artifacts
        ],
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    )
    return passed
