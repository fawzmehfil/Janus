from __future__ import annotations

import json
from pathlib import Path

from janus.evaluation.budget import BudgetLedger, PricingConfig
from janus.models.base import ModelSettings
from janus.v2 import SCHEMA_VERSION
from janus.v2.collection import run_v2_collection
from janus.v2.providers import OpenAIV2Adapter


def test_collection_writes_blinded_hash_manifest(tmp_path: Path) -> None:
    prompt_directory = tmp_path / "configs/prompts"
    prompt_directory.mkdir(parents=True)
    (prompt_directory / "primary-v2.txt").write_text("{{OBSERVATION_JSON}}")
    public_path = tmp_path / "public.json"
    public_path.write_text(
        json.dumps(
            {
                "benchmark_version": "janus-v2",
                "trials_per_model": 1,
                "decision_schema_sha256": "schema-hash",
                "bundles": [
                    {
                        "bundle_id": "bundle-1",
                        "worlds": [
                            {
                                "world_id": "world-1",
                                "observation": {"rows": []},
                                "replicate_ids": ["replicate-00"],
                            }
                        ],
                    }
                ],
            }
        )
    )
    raw = json.dumps(
        {
            "diagnosis": "seller_collusion",
            "diagnosis_probabilities": {
                "supply_shortage": 0.1,
                "demand_increase": 0.1,
                "seller_collusion": 0.7,
                "momentum_demand": 0.1,
            },
            "action": "collusion_investigation",
            "confidence": 0.8,
        }
    )
    adapter = OpenAIV2Adapter(
        lambda payload: {
            "id": "response-1",
            "model": "mock",
            "status": "completed",
            "output_text": raw,
            "usage": {},
        },
        sdk_version="mock",
        timestamp=lambda: "now",
    )
    settings = ModelSettings(
        provider="openai",
        requested_model_id="mock",
        endpoint="local://mock",
        temperature=0,
        top_p=1,
        reasoning_effort=None,
        maximum_output_tokens=128,
        sdk_version="mock",
        prompt_version="primary-v2",
        schema_version=SCHEMA_VERSION,
    )
    pricing = PricingConfig(
        provider="mock",
        model_id="mock",
        effective_date="2026-10-19",
        source_url="local://mock",
        service_tier="local",
        input_per_million_usd=0,
        cached_input_per_million_usd=0,
        cache_write_per_million_usd=0,
        output_per_million_usd=0,
        reasoning_per_million_usd=0,
    )
    completed, manifest_path = run_v2_collection(
        tmp_path,
        public_path,
        tmp_path / "responses/luna",
        model_role="luna",
        adapter=adapter,
        settings=settings,
        budget=BudgetLedger(pricing, 0, {"hidden-v2": 0}),
    )
    manifest = json.loads(manifest_path.read_text())
    assert completed
    assert manifest["unblinded"] is False
    assert manifest["private_hidden_artifacts_accessed"] is False
    assert len(manifest["response_artifact_sha256"]) == 1
