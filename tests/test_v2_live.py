from __future__ import annotations

import json
from pathlib import Path

import pytest

from janus.v2.live import (
    CONFIRMATIONS,
    FIREWORKS_MODEL_CONFIG,
    FIREWORKS_PRICING_CONFIG,
    run_fireworks_collection,
)


def _frozen_files(root: Path) -> None:
    model_source = Path.cwd() / FIREWORKS_MODEL_CONFIG
    pricing_source = Path.cwd() / FIREWORKS_PRICING_CONFIG
    for source in (model_source, pricing_source):
        target = root / source.relative_to(Path.cwd())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def test_paid_collection_refuses_before_transport_without_confirmation(tmp_path: Path) -> None:
    _frozen_files(tmp_path)
    with pytest.raises(ValueError, match="confirm-spend"):
        run_fireworks_collection(
            tmp_path,
            stage="smoke-v2",
            role="gpt_oss_120b",
            output_directory=tmp_path / "artifacts/v2-fireworks/smoke-v2/gpt_oss_120b",
            api_key="not-used",
            confirmation="wrong",
            ceiling_usd=0.1,
            base_url="https://api.fireworks.ai/inference/v1",
            timeout_seconds=1,
        )


def test_pricing_snapshot_has_four_exact_models_and_five_dollar_ceiling() -> None:
    pricing = json.loads(FIREWORKS_PRICING_CONFIG.read_text())
    models = json.loads(FIREWORKS_MODEL_CONFIG.read_text())
    assert pricing["project_maximum_usd"] == 5
    assert len(pricing["models"]) == 4
    assert {row["model_id"] for row in pricing["models"]} == {
        row["model_id"] for row in models["candidates"]
    }
    assert CONFIRMATIONS["hidden-v2"] == "I CONFIRM FIREWORKS V2 HIDDEN SPEND"
