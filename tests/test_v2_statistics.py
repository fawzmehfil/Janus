from __future__ import annotations

import numpy as np

from janus.v2.statistics import (
    _action_behavior,
    _classification_metrics,
    _risk_coverage,
    fit_hierarchical_mixed_model,
    holm_adjust,
)


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}


def test_holm_handles_empty_family() -> None:
    assert holm_adjust({}) == {}


def test_mixed_model_separates_nested_sampling_effects() -> None:
    rng = np.random.default_rng(12)
    models = ("luna", "strong", "anthropic", "google")
    model_families = ("openai", "openai", "anthropic", "google")
    families = (
        "supply_shortage",
        "demand_increase",
        "seller_collusion",
        "momentum_demand",
    )
    rows = []
    for bundle in range(8):
        bundle_effect = rng.normal(0, 0.3)
        for world, family in enumerate(families):
            world_effect = rng.normal(0, 0.2)
            for replicate in range(2):
                replicate_effect = rng.normal(0, 0.1)
                for model_index, (model, model_family) in enumerate(
                    zip(models, model_families, strict=True)
                ):
                    rows.append(
                        {
                            "decision_loss": model_index
                            + 0.5 * (family == "seller_collusion")
                            + 0.4 * (bundle >= 4)
                            + bundle_effect
                            + world_effect
                            + replicate_effect
                            + rng.normal(0, 0.05),
                            "model_id": model,
                            "model_family": model_family,
                            "bundle_id": f"b{bundle}",
                            "world_id": f"b{bundle}-w{world}",
                            "replicate_id": f"r{replicate}",
                            "family": family,
                            "stratum": "ood" if bundle >= 4 else "id",
                        }
                    )
    result = fit_hierarchical_mixed_model(rows)
    assert result["method"] == "Gaussian linear mixed model fit by REML"
    assert set(result["variance_components"]) == {
        "bundle",
        "world_within_bundle",
        "replicate_within_world",
        "residual",
    }


def test_behavioral_metrics_cover_confusion_repeatability_and_selective_risk() -> None:
    rows = []
    for replicate, action in enumerate(("do_nothing", "do_nothing")):
        rows.append(
            {
                "world_id": "w1",
                "selected_action": action,
                "oracle_optimal_action": "do_nothing",
                "family": "supply_shortage",
                "stratum": "id",
                "diagnosis_probabilities": {
                    "supply_shortage": 0.7,
                    "demand_increase": 0.1,
                    "seller_collusion": 0.1,
                    "momentum_demand": 0.1,
                },
                "confidence": 0.8 - replicate * 0.1,
                "epsilon_optimal": True,
                "full_information_regret": float(replicate),
            }
        )
    classification = _classification_metrics(rows)
    behavior = _action_behavior(rows)
    curve = _risk_coverage(rows)
    assert classification["accuracy"] == 1
    assert classification["macro_f1"] == 0.25
    assert behavior["unanimous_world_rate"] == 1
    assert behavior["action_entropy_nats"] == 0
    assert curve[-1]["retained_trials"] == 2
