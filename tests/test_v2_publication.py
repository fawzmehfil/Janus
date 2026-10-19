from __future__ import annotations

import json
from pathlib import Path

from janus.v2.publication import generate_v2_publication_outputs


def test_v2_publication_outputs_are_deterministic_and_manifested(tmp_path: Path) -> None:
    model_summary = {
        "paired_utility_difference_vs_public_policy": {
            "estimate": 1.0,
            "lower_95": -1.0,
            "upper_95": 3.0,
            "independent_bundles": 40,
        },
        "epsilon_optimal_action_rate": 0.75,
        "completion_rate": 1.0,
        "trial_count": 40,
        "mean_normalized_full_information_regret": 0.2,
        "worst_decile_normalized_regret": 0.6,
        "maximum_full_information_regret": 10.0,
        "economic_outcome_means": {"net_total_surplus": 100.0},
        "calibration": {
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "diagnosis_brier": 0.7,
            "clipped_log_loss": 1.2,
            "action_confidence_brier": 0.2,
            "diagnosis_reliability": [
                {
                    "lower": 0.5,
                    "upper": 0.6,
                    "count": 10,
                    "mean_confidence": 0.55,
                    "observed_rate": 0.6,
                    "lower_95": 0.3,
                    "upper_95": 0.9,
                }
            ],
            "action_confidence_reliability": [],
        },
        "operations": {
            "actual_cost_usd": 0.01,
            "mean_latency_ms": 100,
            "tokens": {"input_tokens": 1000, "output_tokens": 100},
        },
        "behavior": {
            "action_counts": {"do_nothing": 40},
            "action_counts_by_family": {"supply_shortage": {"do_nothing": 40}},
            "action_counts_by_stratum": {"id": {"do_nothing": 40}},
            "action_counts_by_oracle_optimal_action": {"do_nothing": {"do_nothing": 40}},
        },
        "risk_coverage": [
            {
                "target_coverage": 1.0,
                "realized_coverage_of_all_trials": 1.0,
                "retained_trials": 40,
                "mean_full_information_regret": 2.0,
                "epsilon_optimal_rate": 0.75,
            }
        ],
    }
    report = {
        "model_summaries": {"model-a": model_summary},
        "pairwise_model_comparisons": {},
        "secondary_policy_references": {
            "economic_outcome_means": {
                "do_nothing": {"net_total_surplus": 90.0},
                "constant_actions": {},
            }
        },
        "objective_specific_sensitivity": {},
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    output = tmp_path / "publication"
    first = generate_v2_publication_outputs(report_path, output)
    second = generate_v2_publication_outputs(report_path, output)
    assert first == second
    assert (output / "tables/primary-model-estimates.csv").is_file()
    assert (output / "figures/primary-model-estimates.svg").is_file()
    assert set(first["sha256"]) == {
        "figures/primary-model-estimates.svg",
        "tables/calibration-reliability.csv",
        "tables/action-counts.csv",
        "tables/economic-outcomes.csv",
        "tables/model-performance-and-calibration.csv",
        "tables/objective-sensitivity.csv",
        "tables/operations.csv",
        "tables/pairwise-model-comparisons.csv",
        "tables/primary-model-estimates.csv",
        "tables/risk-coverage.csv",
    }
