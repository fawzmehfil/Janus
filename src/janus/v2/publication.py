"""Deterministic Janus v2 publication tables and vector figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from janus.evaluation.publication import _forest_svg, _write_csv


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def generate_v2_publication_outputs(report_path: Path, output_directory: Path) -> dict[str, Any]:
    """Regenerate the preregistered core tables and primary forest plot."""

    report = _read(report_path)
    tables = output_directory / "tables"
    figures = output_directory / "figures"
    primary_rows: list[tuple[object, ...]] = []
    forest_rows = []
    for model, summary in sorted(report["model_summaries"].items()):
        interval = summary["paired_utility_difference_vs_public_policy"]
        row = (
            model,
            interval["estimate"],
            interval["lower_95"],
            interval["upper_95"],
            interval["independent_bundles"],
            summary["epsilon_optimal_action_rate"],
            summary["completion_rate"],
        )
        primary_rows.append(row)
        forest_rows.append((model, float(row[1]), float(row[2]), float(row[3])))
    _write_csv(
        tables / "primary-model-estimates.csv",
        (
            "model",
            "utility_difference_vs_public_policy",
            "lower_95",
            "upper_95",
            "independent_bundles",
            "epsilon_optimal_rate",
            "valid_completion_rate",
        ),
        primary_rows,
    )
    _forest_svg(
        figures / "primary-model-estimates.svg",
        title="Paired utility relative to the frozen public-information policy",
        description=(
            "Positive estimates favor the model. Intervals use ambiguity bundles as the "
            "independent sampling unit."
        ),
        rows=forest_rows,
        x_label="Model utility minus frozen public-policy utility",
    )

    pairwise_rows: list[tuple[object, ...]] = [
        (
            comparison,
            values["estimate"],
            values["raw_p_value"],
            values["holm_adjusted_p_value"],
        )
        for comparison, values in sorted(report["pairwise_model_comparisons"].items())
    ]
    _write_csv(
        tables / "pairwise-model-comparisons.csv",
        ("comparison", "decision_loss_difference", "raw_p_value", "holm_p_value"),
        pairwise_rows,
    )

    economic_rows: list[tuple[object, ...]] = []
    for model, summary in sorted(report["model_summaries"].items()):
        economic_rows.extend(
            ("model", model, metric, value)
            for metric, value in sorted(summary["economic_outcome_means"].items())
        )
    for baseline, values in sorted(
        report.get("secondary_policy_references", {}).get("economic_outcome_means", {}).items()
    ):
        if baseline == "constant_actions":
            continue
        economic_rows.extend(
            ("baseline", baseline, metric, value) for metric, value in sorted(values.items())
        )
    _write_csv(
        tables / "economic-outcomes.csv",
        ("kind", "configuration", "metric", "mean"),
        economic_rows,
    )

    calibration_rows: list[tuple[object, ...]] = []
    for model, summary in sorted(report["model_summaries"].items()):
        for target in ("diagnosis_reliability", "action_confidence_reliability"):
            calibration_rows.extend(
                (
                    model,
                    target,
                    row["lower"],
                    row["upper"],
                    row["count"],
                    row["mean_confidence"],
                    row["observed_rate"],
                    row["lower_95"],
                    row["upper_95"],
                )
                for row in summary["calibration"][target]
            )
    _write_csv(
        tables / "calibration-reliability.csv",
        (
            "model",
            "target",
            "bin_lower",
            "bin_upper",
            "count",
            "mean_confidence",
            "observed_rate",
            "lower_95",
            "upper_95",
        ),
        calibration_rows,
    )

    performance_rows: list[tuple[object, ...]] = []
    operations_rows: list[tuple[object, ...]] = []
    action_rows: list[tuple[object, ...]] = []
    risk_rows: list[tuple[object, ...]] = []
    for model, summary in sorted(report["model_summaries"].items()):
        calibration = summary["calibration"]
        performance_rows.append(
            (
                model,
                summary["trial_count"],
                summary["completion_rate"],
                summary["epsilon_optimal_action_rate"],
                summary["mean_normalized_full_information_regret"],
                summary["worst_decile_normalized_regret"],
                summary["maximum_full_information_regret"],
                calibration["accuracy"],
                calibration["macro_f1"],
                calibration["diagnosis_brier"],
                calibration["clipped_log_loss"],
                calibration["action_confidence_brier"],
            )
        )
        operations = summary["operations"]
        operations_rows.append(
            (
                model,
                operations["actual_cost_usd"],
                operations["mean_latency_ms"],
                *(operations["tokens"][key] for key in sorted(operations["tokens"])),
            )
        )
        action_rows.extend(
            (model, "overall", "all", action, count)
            for action, count in sorted(summary["behavior"]["action_counts"].items())
        )
        for grouping in (
            "action_counts_by_family",
            "action_counts_by_stratum",
            "action_counts_by_oracle_optimal_action",
        ):
            action_rows.extend(
                (model, grouping, level, action, count)
                for level, counts in sorted(summary["behavior"][grouping].items())
                for action, count in sorted(counts.items())
            )
        risk_rows.extend(
            (
                model,
                row["target_coverage"],
                row["realized_coverage_of_all_trials"],
                row["retained_trials"],
                row["mean_full_information_regret"],
                row["epsilon_optimal_rate"],
            )
            for row in summary["risk_coverage"]
        )
    _write_csv(
        tables / "model-performance-and-calibration.csv",
        (
            "model",
            "trials",
            "valid_completion_rate",
            "epsilon_optimal_rate",
            "mean_normalized_regret",
            "worst_decile_normalized_regret",
            "maximum_regret",
            "diagnosis_accuracy",
            "diagnosis_macro_f1",
            "diagnosis_brier",
            "diagnosis_log_loss",
            "action_confidence_brier",
        ),
        performance_rows,
    )
    token_fields = tuple(
        sorted(next(iter(report["model_summaries"].values()))["operations"]["tokens"])
    )
    _write_csv(
        tables / "operations.csv",
        ("model", "actual_cost_usd", "mean_latency_ms", *token_fields),
        operations_rows,
    )
    _write_csv(
        tables / "action-counts.csv",
        ("model", "grouping", "group", "action", "count"),
        action_rows,
    )
    _write_csv(
        tables / "risk-coverage.csv",
        (
            "model",
            "target_coverage",
            "realized_coverage",
            "retained_trials",
            "mean_full_information_regret",
            "epsilon_optimal_rate",
        ),
        risk_rows,
    )

    objective_rows: list[tuple[object, ...]] = []
    objective_forest_rows = []
    for objective, values in sorted(report.get("objective_specific_sensitivity", {}).items()):
        for model in sorted(report["model_summaries"]):
            interval = values[model]
            objective_rows.append(
                (
                    objective,
                    model,
                    interval["estimate"],
                    interval["lower_95"],
                    interval["upper_95"],
                )
            )
            objective_forest_rows.append(
                (
                    f"{objective}: {model}",
                    -float(interval["estimate"]),
                    -float(interval["upper_95"]),
                    -float(interval["lower_95"]),
                )
            )
    _write_csv(
        tables / "objective-sensitivity.csv",
        ("objective", "model", "decision_loss", "lower_95", "upper_95"),
        objective_rows,
    )
    if objective_forest_rows:
        _forest_svg(
            figures / "objective-sensitivity.svg",
            title="Objective-specific utility relative to each frozen public policy",
            description=(
                "Positive values favor the model. Each objective uses its separately frozen "
                "observation-only policy."
            ),
            rows=objective_forest_rows,
            x_label="Model utility minus objective-specific public-policy utility",
        )

    files = sorted(
        path
        for path in output_directory.rglob("*")
        if path.is_file() and path.name != "publication-manifest.json"
    )
    manifest = {
        "schema_version": "janus-v2-publication-manifest-v1",
        "source_report": str(report_path),
        "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "sha256": {
            str(path.relative_to(output_directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    manifest_path = output_directory / "publication-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest
