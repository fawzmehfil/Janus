"""Generate deterministic publication tables and vector figures."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, cast
from xml.sax.saxutils import escape


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _svg_text(x: float, y: float, value: str, **attributes: object) -> str:
    attrs = " ".join(f'{key.replace("_", "-")}="{item}"' for key, item in attributes.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {attrs}>{escape(value)}</text>'


def _forest_svg(
    path: Path,
    *,
    title: str,
    description: str,
    rows: list[tuple[str, float, float, float]],
    x_label: str,
    width: int = 1000,
) -> None:
    height = 150 + 72 * len(rows)
    left, right, top, bottom = 315, 70, 88, 68
    plot_width = width - left - right
    values = [number for _, estimate, low, high in rows for number in (estimate, low, high)]
    minimum = min([*values, 0.0])
    maximum = max([*values, 0.0])
    padding = max((maximum - minimum) * 0.10, 1.0)
    minimum -= padding
    maximum += padding

    def x(value: float) -> float:
        return left + (value - minimum) / (maximum - minimum) * plot_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(
            28,
            38,
            title,
            fill="#172033",
            font_family="Arial, sans-serif",
            font_size=24,
            font_weight=700,
        ),
        _svg_text(
            28,
            64,
            "Points are estimates; lines are 95% bundle-clustered intervals.",
            fill="#526078",
            font_family="Arial, sans-serif",
            font_size=14,
        ),
    ]
    zero_x = x(0.0)
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{top - 14}" x2="{zero_x:.1f}" '
        f'y2="{height - bottom}" stroke="#758197" stroke-width="1.5"/>'
    )
    for index, (label, estimate, low, high) in enumerate(rows):
        y = top + index * 72
        if index % 2 == 0:
            parts.append(
                f'<rect x="20" y="{y - 26}" width="{width - 40}" height="52" fill="#f5f7fa"/>'
            )
        parts.append(
            _svg_text(
                left - 18,
                y + 5,
                label,
                fill="#172033",
                font_family="Arial, sans-serif",
                font_size=16,
                text_anchor="end",
            )
        )
        parts.append(
            f'<line x1="{x(low):.1f}" y1="{y}" x2="{x(high):.1f}" y2="{y}" '
            'stroke="#1f5a8a" stroke-width="4" stroke-linecap="round"/>'
        )
        parts.append(
            f'<circle cx="{x(estimate):.1f}" cy="{y}" r="7" fill="#c44536" '
            'stroke="#ffffff" stroke-width="2"/>'
        )
        value_x = min(width - right + 8, x(maximum) + 8)
        parts.append(
            _svg_text(
                value_x,
                y + 5,
                f"{estimate:+.1f} [{low:+.1f}, {high:+.1f}]",
                fill="#172033",
                font_family="Arial, sans-serif",
                font_size=13,
                text_anchor="end",
            )
        )
    axis_y = height - bottom + 18
    for index in range(5):
        value = minimum + index * (maximum - minimum) / 4
        x_value = x(value)
        parts.extend(
            (
                f'<line x1="{x_value:.1f}" y1="{height - bottom}" x2="{x_value:.1f}" '
                f'y2="{height - bottom + 6}" stroke="#758197"/>',
                _svg_text(
                    x_value,
                    axis_y + 16,
                    f"{value:.0f}",
                    fill="#526078",
                    font_family="Arial, sans-serif",
                    font_size=12,
                    text_anchor="middle",
                ),
            )
        )
    parts.append(
        _svg_text(
            left + plot_width / 2,
            height - 12,
            x_label,
            fill="#172033",
            font_family="Arial, sans-serif",
            font_size=14,
            text_anchor="middle",
        )
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")


def _failure_profile_svg(path: Path, report: dict[str, Any]) -> None:
    family_order = (
        "supply_shortage",
        "demand_increase",
        "momentum_demand",
        "seller_collusion",
    )
    labels = {
        "supply_shortage": "Supply shortage",
        "demand_increase": "Demand increase",
        "momentum_demand": "Momentum demand",
        "seller_collusion": "Seller collusion",
    }
    values = [
        float(report["subgroups"]["family"][family]["mean_normalized_regret"])
        for family in family_order
    ]
    width, height = 1000, 475
    left, right, top = 235, 90, 95
    plot_width = width - left - right
    row_height = 70
    maximum = max(values) * 1.15
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Regret was concentrated in seller-collusion worlds</title>',
        '<desc id="desc">Horizontal bars compare mean normalized regret across four '
        "scenario families.</desc>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(
            28,
            38,
            "Regret was concentrated in seller-collusion worlds",
            fill="#172033",
            font_family="Arial, sans-serif",
            font_size=24,
            font_weight=700,
        ),
        _svg_text(
            28,
            64,
            "Mean regret as a share of the 20-period reference surplus.",
            fill="#526078",
            font_family="Arial, sans-serif",
            font_size=14,
        ),
    ]
    for index, (family, value) in enumerate(zip(family_order, values, strict=True)):
        y = top + index * row_height
        bar_width = value / maximum * plot_width
        fill = "#c44536" if family == "seller_collusion" else "#1f5a8a"
        parts.append(
            _svg_text(
                left - 18,
                y + 23,
                labels[family],
                fill="#172033",
                font_family="Arial, sans-serif",
                font_size=16,
                text_anchor="end",
            )
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="34" fill="{fill}"/>'
        )
        parts.append(
            _svg_text(
                left + bar_width + 10,
                y + 23,
                f"{100 * value:.3f}%",
                fill="#172033",
                font_family="Arial, sans-serif",
                font_size=14,
            )
        )
    axis_y = top + len(values) * row_height
    parts.append(
        f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_width}" y2="{axis_y}" stroke="#758197"/>'
    )
    for index in range(4):
        value = index * maximum / 3
        x_value = left + value / maximum * plot_width
        parts.extend(
            (
                f'<line x1="{x_value:.1f}" y1="{axis_y}" x2="{x_value:.1f}" '
                f'y2="{axis_y + 6}" stroke="#758197"/>',
                _svg_text(
                    x_value,
                    axis_y + 23,
                    f"{100 * value:.1f}%",
                    fill="#526078",
                    font_family="Arial, sans-serif",
                    font_size=12,
                    text_anchor="middle",
                ),
            )
        )
    parts.append(
        _svg_text(
            left + plot_width / 2,
            height - 14,
            "Mean normalized full-information regret",
            fill="#172033",
            font_family="Arial, sans-serif",
            font_size=14,
            text_anchor="middle",
        )
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")


def generate_publication_outputs(
    project_root: Path,
    primary_report_path: Path,
    sensitivity_report_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Write CSV tables and accessible SVG figures from immutable reports."""

    paths = [primary_report_path, sensitivity_report_path, output_directory]
    primary_report_path, sensitivity_report_path, output_directory = [
        path if path.is_absolute() else project_root / path for path in paths
    ]
    primary = _read(primary_report_path)
    sensitivity = _read(sensitivity_report_path)
    tables = output_directory / "tables"
    figures = output_directory / "figures"

    comparator_fields = (
        ("Uniform random", "mean_utility_difference_vs_random"),
        ("Always do nothing", "mean_utility_difference_vs_do_nothing"),
        ("Frozen heuristic", "mean_utility_difference_vs_heuristic"),
        ("Public-information ridge", "mean_utility_difference_vs_partial_information"),
    )
    forest_rows = []
    primary_rows: list[tuple[object, ...]] = []
    for label, field in comparator_fields:
        interval = primary["primary"][field]
        forest_rows.append(
            (
                label,
                float(interval["estimate"]),
                float(interval["lower_95"]),
                float(interval["upper_95"]),
            )
        )
        primary_rows.append(
            (
                label,
                interval["estimate"],
                interval["lower_95"],
                interval["upper_95"],
            )
        )
    _write_csv(
        tables / "primary-policy-comparisons.csv",
        ("comparator", "estimate", "lower_95", "upper_95"),
        primary_rows,
    )
    _forest_svg(
        figures / "primary-policy-comparisons.svg",
        title="Luna beat random, but not the non-random baselines",
        description=(
            "Mean paired utility differences show a decisive gain over uniform random, "
            "no clear gain over doing nothing or the frozen heuristic, and a deficit to "
            "the public-information ridge baseline."
        ),
        rows=forest_rows,
        x_label="Luna utility minus comparator utility",
    )

    subgroup_rows: list[tuple[object, ...]] = []
    for field in ("stratum", "family"):
        for name, values in primary["subgroups"][field].items():
            subgroup_rows.append(
                (
                    field,
                    name,
                    values["trials"],
                    values["mean_normalized_regret"],
                    values["epsilon_optimal_rate"],
                    values["valid_rate"],
                )
            )
    _write_csv(
        tables / "subgroup-results.csv",
        (
            "group_type",
            "group",
            "trials",
            "mean_normalized_regret",
            "epsilon_optimal_rate",
            "valid_response_rate",
        ),
        subgroup_rows,
    )
    _failure_profile_svg(figures / "family-regret.svg", primary)

    sensitivity_rows: list[tuple[object, ...]] = []
    for objective, scoring_rules in sensitivity["analyses"].items():
        for scoring_rule, summary in scoring_rules.items():
            for comparator, field in comparator_fields:
                interval = summary[field]
                sensitivity_rows.append(
                    (
                        objective,
                        scoring_rule,
                        comparator,
                        summary["trials"],
                        interval["estimate"],
                        interval["lower_95"],
                        interval["upper_95"],
                    )
                )
    _write_csv(
        tables / "sensitivity-results.csv",
        (
            "objective",
            "scoring_rule",
            "comparator",
            "trials",
            "estimate",
            "lower_95",
            "upper_95",
        ),
        sensitivity_rows,
    )

    normative_rows: list[tuple[object, ...]] = []
    for objective in sensitivity["objectives"]:
        summary = sensitivity["analyses"][objective]["intention_to_evaluate"]
        random_interval = summary["mean_utility_difference_vs_random"]
        do_nothing_interval = summary["mean_utility_difference_vs_do_nothing"]
        partial_interval = summary["mean_utility_difference_vs_partial_information"]
        normative_rows.append(
            (
                objective,
                random_interval["estimate"],
                random_interval["lower_95"],
                random_interval["upper_95"],
                do_nothing_interval["estimate"],
                do_nothing_interval["lower_95"],
                do_nothing_interval["upper_95"],
                partial_interval["estimate"],
                partial_interval["lower_95"],
                partial_interval["upper_95"],
                summary["best_constant_action"],
            )
        )
    _write_csv(
        tables / "normative-summary.csv",
        (
            "objective",
            "vs_random_estimate",
            "vs_random_lower_95",
            "vs_random_upper_95",
            "vs_do_nothing_estimate",
            "vs_do_nothing_lower_95",
            "vs_do_nothing_upper_95",
            "vs_public_information_estimate",
            "vs_public_information_lower_95",
            "vs_public_information_upper_95",
            "best_constant_action",
        ),
        normative_rows,
    )

    invalid_rows: list[tuple[object, ...]] = []
    for scoring_rule in sensitivity["scoring_rules"]:
        summary = sensitivity["analyses"]["primary"][scoring_rule]
        do_nothing_interval = summary["mean_utility_difference_vs_do_nothing"]
        partial_interval = summary["mean_utility_difference_vs_partial_information"]
        invalid_rows.append(
            (
                scoring_rule,
                summary["trials"],
                do_nothing_interval["estimate"],
                do_nothing_interval["lower_95"],
                do_nothing_interval["upper_95"],
                partial_interval["estimate"],
                partial_interval["lower_95"],
                partial_interval["upper_95"],
            )
        )
    _write_csv(
        tables / "invalid-response-sensitivity.csv",
        (
            "scoring_rule",
            "trials",
            "vs_do_nothing_estimate",
            "vs_do_nothing_lower_95",
            "vs_do_nothing_upper_95",
            "vs_public_information_estimate",
            "vs_public_information_lower_95",
            "vs_public_information_upper_95",
        ),
        invalid_rows,
    )

    objective_labels = {
        "primary": "Primary",
        "consumer_protection": "Consumer protection",
        "producer_continuity": "Producer continuity",
        "stability_aware": "Stability aware",
        "fiscal_neutral": "Fiscal neutral",
    }
    robustness_rows = []
    for objective in sensitivity["objectives"]:
        interval = sensitivity["analyses"][objective]["intention_to_evaluate"][
            "mean_utility_difference_vs_partial_information"
        ]
        robustness_rows.append(
            (
                objective_labels[objective],
                float(interval["estimate"]),
                float(interval["lower_95"]),
                float(interval["upper_95"]),
            )
        )
    _forest_svg(
        figures / "normative-robustness-vs-public-information.svg",
        title="The public-information baseline remained stronger across objectives",
        description=(
            "Paired utility differences between Luna and the frozen observation-only ridge "
            "under five preregistered welfare objectives."
        ),
        rows=robustness_rows,
        x_label="Luna utility minus public-information baseline utility",
    )

    primary_sensitivity_rows = []
    scoring_labels = {
        "intention_to_evaluate": "Do-nothing fallback (primary)",
        "valid_only": "Valid responses only",
        "incomplete_as_worst": "Incomplete as worst action",
    }
    for scoring_rule in sensitivity["scoring_rules"]:
        interval = sensitivity["analyses"]["primary"][scoring_rule][
            "mean_utility_difference_vs_do_nothing"
        ]
        primary_sensitivity_rows.append(
            (
                scoring_labels[scoring_rule],
                float(interval["estimate"]),
                float(interval["lower_95"]),
                float(interval["upper_95"]),
            )
        )
    _forest_svg(
        figures / "invalid-response-sensitivity.svg",
        title="Invalid-response treatment did not rescue the primary conclusion",
        description=(
            "Paired Luna-versus-do-nothing utility differences under the preregistered "
            "fallback, valid-only, and worst-action scoring rules."
        ),
        rows=primary_sensitivity_rows,
        x_label="Luna utility minus always-do-nothing utility",
    )

    manifest = {
        "schema_version": "janus-publication-outputs-v1",
        "primary_report": str(primary_report_path.relative_to(project_root)),
        "primary_report_sha256": _sha256(primary_report_path),
        "sensitivity_report": str(sensitivity_report_path.relative_to(project_root)),
        "sensitivity_report_sha256": _sha256(sensitivity_report_path),
        "tables": sorted(path.name for path in tables.glob("*.csv")),
        "figures": sorted(path.name for path in figures.glob("*.svg")),
        "table_sha256": {path.name: _sha256(path) for path in sorted(tables.glob("*.csv"))},
        "figure_sha256": {path.name: _sha256(path) for path in sorted(figures.glob("*.svg"))},
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "publication-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    return manifest
