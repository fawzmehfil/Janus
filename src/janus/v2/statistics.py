"""Preregistered paired and hierarchical analysis primitives for Janus v2."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp

FAMILIES = (
    "supply_shortage",
    "demand_increase",
    "seller_collusion",
    "momentum_demand",
)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm family-wise error correction with monotone adjusted values."""

    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (count - rank) * p_values[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def paired_bundle_interval(
    rows: list[dict[str, Any]],
    value_key: str,
    *,
    seed: int,
    samples: int = 10_000,
) -> dict[str, float | int]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["bundle_id"])].append(float(row[value_key]))
    bundle_values = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)])
    if len(bundle_values) < 2:
        raise ValueError("paired bundle interval requires at least two bundles")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for index in range(samples):
        selected = rng.integers(0, len(bundle_values), size=len(bundle_values))
        draws[index] = np.mean(bundle_values[selected])
    return {
        "estimate": float(np.mean(bundle_values)),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "independent_bundles": len(bundle_values),
        "bootstrap_samples": samples,
    }


def _paired_t_p_value(values: np.ndarray) -> float:
    if len(values) < 2:
        return 1.0
    if float(np.std(values, ddof=1)) < 1e-15:
        return 0.0 if abs(float(np.mean(values))) > 1e-15 else 1.0
    return float(ttest_1samp(values, popmean=0).pvalue)


def _expected_calibration_error(
    confidences: list[float], outcomes: list[float], *, bins: int = 10
) -> float | None:
    if not confidences:
        return None
    confidence = np.asarray(confidences)
    outcome = np.asarray(outcomes)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if np.any(selected):
            total += float(np.mean(selected)) * abs(
                float(np.mean(confidence[selected])) - float(np.mean(outcome[selected]))
            )
    return total


def _reliability_curve(
    confidences: list[float], outcomes: list[float], *, bins: int = 10
) -> list[dict[str, float | int]]:
    confidence = np.asarray(confidences, dtype=float)
    outcome = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for index in range(bins):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        count = int(np.sum(selected))
        if not count:
            continue
        observed = float(np.mean(outcome[selected]))
        standard_error = float(np.sqrt(observed * (1 - observed) / count))
        rows.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_confidence": float(np.mean(confidence[selected])),
                "observed_rate": observed,
                "standard_error": standard_error,
                "lower_95": max(0.0, observed - 1.96 * standard_error),
                "upper_95": min(1.0, observed + 1.96 * standard_error),
            }
        )
    return rows


def _classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {actual: {predicted: 0 for predicted in FAMILIES} for actual in FAMILIES}
    log_losses = []
    probabilities: list[list[float]] = []
    correct = 0
    for row in rows:
        values = row["diagnosis_probabilities"]
        predicted = max(FAMILIES, key=lambda family: float(values.get(family, 0.0)))
        actual = str(row["family"])
        confusion[actual][predicted] += 1
        correct += predicted == actual
        log_losses.append(-np.log(np.clip(float(values.get(actual, 0.0)), 1e-15, 1.0)))
        probabilities.append([float(values.get(family, 0.0)) for family in FAMILIES])
    f1_scores = []
    for family in FAMILIES:
        true_positive = confusion[family][family]
        false_positive = sum(confusion[actual][family] for actual in FAMILIES if actual != family)
        false_negative = sum(
            confusion[family][predicted] for predicted in FAMILIES if predicted != family
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "accuracy": None if not rows else correct / len(rows),
        "macro_f1": None if not rows else float(np.mean(f1_scores)),
        "confusion_matrix_actual_by_predicted": confusion,
        "clipped_log_loss": None if not log_losses else float(np.mean(log_losses)),
        "diagnosis_probability_variance": (
            None if not probabilities else float(np.mean(np.var(probabilities, axis=0)))
        ),
    }


def _action_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter(str(row["selected_action"]) for row in rows)
    total = sum(action_counts.values())
    probabilities = np.asarray(list(action_counts.values()), dtype=float) / max(1, total)
    entropy = float(-np.sum(probabilities * np.log(probabilities))) if total else None

    def cross_tab(field: str) -> dict[str, dict[str, int]]:
        return {
            level: dict(
                sorted(
                    Counter(
                        str(row["selected_action"]) for row in rows if str(row[field]) == level
                    ).items()
                )
            )
            for level in sorted({str(row[field]) for row in rows})
        }

    by_world: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_world[str(row["world_id"])].append(str(row["selected_action"]))
    repeated = [values for values in by_world.values() if len(values) > 1]
    agreements = [max(Counter(values).values()) / len(values) for values in repeated]
    return {
        "action_counts": dict(sorted(action_counts.items())),
        "action_entropy_nats": entropy,
        "action_counts_by_family": cross_tab("family"),
        "action_counts_by_stratum": cross_tab("stratum"),
        "action_counts_by_oracle_optimal_action": cross_tab("oracle_optimal_action"),
        "repeated_worlds": len(repeated),
        "mean_within_world_agreement": None if not agreements else float(np.mean(agreements)),
        "unanimous_world_rate": (
            None if not repeated else float(np.mean([len(set(values)) == 1 for values in repeated]))
        ),
    }


def _risk_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = sorted(
        (row for row in rows if row.get("confidence") is not None),
        key=lambda row: (-float(row["confidence"]), str(row["world_id"])),
    )
    result = []
    for coverage in (0.25, 0.50, 0.75, 1.0):
        retained = max(1, int(np.ceil(coverage * len(scored)))) if scored else 0
        selected = scored[:retained]
        result.append(
            {
                "target_coverage": coverage,
                "retained_trials": retained,
                "realized_coverage_of_all_trials": retained / len(rows),
                "mean_full_information_regret": (
                    None
                    if not selected
                    else float(np.mean([row["full_information_regret"] for row in selected]))
                ),
                "epsilon_optimal_rate": (
                    None
                    if not selected
                    else float(np.mean([bool(row["epsilon_optimal"]) for row in selected]))
                ),
            }
        )
    return result


def _mean_vector(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    names = sorted(
        {str(name) for row in rows if isinstance(row.get(key), dict) for name in row[key]}
    )
    return {
        name: float(np.mean([float(row[key][name]) for row in rows if name in row[key]]))
        for name in names
    }


def pairwise_model_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model_bundle: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_model_bundle[str(row["model_id"])][str(row["bundle_id"])].append(
            float(row["decision_loss"])
        )
    raw: dict[str, float] = {}
    estimates: dict[str, float] = {}
    for left, right in combinations(sorted(by_model_bundle), 2):
        shared = sorted(set(by_model_bundle[left]) & set(by_model_bundle[right]))
        differences = np.asarray(
            [
                np.mean(by_model_bundle[left][bundle]) - np.mean(by_model_bundle[right][bundle])
                for bundle in shared
            ]
        )
        key = f"{left} minus {right}"
        estimates[key] = float(np.mean(differences))
        raw[key] = _paired_t_p_value(differences)
    adjusted = holm_adjust(raw)
    return {
        key: {
            "estimate": estimates[key],
            "raw_p_value": raw[key],
            "holm_adjusted_p_value": adjusted[key],
        }
        for key in sorted(raw)
    }


def hierarchical_variance_decomposition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Transparent nested variance decomposition plus preregistered fixed contrasts.

    The final archive may additionally fit a Bayesian mixed model, but this
    deterministic method-of-moments result is always reported and auditable.
    """

    factors = ("model_id", "model_family", "family", "stratum")
    fixed = {
        factor: {
            level: float(
                np.mean([float(row["decision_loss"]) for row in rows if str(row[factor]) == level])
            )
            for level in sorted({str(row[factor]) for row in rows})
        }
        for factor in factors
    }
    grand = float(np.mean([float(row["decision_loss"]) for row in rows]))
    centered = []
    for row in rows:
        expected = grand + sum(fixed[factor][str(row[factor])] - grand for factor in factors)
        centered.append((row, float(row["decision_loss"]) - expected))

    by_world_model: dict[tuple[str, str], list[float]] = defaultdict(list)
    world_bundle: dict[tuple[str, str], tuple[str, str]] = {}
    for row, residual in centered:
        world_key = (str(row["model_id"]), str(row["world_id"]))
        by_world_model[world_key].append(residual)
        world_bundle[world_key] = (str(row["model_id"]), str(row["bundle_id"]))
    world_means = {key: float(np.mean(values)) for key, values in by_world_model.items()}
    by_bundle_model: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key, value in world_means.items():
        by_bundle_model[world_bundle[key]].append(value)
    bundle_means = {key: float(np.mean(values)) for key, values in by_bundle_model.items()}
    replicate_residuals = [
        value - world_means[key] for key, values in by_world_model.items() for value in values
    ]
    world_residuals = [
        value - bundle_means[world_bundle[key]] for key, value in world_means.items()
    ]
    return {
        "method": "fixed marginal means plus nested method-of-moments residual decomposition",
        "fixed_marginal_means": fixed,
        "variance_components": {
            "bundle_within_model": float(np.var(list(bundle_means.values()), ddof=1)),
            "world_within_bundle_model": float(np.var(world_residuals, ddof=1)),
            "replicate_within_world_model": float(np.var(replicate_residuals, ddof=1)),
        },
        "interactions_preregistered": [
            "model:family",
            "model:stratum",
            "model_family:stratum",
        ],
    }


def fit_hierarchical_mixed_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit the preregistered Gaussian mixed model with nested sampling effects."""

    frame = pd.DataFrame(rows)
    frame["bundle_group"] = frame["bundle_id"].astype(str)
    frame["world_group"] = frame["bundle_id"].astype(str) + ":" + frame["world_id"].astype(str)
    frame["replicate_group"] = frame["world_group"] + ":" + frame["replicate_id"].astype(str)
    formula = "decision_loss ~ C(model_id) * C(family) + C(model_id) * C(stratum)"
    model = smf.mixedlm(
        formula,
        frame,
        groups=frame["bundle_group"],
        re_formula="1",
        vc_formula={
            "world_within_bundle": "0 + C(world_group)",
            "replicate_within_world": "0 + C(replicate_group)",
        },
    )
    result = model.fit(reml=True, method=["lbfgs", "powell"], maxiter=1_000)
    fixed = {
        str(name): {
            "estimate": float(value),
            "standard_error": float(result.bse_fe[name]),
        }
        for name, value in result.fe_params.items()
    }
    variance_names = tuple(model.exog_vc.names)
    variance_values = [float(value) for value in result.vcomp]
    return {
        "method": "Gaussian linear mixed model fit by REML",
        "formula": formula,
        "converged": bool(result.converged),
        "fixed_effects": fixed,
        "variance_components": {
            "bundle": float(result.cov_re.iloc[0, 0]),
            **dict(zip(variance_names, variance_values, strict=True)),
            "residual": float(result.scale),
        },
        "provider_family_marginal_means": {
            family: float(frame.loc[frame["model_family"] == family, "decision_loss"].mean())
            for family in sorted(frame["model_family"].unique())
        },
        "identifiability_note": (
            "Provider family is reported as a marginal summary because Anthropic and Google "
            "each contribute one model; it is not interpreted as a provider-family causal effect."
        ),
    }


def analyze_comparative_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = {
        "model_id",
        "model_family",
        "bundle_id",
        "world_id",
        "replicate_id",
        "family",
        "stratum",
        "model_utility",
        "public_policy_utility",
        "oracle_utility",
        "validation_status",
    }
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("comparative rows do not satisfy the Janus v2 analysis contract")
    enriched = []
    for row in rows:
        copy = dict(row)
        copy["decision_loss"] = float(row["public_policy_utility"]) - float(row["model_utility"])
        copy["full_information_regret"] = float(row["oracle_utility"]) - float(row["model_utility"])
        copy["information_loss"] = float(row["oracle_utility"]) - float(
            row["public_policy_utility"]
        )
        copy["utility_difference_vs_public_policy"] = -copy["decision_loss"]
        action_range = float(row["oracle_utility"]) - min(
            float(value) for value in row["constant_action_utilities"].values()
        )
        copy["normalized_full_information_regret"] = (
            0.0 if action_range <= 1e-15 else copy["full_information_regret"] / action_range
        )
        enriched.append(copy)
    models = sorted({str(row["model_id"]) for row in enriched})
    summaries: dict[str, Any] = {}
    for index, model in enumerate(models):
        selected = [row for row in enriched if row["model_id"] == model]
        calibration_rows = [
            row
            for row in selected
            if isinstance(row.get("diagnosis_probabilities"), dict)
            and row.get("confidence") is not None
            and row.get("epsilon_optimal") is not None
        ]
        diagnosis_brier = [
            sum(
                (
                    float(row["diagnosis_probabilities"].get(family, 0.0))
                    - float(family == row["family"])
                )
                ** 2
                for family in FAMILIES
            )
            for row in calibration_rows
        ]
        action_brier = [
            (float(row["confidence"]) - float(row["epsilon_optimal"])) ** 2
            for row in calibration_rows
        ]
        diagnosis_confidence = [
            max(float(value) for value in row["diagnosis_probabilities"].values())
            for row in calibration_rows
        ]
        diagnosis_correct = [
            float(
                max(
                    row["diagnosis_probabilities"],
                    key=row["diagnosis_probabilities"].get,
                )
                == row["family"]
            )
            for row in calibration_rows
        ]
        regrets = sorted(float(row["full_information_regret"]) for row in selected)
        cutoff = max(1, int(np.ceil(0.1 * len(regrets))))
        diagnosis_correct = [float(value) for value in diagnosis_correct]
        confidence_values = [float(row["confidence"]) for row in calibration_rows]
        epsilon_values = [float(row["epsilon_optimal"]) for row in calibration_rows]
        confidence_correlation = (
            None
            if len(confidence_values) < 2
            or np.std(confidence_values) < 1e-15
            or np.std(epsilon_values) < 1e-15
            else float(np.corrcoef(confidence_values, epsilon_values)[0, 1])
        )
        summaries[model] = {
            "paired_utility_difference_vs_public_policy": paired_bundle_interval(
                selected, "utility_difference_vs_public_policy", seed=20260021 + index
            ),
            "paired_decision_loss": paired_bundle_interval(
                selected, "decision_loss", seed=20261021 + index
            ),
            "full_information_regret": paired_bundle_interval(
                selected, "full_information_regret", seed=20261121 + index
            ),
            "information_loss": paired_bundle_interval(
                selected, "information_loss", seed=20261221 + index
            ),
            "worst_decile_full_information_regret": float(np.mean(regrets[-cutoff:])),
            "worst_decile_normalized_regret": float(
                np.mean(
                    sorted(float(row["normalized_full_information_regret"]) for row in selected)[
                        -cutoff:
                    ]
                )
            ),
            "maximum_full_information_regret": float(max(regrets)),
            "mean_normalized_full_information_regret": float(
                np.mean([row["normalized_full_information_regret"] for row in selected])
            ),
            "epsilon_optimal_action_rate": float(
                np.mean([bool(row["epsilon_optimal"]) for row in selected])
            ),
            "paired_differences_vs_secondary_policies": {
                name: paired_bundle_interval(
                    [
                        {
                            "bundle_id": row["bundle_id"],
                            "difference": float(row["model_utility"]) - float(row[utility_key]),
                        }
                        for row in selected
                    ],
                    "difference",
                    seed=20263000 + index * 10 + offset,
                )
                for offset, (name, utility_key) in enumerate(
                    (
                        ("do_nothing", "do_nothing_utility"),
                        ("heuristic", "heuristic_utility"),
                        ("uniform_random", "random_utility"),
                    )
                )
            },
            "completion_rate": float(
                np.mean([row["validation_status"] == "valid" for row in selected])
            ),
            "trial_count": len(selected),
            "status_counts": dict(Counter(str(row["validation_status"]) for row in selected)),
            "status_rates": {
                status: count / len(selected)
                for status, count in sorted(
                    Counter(str(row["validation_status"]) for row in selected).items()
                )
            },
            "calibration": {
                "valid_scored_trials": len(calibration_rows),
                "diagnosis_brier": (
                    None if not diagnosis_brier else float(np.mean(diagnosis_brier))
                ),
                "action_confidence_brier": (
                    None if not action_brier else float(np.mean(action_brier))
                ),
                "diagnosis_ece_10_bin": _expected_calibration_error(
                    diagnosis_confidence, diagnosis_correct
                ),
                "action_confidence_ece_10_bin": _expected_calibration_error(
                    confidence_values,
                    epsilon_values,
                ),
                "diagnosis_reliability": _reliability_curve(
                    diagnosis_confidence, diagnosis_correct
                ),
                "action_confidence_reliability": _reliability_curve(
                    confidence_values, epsilon_values
                ),
                "confidence_epsilon_optimality_correlation": confidence_correlation,
                **_classification_metrics(calibration_rows),
            },
            "behavior": _action_behavior(selected),
            "risk_coverage": _risk_coverage(selected),
            "action_conditioned": {
                action: {
                    "trials": len(action_rows),
                    "mean_decision_loss": float(
                        np.mean([float(row["decision_loss"]) for row in action_rows])
                    ),
                }
                for action in sorted(
                    {
                        str(row["selected_action"])
                        for row in selected
                        if row.get("selected_action") is not None
                    }
                )
                if (
                    action_rows := [
                        row for row in selected if str(row.get("selected_action")) == action
                    ]
                )
            },
            "oracle_optimal_action_conditioned": {
                action: {
                    "trials": len(action_rows),
                    "mean_decision_loss": float(
                        np.mean([float(row["decision_loss"]) for row in action_rows])
                    ),
                    "epsilon_optimal_rate": float(
                        np.mean([bool(row["epsilon_optimal"]) for row in action_rows])
                    ),
                }
                for action in sorted({str(row["oracle_optimal_action"]) for row in selected})
                if (
                    action_rows := [
                        row for row in selected if str(row["oracle_optimal_action"]) == action
                    ]
                )
            },
            "operations": {
                "actual_cost_usd": float(
                    np.sum([float(row.get("actual_cost_usd", 0)) for row in selected])
                ),
                "mean_latency_ms": float(
                    np.mean(
                        [
                            float(row["latency_ms"])
                            for row in selected
                            if row.get("latency_ms") is not None
                        ]
                        or [0.0]
                    )
                ),
                "tokens": {
                    key: int(np.sum([int(row.get("usage", {}).get(key, 0)) for row in selected]))
                    for key in (
                        "input_tokens",
                        "cached_input_tokens",
                        "cache_write_tokens",
                        "output_tokens",
                        "reasoning_tokens",
                    )
                },
            },
            "economic_outcome_means": {
                key: float(
                    np.mean(
                        [
                            float(row["economic_outcome"][key])
                            for row in selected
                            if isinstance(row.get("economic_outcome"), dict)
                            and key in row["economic_outcome"]
                        ]
                    )
                )
                for key in sorted(
                    {
                        str(key)
                        for row in selected
                        if isinstance(row.get("economic_outcome"), dict)
                        for key in row["economic_outcome"]
                    }
                )
            },
        }
    reference_rows = [row for row in enriched if row["model_id"] == models[0]]
    secondary_references: dict[str, Any] = {}
    if all("heuristic_utility" in row for row in reference_rows):
        secondary_references = {
            "heuristic_mean_utility": float(
                np.mean([float(row["heuristic_utility"]) for row in reference_rows])
            ),
            "uniform_random_mean_utility": float(
                np.mean([float(row["random_utility"]) for row in reference_rows])
            ),
            "do_nothing_mean_utility": float(
                np.mean([float(row["do_nothing_utility"]) for row in reference_rows])
            ),
            "constant_action_mean_utilities": {
                action: float(
                    np.mean(
                        [float(row["constant_action_utilities"][action]) for row in reference_rows]
                    )
                )
                for action in sorted(reference_rows[0]["constant_action_utilities"])
            },
            "economic_outcome_means": {
                "frozen_public_information_policy": _mean_vector(
                    reference_rows, "public_policy_economic_outcome"
                ),
                "heuristic": _mean_vector(reference_rows, "heuristic_economic_outcome"),
                "do_nothing": _mean_vector(reference_rows, "do_nothing_economic_outcome"),
                "uniform_random": _mean_vector(reference_rows, "uniform_random_economic_outcome"),
                "constant_actions": {
                    action: {
                        name: float(
                            np.mean(
                                [
                                    row["constant_action_economic_outcomes"][action][name]
                                    for row in reference_rows
                                ]
                            )
                        )
                        for name in sorted(
                            reference_rows[0]["constant_action_economic_outcomes"][action]
                        )
                    }
                    for action in sorted(reference_rows[0]["constant_action_economic_outcomes"])
                },
            },
            "status": "secondary_descriptive_references_not_primary_comparators",
        }
    objective_sensitivity: dict[str, Any] = {}
    if all(isinstance(row.get("objective_comparisons"), dict) for row in enriched):
        objective_names = sorted(enriched[0]["objective_comparisons"])
        for objective_index, objective in enumerate(objective_names):
            objective_sensitivity[objective] = {}
            for model_index, model in enumerate(models):
                selected = [row for row in enriched if row["model_id"] == model]
                objective_rows = []
                for row in selected:
                    comparison = row["objective_comparisons"][objective]
                    objective_rows.append(
                        {
                            "bundle_id": row["bundle_id"],
                            "objective_decision_loss": float(comparison["public_policy_utility"])
                            - float(comparison["model_utility"]),
                        }
                    )
                objective_sensitivity[objective][model] = paired_bundle_interval(
                    objective_rows,
                    "objective_decision_loss",
                    seed=20262000 + 10 * objective_index + model_index,
                )
            objective_sensitivity[objective]["direction_summary"] = {
                "models_better_than_objective_specific_public_policy": sorted(
                    model
                    for model in models
                    if float(objective_sensitivity[objective][model]["estimate"]) < 0
                ),
                "estimand_sign": "decision_loss; negative favors model",
            }
        if "primary" in objective_sensitivity:
            primary_signs = {
                model: np.sign(float(objective_sensitivity["primary"][model]["estimate"]))
                for model in models
            }
            for objective, values in objective_sensitivity.items():
                if objective == "primary":
                    continue
                values["sign_reversals_vs_primary"] = sorted(
                    model
                    for model in models
                    if np.sign(float(values[model]["estimate"])) != primary_signs[model]
                )
    return {
        "schema_version": "janus-v2-comparative-analysis-v1",
        "primary_estimand": "model_utility minus frozen_public_policy_utility",
        "intention_to_evaluate": True,
        "normalization": (
            "full-information regret divided by the within-world oracle-to-worst-action "
            "utility range; zero when the range is numerically zero"
        ),
        "population_scope": {
            "fixed_sample": "descriptive and paired estimates apply to the sealed hidden-v2 sample",
            "procedural_distribution": (
                "generalization requires exchangeable bundles from the declared generator and "
                "correct hierarchical-model specification"
            ),
        },
        "model_summaries": summaries,
        "pairwise_model_comparisons": pairwise_model_comparisons(enriched),
        "secondary_policy_references": secondary_references,
        "objective_specific_sensitivity": objective_sensitivity,
        "hierarchical_mixed_model": fit_hierarchical_mixed_model(enriched),
        "deterministic_variance_crosscheck": hierarchical_variance_decomposition(enriched),
    }
