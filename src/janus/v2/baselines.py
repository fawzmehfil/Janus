"""Strong public-information baseline selection and freezing for Janus v2."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from janus.interventions.actions import ActionID
from janus.policies.partial_information import features_from_serialized


class FittedRegressor(Protocol):
    def predict(self, values: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class BaselineDataset:
    observations: tuple[bytes, ...]
    action_values: np.ndarray
    bundle_ids: tuple[str, ...]
    world_ids: tuple[str, ...]

    def validate(self) -> None:
        rows = len(self.observations)
        if self.action_values.shape != (rows, len(ActionID)):
            raise ValueError("action_values must have shape (worlds, actions)")
        if len(self.bundle_ids) != rows or len(self.world_ids) != rows:
            raise ValueError("every row must have a bundle_id and world_id")
        counts = {bundle: self.bundle_ids.count(bundle) for bundle in set(self.bundle_ids)}
        if len(set(counts.values())) != 1:
            raise ValueError("bundle-preserving CV requires equal-sized bundles")


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    name: str
    parameters: tuple[tuple[str, float | int], ...]

    @property
    def identifier(self) -> str:
        values = ",".join(f"{key}={value}" for key, value in self.parameters)
        return f"{self.name}:{values}"


CANDIDATES = (
    CandidateSpec("ridge", (("alpha", 0.1),)),
    CandidateSpec("ridge", (("alpha", 1.0),)),
    CandidateSpec("ridge", (("alpha", 10.0),)),
    CandidateSpec("ridge", (("alpha", 100.0),)),
    CandidateSpec(
        "hist_gradient_boosting",
        (("learning_rate", 0.05), ("max_depth", 2), ("max_iter", 120)),
    ),
    CandidateSpec(
        "hist_gradient_boosting",
        (("learning_rate", 0.08), ("max_depth", 3), ("max_iter", 120)),
    ),
    CandidateSpec(
        "polynomial_ridge",
        (("alpha", 10.0), ("degree", 2)),
    ),
    CandidateSpec(
        "extra_trees",
        (("max_depth", 4), ("min_samples_leaf", 3), ("n_estimators", 200)),
    ),
)


def _parameters(spec: CandidateSpec) -> dict[str, float | int]:
    return dict(spec.parameters)


def _build(spec: CandidateSpec) -> FittedRegressor:
    parameters = _parameters(spec)
    if spec.name == "ridge":
        return cast(
            FittedRegressor,
            make_pipeline(StandardScaler(), Ridge(alpha=float(parameters["alpha"]))),
        )
    if spec.name == "polynomial_ridge":
        return cast(
            FittedRegressor,
            make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=int(parameters["degree"]), include_bias=False),
                Ridge(alpha=float(parameters["alpha"])),
            ),
        )
    if spec.name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            learning_rate=float(parameters["learning_rate"]),
            max_depth=int(parameters["max_depth"]),
            max_iter=int(parameters["max_iter"]),
            l2_regularization=1.0,
            random_state=2026,
        )
        return cast(FittedRegressor, MultiOutputRegressor(estimator))
    if spec.name == "extra_trees":
        return cast(
            FittedRegressor,
            ExtraTreesRegressor(
                n_estimators=int(parameters["n_estimators"]),
                max_depth=int(parameters["max_depth"]),
                min_samples_leaf=int(parameters["min_samples_leaf"]),
                random_state=2026,
                n_jobs=1,
            ),
        )
    raise ValueError(f"unknown baseline class: {spec.name}")


def _features(observations: tuple[bytes, ...]) -> np.ndarray:
    return np.vstack([features_from_serialized(value) for value in observations])


def _decision_utility(predictions: np.ndarray, targets: np.ndarray) -> float:
    choices = np.argmax(predictions, axis=1)
    return float(np.mean(targets[np.arange(len(targets)), choices]))


def _indices(bundle_ids: tuple[str, ...], selected: set[str]) -> np.ndarray:
    return np.asarray([index for index, value in enumerate(bundle_ids) if value in selected])


def _cross_validated_scores(
    x: np.ndarray,
    y: np.ndarray,
    bundle_ids: tuple[str, ...],
    specs: tuple[CandidateSpec, ...],
) -> dict[str, list[float]]:
    bundles = tuple(sorted(set(bundle_ids)))
    scores: dict[str, list[float]] = {spec.identifier: [] for spec in specs}
    for held_out in bundles:
        validation = _indices(bundle_ids, {held_out})
        training = _indices(bundle_ids, set(bundles) - {held_out})
        for spec in specs:
            model = _build(spec)
            model.fit(x[training], y[training])  # type: ignore[attr-defined]
            scores[spec.identifier].append(
                _decision_utility(model.predict(x[validation]), y[validation])
            )
    return scores


def _select(scores: dict[str, list[float]], specs: tuple[CandidateSpec, ...]) -> CandidateSpec:
    return max(
        specs,
        key=lambda spec: (float(np.mean(scores[spec.identifier])), spec.identifier),
    )


def _nested_outer_scores(
    x: np.ndarray,
    y: np.ndarray,
    bundle_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    bundles = tuple(sorted(set(bundle_ids)))
    rows: list[dict[str, Any]] = []
    for held_out in bundles:
        outer_train_bundles = set(bundles) - {held_out}
        outer_train = _indices(bundle_ids, outer_train_bundles)
        outer_test = _indices(bundle_ids, {held_out})
        inner_bundle_ids = tuple(bundle_ids[index] for index in outer_train)
        inner_scores = _cross_validated_scores(
            x[outer_train], y[outer_train], inner_bundle_ids, CANDIDATES
        )
        selected = _select(inner_scores, CANDIDATES)
        model = _build(selected)
        model.fit(x[outer_train], y[outer_train])  # type: ignore[attr-defined]
        rows.append(
            {
                "held_out_bundle": held_out,
                "selected_candidate": selected.identifier,
                "decision_utility": _decision_utility(model.predict(x[outer_test]), y[outer_test]),
                "oracle_utility": float(np.mean(np.max(y[outer_test], axis=1))),
            }
        )
    return rows


def _learning_curve(
    x: np.ndarray,
    y: np.ndarray,
    bundle_ids: tuple[str, ...],
    spec: CandidateSpec,
) -> list[dict[str, float | int]]:
    bundles = tuple(sorted(set(bundle_ids)))
    if len(bundles) < 3:
        return []
    sizes = sorted({max(2, len(bundles) // 3), max(2, 2 * len(bundles) // 3)})
    rows: list[dict[str, float | int]] = []
    for size in sizes:
        fold_scores: list[float] = []
        for offset, held_out in enumerate(bundles):
            pool = tuple(value for value in bundles if value != held_out)
            selected = {pool[(offset + index) % len(pool)] for index in range(min(size, len(pool)))}
            train = _indices(bundle_ids, selected)
            test = _indices(bundle_ids, {held_out})
            model = _build(spec)
            model.fit(x[train], y[train])  # type: ignore[attr-defined]
            fold_scores.append(_decision_utility(model.predict(x[test]), y[test]))
        rows.append(
            {
                "training_bundles": min(size, len(bundles) - 1),
                "mean_decision_utility": float(np.mean(fold_scores)),
                "standard_error": float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores))),
            }
        )
    return rows


def fit_and_freeze_baseline(
    dataset: BaselineDataset,
    output_directory: Path,
    *,
    objective: str,
    training_source_sha256: str,
) -> dict[str, Any]:
    """Select with bundle-preserving nested CV, fit, and freeze a trusted local artifact."""

    dataset.validate()
    x = _features(dataset.observations)
    y = dataset.action_values
    scores = _cross_validated_scores(x, y, dataset.bundle_ids, CANDIDATES)
    selected = _select(scores, CANDIDATES)
    outer = _nested_outer_scores(x, y, dataset.bundle_ids)
    model = _build(selected)
    model.fit(x, y)  # type: ignore[attr-defined]
    learning_curve = _learning_curve(x, y, dataset.bundle_ids, selected)
    curve_improvement = (
        0.0
        if len(learning_curve) < 2
        else float(learning_curve[-1]["mean_decision_utility"])
        - float(learning_curve[0]["mean_decision_utility"])
    )
    stabilization_threshold = 0.0025 * 840.0 * 20
    serialized = pickle.dumps(model, protocol=5)
    output_directory.mkdir(parents=True, exist_ok=True)
    model_path = output_directory / f"{objective}.pkl"
    model_path.write_bytes(serialized)
    manifest = {
        "schema_version": "janus-v2-public-baseline-v1",
        "objective": objective,
        "observation_only": True,
        "selected_candidate": selected.identifier,
        "candidate_cv": {
            key: {
                "bundle_scores": value,
                "mean_decision_utility": float(np.mean(value)),
            }
            for key, value in scores.items()
        },
        "nested_outer_folds": outer,
        "nested_mean_decision_utility": float(
            np.mean([float(row["decision_utility"]) for row in outer])
        ),
        "learning_curve": learning_curve,
        "learning_curve_stability": {
            "last_minus_first_utility": curve_improvement,
            "maximum_stabilized_change": stabilization_threshold,
            "criterion": (
                "absolute change from 3 to 6 training bundles is at most 0.25% "
                "of 20-period analytic reference surplus"
            ),
            "stabilized": abs(curve_improvement) <= stabilization_threshold,
        },
        "training_bundle_count": len(set(dataset.bundle_ids)),
        "training_world_count": len(dataset.world_ids),
        "training_world_ids": list(dataset.world_ids),
        "training_source_sha256": training_source_sha256,
        "feature_contract": "janus.public_summary.v1",
        "action_order": [action.value for action in ActionID],
        "random_seed": 2026,
        "software": {"scikit_learn": sklearn.__version__},
        "model_file": model_path.name,
        "model_sha256": hashlib.sha256(serialized).hexdigest(),
    }
    manifest_path = output_directory / f"{objective}.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


@dataclass(frozen=True, slots=True)
class FrozenPublicPolicy:
    model: FittedRegressor
    manifest: dict[str, Any]

    @classmethod
    def load(cls, manifest_path: Path) -> FrozenPublicPolicy:
        manifest = json.loads(manifest_path.read_text())
        model_path = manifest_path.parent / manifest["model_file"]
        serialized = model_path.read_bytes()
        if hashlib.sha256(serialized).hexdigest() != manifest["model_sha256"]:
            raise ValueError("frozen baseline model hash mismatch")
        # This loads only a trusted local artifact after verifying its frozen hash.
        model = pickle.loads(serialized)
        return cls(model=model, manifest=manifest)

    def predict_values(self, observation: bytes) -> dict[ActionID, float]:
        prediction = np.asarray(
            self.model.predict(features_from_serialized(observation).reshape(1, -1))[0]
        )
        return {action: float(prediction[index]) for index, action in enumerate(ActionID)}

    def act(self, observation: bytes) -> ActionID:
        values = self.predict_values(observation)
        return max(ActionID, key=lambda action: (values[action], action.value))
