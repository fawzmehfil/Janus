"""Observation-only ridge action-value baseline with portable coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from janus.evaluation.observations import PublicObservationHistory
from janus.interventions.actions import ActionID
from janus.scenarios.generator import public_summary


def features_from_serialized(observation: bytes) -> np.ndarray:
    parsed = PublicObservationHistory.model_validate_json(observation)
    return np.asarray(public_summary(parsed), dtype=float)


@dataclass(frozen=True, slots=True)
class RidgeActionValuePolicy:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]
    regularization: float

    @classmethod
    def fit(
        cls,
        observations: tuple[bytes, ...],
        targets: np.ndarray,
        *,
        regularization: float = 10.0,
    ) -> RidgeActionValuePolicy:
        if targets.shape != (len(observations), len(ActionID)):
            raise ValueError("targets must have shape (worlds, actions)")
        matrix = np.vstack([features_from_serialized(item) for item in observations])
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-12] = 1.0
        standardized = (matrix - mean) / scale
        target_mean = targets.mean(axis=0)
        centered_targets = targets - target_mean
        gram = standardized.T @ standardized
        penalty = regularization * np.eye(gram.shape[0])
        coefficients = np.linalg.solve(gram + penalty, standardized.T @ centered_targets)
        return cls(
            feature_mean=tuple(float(value) for value in mean),
            feature_scale=tuple(float(value) for value in scale),
            coefficients=tuple(
                tuple(float(value) for value in coefficients[:, action_index])
                for action_index in range(len(ActionID))
            ),
            intercepts=tuple(float(value) for value in target_mean),
            regularization=regularization,
        )

    def predict_values(self, observation: bytes) -> dict[ActionID, float]:
        features = features_from_serialized(observation)
        standardized = (features - np.asarray(self.feature_mean)) / np.asarray(self.feature_scale)
        return {
            action: self.intercepts[index]
            + float(standardized @ np.asarray(self.coefficients[index]))
            for index, action in enumerate(ActionID)
        }

    def act(self, observation: bytes) -> ActionID:
        values = self.predict_values(observation)
        return max(ActionID, key=lambda action: (values[action], action.value))
