"""Bundle-clustered bootstrap and pilot precision calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    bootstrap_samples: int


def cluster_bootstrap_mean(
    bundle_values: np.ndarray,
    *,
    root_seed: int,
    bootstrap_samples: int = 2_000,
) -> BootstrapInterval:
    """Resample rows (bundles), preserving all nested worlds in each row."""

    if bundle_values.ndim != 2 or bundle_values.shape[0] < 2:
        raise ValueError("bundle_values must be a 2D array with at least two bundles")
    generator = np.random.default_rng(root_seed)
    estimates = np.empty(bootstrap_samples)
    for index in range(bootstrap_samples):
        rows = generator.integers(0, bundle_values.shape[0], size=bundle_values.shape[0])
        estimates[index] = bundle_values[rows].mean()
    return BootstrapInterval(
        estimate=float(bundle_values.mean()),
        lower=float(np.quantile(estimates, 0.025)),
        upper=float(np.quantile(estimates, 0.975)),
        bootstrap_samples=bootstrap_samples,
    )


def projected_half_width(bundle_effects: np.ndarray, target_bundles: int = 25) -> float:
    if bundle_effects.ndim != 1 or bundle_effects.size < 2:
        raise ValueError("bundle effects require at least two observations")
    return float(1.96 * bundle_effects.std(ddof=1) / np.sqrt(target_bundles))
