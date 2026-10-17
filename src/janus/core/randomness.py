"""Named streams and indexed disturbance tapes.

The indexed tape makes a draw a pure function of the root seed and semantic key.
Changing one action's control flow therefore cannot shift later exogenous draws.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")

REQUIRED_STREAMS: Final[tuple[str, ...]] = (
    "scenario_parameters",
    "affected_participants",
    "pre_decision_disturbances",
    "post_decision_tapes",
    "tie_breaking",
    "audit_outcomes",
    "random_policy",
    "model_replicates",
)


def _semantic_entropy(root_seed: int, parts: tuple[object, ...]) -> list[int]:
    encoded = json.dumps(parts, ensure_ascii=True, separators=(",", ":"), default=str).encode()
    digest = hashlib.sha256(encoded).digest()
    return [root_seed, *np.frombuffer(digest[:16], dtype=np.uint32).astype(int).tolist()]


@dataclass(frozen=True, slots=True)
class NamedRNG:
    """Factory for independent NumPy generators identified by stable names."""

    root_seed: int

    def generator(self, stream: str, *indices: object) -> np.random.Generator:
        if stream not in REQUIRED_STREAMS:
            raise KeyError(f"unknown RNG stream: {stream}")
        seed = np.random.SeedSequence(_semantic_entropy(self.root_seed, (stream, *indices)))
        return np.random.default_rng(seed)


@dataclass(frozen=True, slots=True)
class DisturbanceTape:
    """Immutable, random-access exogenous disturbance tape."""

    root_seed: int
    tape_id: str

    def _generator(self, kind: str, key: tuple[object, ...]) -> np.random.Generator:
        seed = np.random.SeedSequence(
            _semantic_entropy(self.root_seed, ("post_decision_tapes", self.tape_id, kind, *key))
        )
        return np.random.default_rng(seed)

    def uniform(self, *key: object, low: float = 0.0, high: float = 1.0) -> float:
        return float(self._generator("uniform", key).uniform(low, high))

    def normal(self, *key: object, mean: float = 0.0, std: float = 1.0) -> float:
        if std < 0:
            raise ValueError("std must be nonnegative")
        return float(self._generator("normal", key).normal(mean, std))

    def bernoulli(self, probability: float, *key: object) -> bool:
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        return self.uniform(*key) < probability

    def permutation(self, values: tuple[T, ...], *key: object) -> tuple[T, ...]:
        if not values:
            return ()
        indices: NDArray[np.int64] = self._generator("permutation", key).permutation(len(values))
        return tuple(values[int(index)] for index in indices)

    def tie_priority(self, order_ids: tuple[str, ...], period: int) -> dict[str, int]:
        """Assign an indexed priority that is stable when other orders appear or disappear."""

        priorities: dict[str, int] = {}
        for order_id in order_ids:
            generator = self._generator("tie_priority", (period, order_id))
            priorities[order_id] = int(generator.integers(0, 2**63 - 1))
        return priorities
