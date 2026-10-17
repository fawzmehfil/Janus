from __future__ import annotations

from dataclasses import dataclass

from janus.core.randomness import NamedRNG
from janus.interventions.actions import ActionID


@dataclass(frozen=True, slots=True)
class UniformRandomPolicy:
    root_seed: int

    def act(self, _: bytes, *, trial_id: str) -> ActionID:
        generator = NamedRNG(self.root_seed).generator("random_policy", trial_id)
        return tuple(ActionID)[int(generator.integers(0, len(ActionID)))]
