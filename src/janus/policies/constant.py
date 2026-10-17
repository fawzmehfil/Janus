from __future__ import annotations

from dataclasses import dataclass

from janus.interventions.actions import ActionID


@dataclass(frozen=True, slots=True)
class ConstantPolicy:
    action: ActionID

    def act(self, _: bytes) -> ActionID:
        return self.action
