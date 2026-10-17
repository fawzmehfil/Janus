"""Deterministic no-network adapter for local harness validation."""

from __future__ import annotations

from dataclasses import dataclass, field

from janus.models.base import ModelRequest, ProviderReply, UsageRecord


@dataclass(slots=True)
class MockAdapter:
    replies: tuple[str | None, ...]
    refusal_indices: frozenset[int] = frozenset()
    timeout_indices: frozenset[int] = frozenset()
    usage: UsageRecord = field(default_factory=UsageRecord)
    call_count: int = 0

    def complete(self, request: ModelRequest) -> ProviderReply:
        index = self.call_count
        self.call_count += 1
        raw = self.replies[min(index, len(self.replies) - 1)] if self.replies else None
        return ProviderReply(
            raw_text=raw,
            returned_model_id="janus-mock-v1",
            provider_request_id=f"mock-{request.logical_trial_id}-{index}",
            usage=self.usage,
            refusal=index in self.refusal_indices,
            timed_out=index in self.timeout_indices,
            timestamp_utc="2026-10-17T00:00:00Z",
        )
