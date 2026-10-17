"""Preflight estimation, hard ceilings, and usage reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import Field

from janus.core.config import StrictFrozenModel
from janus.models.base import UsageRecord


class PricingConfig(StrictFrozenModel):
    provider: str
    model_id: str
    effective_date: str
    source_url: str
    service_tier: str
    input_per_million_usd: float = Field(ge=0)
    cached_input_per_million_usd: float = Field(ge=0)
    cache_write_per_million_usd: float = Field(ge=0)
    output_per_million_usd: float = Field(ge=0)
    reasoning_per_million_usd: float = Field(ge=0)

    def cost(self, usage: UsageRecord) -> float:
        return (
            usage.input_tokens * self.input_per_million_usd
            + usage.cached_input_tokens * self.cached_input_per_million_usd
            + usage.cache_write_tokens * self.cache_write_per_million_usd
            + usage.output_tokens * self.output_per_million_usd
            + usage.reasoning_tokens * self.reasoning_per_million_usd
        ) / 1_000_000


class BudgetExceeded(RuntimeError):
    pass


@dataclass(slots=True)
class BudgetLedger:
    pricing: PricingConfig
    total_ceiling_usd: float
    stage_ceilings_usd: dict[str, float]
    total_spent_usd: float = 0.0
    stage_spent_usd: dict[str, float] = field(default_factory=dict)

    def preflight(self, stage: str, estimated_usage: UsageRecord) -> float:
        estimate = self.pricing.cost(estimated_usage)
        stage_ceiling = self.stage_ceilings_usd.get(stage, self.total_ceiling_usd)
        stage_spent = self.stage_spent_usd.get(stage, 0.0)
        if stage_spent + estimate > stage_ceiling + 1e-12:
            raise BudgetExceeded(f"stage {stage!r} would exceed ${stage_ceiling:.6f} ceiling")
        if self.total_spent_usd + estimate > self.total_ceiling_usd + 1e-12:
            raise BudgetExceeded(
                f"next request would exceed ${self.total_ceiling_usd:.6f} total ceiling"
            )
        return estimate

    def charge(self, stage: str, usage: UsageRecord) -> float:
        actual = self.pricing.cost(usage)
        self.total_spent_usd += actual
        self.stage_spent_usd[stage] = self.stage_spent_usd.get(stage, 0.0) + actual
        return actual


def conservative_usage_estimate(input_bytes: int, maximum_output_tokens: int) -> UsageRecord:
    estimated_input = (input_bytes + 3) // 4
    return UsageRecord(
        input_tokens=estimated_input,
        output_tokens=maximum_output_tokens,
        reasoning_tokens=maximum_output_tokens,
    )
