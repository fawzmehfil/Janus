"""Canonical serialization and content-addressed replay records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from janus.core.clearing import Ask, Bid
from janus.core.config import AccountingConfig, MarketConfig
from janus.core.engine import PeriodResult, run_period
from janus.core.ledger import PolicyFlows
from janus.core.state import BuyerState, MarketState, SellerState


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def state_from_dict(data: Mapping[str, Any]) -> MarketState:
    return MarketState(
        period=int(data["period"]),
        buyers=tuple(BuyerState(**buyer) for buyer in data["buyers"]),
        sellers=tuple(
            SellerState(**{**seller, "marginal_costs": tuple(seller["marginal_costs"])})
            for seller in data["sellers"]
        ),
        government_balance=float(data["government_balance"]),
        last_price=None if data["last_price"] is None else float(data["last_price"]),
        recent_transaction_prices=tuple(
            float(value) for value in data.get("recent_transaction_prices", ())
        ),
        cumulative_real_cost=float(data["cumulative_real_cost"]),
    )


@dataclass(frozen=True, slots=True)
class PeriodReplay:
    state: dict[str, Any]
    bids: tuple[dict[str, Any], ...]
    asks: tuple[dict[str, Any], ...]
    tie_priority: dict[str, int]
    market_config: dict[str, Any]
    accounting_config: dict[str, Any]
    policy: dict[str, Any]
    price_cap: float | None
    expected_result_hash: str

    @classmethod
    def capture(
        cls,
        result: PeriodResult,
        *,
        bids: tuple[Bid, ...],
        asks: tuple[Ask, ...],
        tie_priority: Mapping[str, int],
        market_config: MarketConfig,
        accounting_config: AccountingConfig,
        policy: PolicyFlows,
        price_cap: float | None = None,
    ) -> PeriodReplay:
        return cls(
            state=asdict(result.previous_state),
            bids=tuple(asdict(bid) for bid in bids),
            asks=tuple(asdict(ask) for ask in asks),
            tie_priority=dict(tie_priority),
            market_config=market_config.model_dump(mode="json"),
            accounting_config=accounting_config.model_dump(mode="json"),
            policy=asdict(policy),
            price_cap=price_cap,
            expected_result_hash=content_hash(asdict(result)),
        )

    def replay(self) -> PeriodResult:
        policy_data = dict(self.policy)
        policy_data["seller_fines"] = tuple(tuple(item) for item in policy_data["seller_fines"])
        result = run_period(
            state_from_dict(self.state),
            tuple(Bid(**item) for item in self.bids),
            tuple(Ask(**item) for item in self.asks),
            tie_priority=self.tie_priority,
            market_config=MarketConfig.model_validate(self.market_config),
            accounting_config=AccountingConfig.model_validate(self.accounting_config),
            policy=PolicyFlows(**policy_data),
            price_cap=self.price_cap,
        )
        if content_hash(asdict(result)) != self.expected_result_hash:
            raise AssertionError("replayed period does not match the captured result")
        return result
