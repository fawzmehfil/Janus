"""Functional one-period transition for the Phase 0 market."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from janus.core.clearing import Ask, Bid, ClearingPriceRule, ClearingResult, clear_market
from janus.core.config import AccountingConfig, MarketConfig
from janus.core.ledger import PeriodLedger, PolicyFlows, compute_period_ledger
from janus.core.state import MarketState


@dataclass(frozen=True, slots=True)
class PeriodResult:
    previous_state: MarketState
    next_state: MarketState
    clearing: ClearingResult
    ledger: PeriodLedger


def run_period(
    state: MarketState,
    bids: Sequence[Bid],
    asks: Sequence[Ask],
    *,
    tie_priority: Mapping[str, int],
    market_config: MarketConfig,
    accounting_config: AccountingConfig | None = None,
    policy: PolicyFlows | None = None,
    price_cap: float | None = None,
    price_rule: ClearingPriceRule = "midpoint",
) -> PeriodResult:
    accounting_config = accounting_config or AccountingConfig()
    policy = policy or PolicyFlows()
    buyer_map = {buyer.buyer_id: buyer for buyer in state.buyers}
    seller_map = {seller.seller_id: seller for seller in state.sellers}

    eligible_bids: list[Bid] = []
    for bid in bids:
        buyer = buyer_map.get(bid.buyer_id)
        if buyer is None or not buyer.active:
            raise ValueError(f"unknown or inactive buyer: {bid.buyer_id}")
        gross_bid = bid.amount * (1 + policy.buyer_tax_rate)
        if market_config.allow_credit or buyer.cash >= gross_bid:
            eligible_bids.append(bid)

    ask_count: dict[str, int] = {}
    for ask in asks:
        seller = seller_map.get(ask.seller_id)
        if seller is None or not seller.active:
            raise ValueError(f"unknown or inactive seller: {ask.seller_id}")
        ask_count[ask.seller_id] = ask_count.get(ask.seller_id, 0) + 1
    for seller_id, count in ask_count.items():
        if count > seller_map[seller_id].inventory:
            raise ValueError(f"seller {seller_id} offered more units than inventory")

    clearing = clear_market(
        eligible_bids,
        asks,
        tie_priority=tie_priority,
        price_cap=price_cap,
        price_rule=price_rule,
    )
    sold_by_seller: dict[str, int] = {}
    for trade in clearing.trades:
        sold_by_seller[trade.seller_id] = sold_by_seller.get(trade.seller_id, 0) + 1

    end_inventory = {
        seller.seller_id: seller.inventory
        - sold_by_seller.get(seller.seller_id, 0)
        + seller.capacity
        for seller in state.sellers
    }
    holding_cost = market_config.holding_cost_per_unit * sum(end_inventory.values())
    ledger = compute_period_ledger(
        clearing.trades,
        policy=policy,
        other_real_costs=holding_cost,
        marginal_cost_of_public_funds=accounting_config.marginal_cost_of_public_funds,
    )

    buyers = tuple(
        replace(buyer, cash=buyer.cash + ledger.account_delta(f"buyer:{buyer.buyer_id}"))
        for buyer in state.buyers
    )
    sellers = tuple(
        replace(
            seller,
            cash=seller.cash + ledger.account_delta(f"seller:{seller.seller_id}"),
            inventory=end_inventory[seller.seller_id],
        )
        for seller in state.sellers
    )
    recent_prices = state.recent_transaction_prices
    if clearing.transaction_price is not None:
        recent_prices = (*recent_prices, clearing.transaction_price)[-4:]
    next_state = MarketState(
        period=state.period + 1,
        buyers=buyers,
        sellers=sellers,
        government_balance=state.government_balance + ledger.account_delta("government"),
        last_price=clearing.transaction_price
        if clearing.transaction_price is not None
        else state.last_price,
        recent_transaction_prices=recent_prices,
        cumulative_real_cost=state.cumulative_real_cost + ledger.real_cost,
    )
    return PeriodResult(state, next_state, clearing, ledger)
