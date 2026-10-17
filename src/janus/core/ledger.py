"""Transfer-correct period welfare accounting and cash-flow ledger."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isclose

from janus.core.clearing import Trade


@dataclass(frozen=True, slots=True)
class PolicyFlows:
    buyer_tax_rate: float = 0.0
    seller_subsidy_per_unit: float = 0.0
    seller_fines: tuple[tuple[str, float], ...] = ()
    administration_cost: float = 0.0
    compliance_cost: float = 0.0
    false_positive_cost: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.buyer_tax_rate,
            self.seller_subsidy_per_unit,
            self.administration_cost,
            self.compliance_cost,
            self.false_positive_cost,
        )
        if any(value < 0 for value in values):
            raise ValueError("policy rates and costs must be nonnegative")
        if len({seller for seller, _ in self.seller_fines}) != len(self.seller_fines):
            raise ValueError("a seller may have at most one fine entry per period")
        if any(amount < 0 for _, amount in self.seller_fines):
            raise ValueError("fines must be nonnegative")


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    account: str
    amount: float
    category: str
    reference: str


@dataclass(frozen=True, slots=True)
class TradeAccounting:
    trade: Trade
    buyer_payment: float
    seller_receipt: float
    tax: float
    subsidy: float
    consumer_surplus: float
    producer_surplus: float


@dataclass(frozen=True, slots=True)
class PeriodLedger:
    trades: tuple[TradeAccounting, ...]
    entries: tuple[LedgerEntry, ...]
    consumer_surplus: float
    producer_surplus: float
    government_balance: float
    real_cost: float
    financing_cost: float
    net_social_surplus: float

    def account_delta(self, account: str) -> float:
        return sum(entry.amount for entry in self.entries if entry.account == account)

    @property
    def transfer_balance(self) -> float:
        return sum(entry.amount for entry in self.entries)


def compute_period_ledger(
    trades: Iterable[Trade],
    *,
    policy: PolicyFlows | None = None,
    other_real_costs: float = 0.0,
    marginal_cost_of_public_funds: float = 0.20,
) -> PeriodLedger:
    policy = policy or PolicyFlows()
    if other_real_costs < 0 or marginal_cost_of_public_funds < 0:
        raise ValueError("real costs and marginal cost of funds must be nonnegative")

    trade_rows: list[TradeAccounting] = []
    entries: list[LedgerEntry] = []
    consumer_surplus = 0.0
    producer_surplus = 0.0
    taxes = 0.0
    subsidies = 0.0

    for trade in trades:
        tax = policy.buyer_tax_rate * trade.price
        subsidy = policy.seller_subsidy_per_unit
        buyer_payment = trade.price + tax
        seller_receipt = trade.price
        cs = trade.buyer_value - buyer_payment
        ps = seller_receipt + subsidy - trade.seller_cost
        consumer_surplus += cs
        producer_surplus += ps
        taxes += tax
        subsidies += subsidy
        trade_rows.append(
            TradeAccounting(
                trade=trade,
                buyer_payment=buyer_payment,
                seller_receipt=seller_receipt,
                tax=tax,
                subsidy=subsidy,
                consumer_surplus=cs,
                producer_surplus=ps,
            )
        )
        entries.extend(
            (
                LedgerEntry(
                    f"buyer:{trade.buyer_id}",
                    -buyer_payment,
                    "purchase",
                    trade.bid_order_id,
                ),
                LedgerEntry(
                    f"seller:{trade.seller_id}",
                    seller_receipt + subsidy,
                    "sale",
                    trade.ask_order_id,
                ),
                LedgerEntry("government", tax - subsidy, "tax_and_subsidy", trade.ask_order_id),
            )
        )

    fines = 0.0
    for seller_id, amount in policy.seller_fines:
        fines += amount
        producer_surplus -= amount
        entries.extend(
            (
                LedgerEntry(f"seller:{seller_id}", -amount, "fine", seller_id),
                LedgerEntry("government", amount, "fine", seller_id),
            )
        )

    government_balance = taxes + fines - subsidies
    real_cost = (
        other_real_costs
        + policy.administration_cost
        + policy.compliance_cost
        + policy.false_positive_cost
    )
    financing_cost = marginal_cost_of_public_funds * max(0.0, -government_balance)
    net_social_surplus = (
        consumer_surplus + producer_surplus + government_balance - real_cost - financing_cost
    )
    ledger = PeriodLedger(
        trades=tuple(trade_rows),
        entries=tuple(entries),
        consumer_surplus=consumer_surplus,
        producer_surplus=producer_surplus,
        government_balance=government_balance,
        real_cost=real_cost,
        financing_cost=financing_cost,
        net_social_surplus=net_social_surplus,
    )
    if not isclose(ledger.transfer_balance, 0.0, abs_tol=1e-9):
        raise AssertionError("monetary ledger is not balanced")
    return ledger
