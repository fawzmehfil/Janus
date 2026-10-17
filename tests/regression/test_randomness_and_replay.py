from dataclasses import asdict

from janus.core.config import AccountingConfig, MarketConfig
from janus.core.engine import run_period
from janus.core.fixtures import analytic_market, competitive_orders
from janus.core.ledger import PolicyFlows
from janus.core.randomness import DisturbanceTape, NamedRNG
from janus.core.replay import PeriodReplay, content_hash


def test_named_streams_are_repeatable_and_independent() -> None:
    pool = NamedRNG(123)
    first = pool.generator("scenario_parameters", "world-1").normal(size=8)
    again = pool.generator("scenario_parameters", "world-1").normal(size=8)
    other = pool.generator("tie_breaking", "world-1").normal(size=8)
    assert (first == again).all()
    assert not (first == other).all()


def test_disturbance_draws_do_not_depend_on_call_order() -> None:
    tape = DisturbanceTape(99, "future-0")
    expected = tape.normal("period", 5, "seller", 2)
    for index in range(20):
        tape.uniform("unrelated", index)
    assert tape.normal("period", 5, "seller", 2) == expected


def test_full_period_replay_is_content_identical() -> None:
    market_config = MarketConfig()
    accounting_config = AccountingConfig()
    policy = PolicyFlows()
    state = analytic_market(market_config)
    bids, asks = competitive_orders(state)
    tape = DisturbanceTape(1, "replay")
    priority = tape.tie_priority(tuple(order.order_id for order in (*bids, *asks)), state.period)
    result = run_period(
        state,
        bids,
        asks,
        tie_priority=priority,
        market_config=market_config,
        accounting_config=accounting_config,
        policy=policy,
    )
    record = PeriodReplay.capture(
        result,
        bids=bids,
        asks=asks,
        tie_priority=priority,
        market_config=market_config,
        accounting_config=accounting_config,
        policy=policy,
    )
    replayed = record.replay()
    assert content_hash(asdict(replayed)) == content_hash(asdict(result))
