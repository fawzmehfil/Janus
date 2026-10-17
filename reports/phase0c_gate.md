# Phase 0C gate record

**Date:** 2026-10-17  
**Commit basis:** working tree after Phase 0C implementation

## Automated checks

- `uv run pytest --cov=janus --cov-report=term-missing -q`: 31 passed,
  91% package statement coverage.
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed.
- `uv run mypy`: passed in strict mode.
- `git diff --check`: passed.

## Paired-branch evidence

Regression tests verify that action branches start from an unchanged immutable
state and receive identical indexed buyer-value noise, seller-cost noise, and
per-order tie priorities for every common order. Priorities are keyed by period
and order ID, so adding or removing an order in one branch cannot perturb the
priority of another order.

## Preliminary convergence audit

One representative state from each family was evaluated over the full
20-period recovery horizon at 16, 32, and 64 paired rollouts per action. The
estimated best action was stable at all three scales:

| Family | Best action at 16/32/64 |
|---|---|
| supply shortage | `do_nothing` |
| demand increase | `do_nothing` |
| seller collusion | `do_nothing` |
| momentum demand | `transaction_tax` |

This establishes mechanical convergence for the Phase 0C implementation, not
benchmark validity. Only two actions are optimal in these four default cases,
which is insufficient for the Phase 0D action-diversity gate. Phase 0D must
generate and calibrate varied scenarios that produce at least three optimal
actions without weakening the predeclared gate. The final calibration audit
must repeat convergence at 256, 512, and 1,024 rollouts on its representative
subset.

