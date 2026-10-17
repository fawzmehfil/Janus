# Decision 0004: Pre-freeze validity calibration

- **Status:** provisional, pending the complete Phase 0D gate
- **Date:** 2026-10-17

## Context

The first development/calibration probe had acceptable held-out cause
predictability, but only `do_nothing` and `collusion_investigation` were ever
oracle-optimal. It also exposed two implementation problems: momentum state did
not retain the lagged prices needed across the decision boundary, and affected
participants were always the lexicographically first actors. Separately, the
oracle multiplied the specification's per-period epsilon reference by the full
recovery horizon, changing the stated 0.5% floor from 4.2 to 84 utility units.

## Decision

1. Retain the four most recent transaction prices in immutable `MarketState` so
   counterfactual branches begin from the complete momentum-relevant state.
2. Select affected actors through the named `affected_participants` RNG stream,
   keyed by scenario and role.
3. Complete the predeclared public matching vector with log-return and spread
   paths.
4. Interpret the epsilon floor as 0.5% of the analytic per-period attainable
   surplus (840); paired-difference Monte Carlo error remains the other term.
5. Provisionally calibrate the buyer tax from 10% to 2%, price-ceiling real
   administration from 1 to 5 units per period, and one-time investigation plus
   compliance costs from 7 to 50 units. These magnitudes prevent harmless
   interventions from satisfying the near-optimality gate solely because their
   real costs were below the epsilon threshold.
6. Raise momentum feedback and narrow momentum participation to 30%–50% so the
   mechanism can create welfare-relevant distorted marginal trades rather than
   moving only inframarginal bids.
7. Raise initial cash from 10 to 100 reference-price units for generated worlds.
   This removes unintended cash exhaustion during the 30-period pre-decision
   simulation; cash constraints remain disabled neither implicitly nor through
   credit.

## Guardrails

These changes occur before any hidden-world generation or frozen release. They
must be judged on the entire balanced calibration split, never on hand-picked
worlds. The release still fails unless it has at least three oracle actions,
no constant epsilon-optimal action above 60%, held-out overlap within the frozen
range, and separated baselines. Failed configurations and their audit results
remain in the Phase 0D report.
