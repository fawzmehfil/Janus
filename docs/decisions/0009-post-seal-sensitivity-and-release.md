# 0009 — Post-seal sensitivity analysis and final release

- **Status:** accepted
- **Date:** 2026-10-19

## Context

Hidden-v1 collection was sealed in commit `95b6b9b` before labels and oracle
outcomes were opened. The first post-seal analysis applied the preregistered
`do_nothing` fallback to 44 provider-incomplete responses and evaluated the
primary welfare objective. The research charter also requires valid-response-
only scoring, a worst-allowed-action treatment for invalid responses, and
sensitivity across the four alternative welfare objectives frozen before
collection.

## Decision

1. Keep the original intention-to-evaluate analysis as the only primary
   analysis. No prompt, model response, hidden world, action, or simulator
   setting changes after unblinding.
2. Reconstruct every hidden decision state from the committed private manifest
   and replay the same 256 indexed disturbance tapes for all five actions.
   Each recovery run yields the complete frozen utility vector, so no provider
   request is necessary.
3. Verify the recomputed primary action means against all 100 committed primary
   oracle files to a tolerance of `1e-9` before accepting any alternative-
   objective result.
4. Report three scoring rules for every objective:
   - intention-to-evaluate with the frozen `do_nothing` fallback;
   - valid responses only; and
   - incomplete responses assigned the lowest-utility allowed action in their
     own world.
5. Preserve ambiguity-bundle clustering and 10,000 bootstrap samples in every
   interval. The valid-only analysis retains all valid model replicates in
   their original bundles rather than treating calls as independent worlds.
6. Generate paper tables as CSV and figures as accessible SVG from the two
   machine-readable reports. These are presentation artifacts only and do not
   alter an estimand.

## Consequences

The alternative-objective replay is computationally expensive but incurs zero
API cost. It also reveals whether the central policy comparison depends on the
primary welfare weights. Hidden-v1 remains one-use: this analysis does not make
new model calls or replace incomplete responses. The release tag records the
sealed responses, primary analysis, sensitivity report, and publication
outputs together.
