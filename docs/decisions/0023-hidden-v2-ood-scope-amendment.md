# ADR 0023: Narrow hidden-v2 structural OOD scope before model collection

- **Status:** accepted after seed attempt 5 failed and before drawing attempt 6
- **Date:** 2026-10-25

## Context

The original hidden-v2 gate required at least four non-overlapping close-cue
ID/OOD oracle-action reversals spanning at least two mechanism families. Five
independently committed seeds were generated under that rule. Each produced
between four and fourteen reversal pairs, but every pair belonged to
`momentum_demand`. All five attempts were rejected and archived before any
hidden model request was made.

Across the 800 evaluated worlds, the accepted hidden bundles showed no action
variation in demand increase, supply shortage, or seller collusion. The
revealed v1 development pilot contained one seller-collusion reversal, but the
fresh generator did not reproduce that second family in five draws. Continuing
to draw until a rare second-family result appeared would make the release
conditional on an unstable generator event and consume substantial local
compute without improving the model comparison.

## Decision

Amend the structural OOD acceptance rule before model collection:

- retain the maximum standardized public-cue distance of 1.25;
- retain at least four non-overlapping ID/OOD oracle-action reversal pairs;
- require those pairs in at least one mechanism family rather than two; and
- draw and commit a fresh seed after this amendment rather than retroactively
  accepting any of the five observed attempts.

The code now reads the minimum-family threshold from the frozen hidden plan;
the prior hard-coded default of two remains in place for the revealed v1 pilot.
The new amendment is included in the hidden seed's frozen-input hashes.

## Consequences

The confirmatory OOD result supports relationship-shift claims only for the
mechanism families represented by the accepted reversal audit. Aggregate OOD
minus ID estimates remain reportable, but they must not be described as
evidence that the cue-to-action mapping shifts across multiple mechanisms.
All five rejected commitments, revealed seeds, gate summaries, and private
oracle sets remain preserved for audit. Model responses did not influence this
amendment.
