# ADR 0022: Auditable hidden-seed rejection sampling

- **Status:** accepted after attempt 1 failed a frozen design gate and before
  drawing attempt 2
- **Date:** 2026-10-25

## Context

Hidden-v2 seed attempt 1 was committed before generation. All 160 oracle files
were completed without any model inference calls. The candidate set passed the
frozen ambiguity-distance, action-coverage, and matched-contrast checks. It
produced nine close ID/OOD action-reversal pairs, exceeding the required four,
but those pairs covered only one latent family rather than the frozen minimum
of two. The generator therefore correctly refused to publish a hidden manifest.

Changing the gate after observing this result would invalidate the design.
Silently replacing the seed would also make the process unauditable.

## Decision

Use auditable rejection sampling without changing any acceptance criterion:

1. A failed attempt receives a public rejection record containing its committed
   digest, revealed abandoned seed, frozen criteria, observed gate summary, and
   hashes of its private candidate audit and complete oracle set.
2. Preserve the failed private artifacts under
   `artifacts/hidden-v2-rejections/attempt-NNN/`.
3. Confirm that no model inference request used the rejected observations.
4. Commit the rejection record before drawing and committing the next seed.
5. Repeat only until the unchanged frozen generation gates pass. Every failed
   attempt must be retained and disclosed. No model outputs may be collected
   until a public hidden manifest exists.

This procedure makes seed replacement reproducible and prevents undisclosed
manual selection. The eventual estimand remains conditional on the published
   generation acceptance criteria.
