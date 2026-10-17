# Decision 0005: Evaluation harness and hidden-v1 freeze

- **Status:** accepted and frozen for benchmark v1
- **Date:** 2026-10-17

## Context

The project goal requires a frozen, evaluation-ready local release without paid
model calls. The original staged plan placed a live five-world smoke test before
hidden generation. That ordering conflicts with the zero-paid-call release
boundary but is not necessary for scientific concealment if the prompt, schema,
benchmark, and hidden-seed commitment are frozen first and provider-facing work
is limited to later operational validation.

An initial hidden OOD rehearsal with 80 candidates per family produced a maximum
within-bundle distance of 1.275, above the frozen 1.25 gate. The gate was not
relaxed and no world was manually selected. Increasing the OOD search pool to
160 candidates per family produced a valid release with maximum distance
1.191009.

## Decision

1. Freeze prompt `primary-v1`, decision schema `janus-decision-v1`, Phase 0D
   mechanics, and the 100-world hidden plan before any provider call.
2. Use a host-CSPRNG 63-bit seed; commit only its SHA-256 commitment before
   generation. Store the seed, private manifest, and 100 oracle artifacts under
   owner-only filesystem permissions.
3. Publish observations only under shuffled opaque world and bundle IDs. Do not
   publish family, stratum, scenario, parameter, state, or oracle fields.
4. Permit exactly one fixed syntax-only repair. Refusal, timeout, empty output,
   and terminal schema failure remain distinct statuses; all receive an explicit
   `do_nothing` scoring fallback without being relabeled valid.
5. Include provider/model/settings, exact inputs and schema, benchmark/prompt
   versions, and planned replicate ID in the immutable cache key.
6. Preflight every request using a dated price table and conservative token
   bound before invoking an adapter. Enforce both stage and total dollar caps.
7. Make the live OpenAI adapter depend on an injected transport. Tests may
   inspect payload and usage mapping without an SDK, credential, or network call.
8. Move the paid development smoke test after the local freeze. It requires
   explicit authorization. Any semantic change arising from it creates a new
   benchmark/prompt version and hidden release.
9. Retain midpoint clearing and the primary welfare objective as frozen defaults.
   Marginal-ask clearing and the four secondary objectives are analysis-only
   sensitivity paths and do not alter hidden-v1 artifacts.

## Outcome

The fresh mock gate completed five valid trials at $0.00, then replayed all five
from cache with zero additional adapter calls. The alternate-clearing audit kept
the primary optimal action unchanged in all four representative families. The
consumer-protection objective selected `supply_subsidy` instead, so normative
sensitivity is disclosed rather than treated as a failed engineering gate.

The public hidden release contains 25 bundles and 100 worlds. Its seed remains
unrevealed until response manifests are committed. No paid model call has been
made and no private hidden label or action was inspected during release review.
