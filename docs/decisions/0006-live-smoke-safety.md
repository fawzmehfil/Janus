# Decision 0006: One-cent live smoke boundary

- **Status:** accepted for the development smoke stage
- **Date:** 2026-10-17

## Context

The evaluation-ready release intentionally omitted an authenticated network
transport. Before collecting hidden responses, Janus needs five live requests
to validate API authentication, Structured Outputs, returned model identity,
usage mapping, cost reconciliation, caching, and failure behavior. The project
owner created a dedicated API key and configured a one-cent provider spend
limit, but did not authorize a request during implementation.

The earlier four-bytes-per-token calculation was suitable for cost planning but
was not a defensible hard upper bound. A paid runner also needed to handle
resumption without forgetting prior spend and to ensure SDK retries could not
silently create extra billable requests.

## Decision

1. Use exactly five deterministic observations regenerated from the frozen
   development split. The live smoke path never reads hidden manifests, labels,
   states, or oracles.
2. Use `gpt-6-luna` through `v1/responses`, `reasoning.effort=high`, no
   temperature or top-p override, a 512-token output ceiling, Standard/default
   service tier, strict JSON Schema, and `store=false`.
3. Require both `OPENAI_API_KEY` and the exact CLI confirmation
   `ONE_CENT_SMOKE`. The key is read only from process environment and is never
   written to requests, logs, cache records, or manifests.
4. Set the local ceiling to at most $0.01. Preflight counts every input byte as
   one token, adds 1,024 tokens of request-framing allowance, and reserves the
   complete output ceiling. The frozen five-primary-request bound is $0.0098262.
5. Permit a syntax repair only when remaining actual spend plus the repair's
   conservative bound fits. Otherwise preserve the original invalid artifact
   and record that budget blocked the repair.
6. Disable automatic SDK retries. Send a stable idempotency key derived from the
   logical trial. Any non-timeout provider exception stops the invocation.
7. Reconcile mutually exclusive uncached-input, cached-input, cache-write,
   output, and reasoning usage categories against the dated price table.
8. Sum prior immutable cache-artifact costs before a resumed invocation. Cached
   trials produce no provider call and no new charge.
9. Write a new timestamped manifest per invocation rather than overwrite prior
   evidence.
10. Keep the provider's project hard limit as an independent backstop. Local and
    provider limits may both stop the run; neither is treated as spending
    authorization.

## Outcome

The no-network preflight reports five development trials, zero paid calls, and
a $0.0098262 primary-request bound under a $0.01 local ceiling. Tests exercise
the fake-transport cold path, zero-call cache replay, usage reconciliation,
missing-key refusal, incorrect-confirmation refusal, and ceilings above one cent.
No live request was made while establishing this boundary.
