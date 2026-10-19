# Decision 0007: Corrected live output-token boundary

- **Status:** accepted for the corrected development smoke stage
- **Date:** 2026-10-19

## Context

The first authorized five-world live smoke used GPT-6 Luna with high reasoning
effort and a 512-token output ceiling. All five primary responses contained no
visible text and all five syntax repairs returned generic decisions because the
repair input contained an empty response. The run exercised authentication,
model identity, usage mapping, cost controls, and immutable artifacts, but its
decisions are not scientifically usable.

OpenAI's Responses API reports reasoning tokens inside the total output-token
allowance. A response can therefore end as `incomplete` with reason
`max_output_tokens` before producing visible output. The original adapter did
not preserve `status` or `incomplete_details`, so the empty responses were
misclassified as syntax failures.

## Decision

1. Preserve provider `status`, `incomplete_details`, and usage separately for
   every primary and repair request.
2. Classify provider-incomplete responses as `incomplete`. Never submit them to
   syntax repair, including when the visible response is empty.
3. Validate a 2,048-token ceiling with one fixed development observation, high
   reasoning effort, the frozen `primary-v1` prompt and decision schema, no
   repair, and the original cumulative one-cent ceiling.
4. Freeze 2,048 tokens for the corrected five-world development smoke if the
   diagnostic returns one direct schema-valid response. Adoption for the hidden
   evaluation remains conditional on a clean five-world rerun.
5. Require new spending authorization before that rerun. Its five primary
   requests have a conservative incremental bound of $0.0136662. Including the
   $0.0053307 already recorded, the cumulative bound is $0.0189969, so the next
   local and provider hard limit must be at least $0.02.
6. Continue to prohibit access to hidden observations, labels, and outcomes
   until the corrected five-world response manifest is committed.

## Outcome

The one-request diagnostic returned `status=completed` and a directly valid
structured response with no repair. It used 4,433 input tokens and 1,310 output
tokens, including 1,136 reasoning tokens, and cost $0.0010983. Cumulative
recorded spend is $0.0053307, below the original one-cent ceiling.

This result confirms output-token exhaustion as the operational cause of the
first smoke failure and supports the 2,048-token corrected setting for the next
development-only gate. It does not authorize or validate the 300-request hidden
evaluation.

The corrected five-world gate subsequently returned five direct schema-valid
responses in five provider calls, with no repairs, incomplete responses,
refusals, timeouts, schema failures, cache writes, or hidden-artifact access.
The gate cost $0.0054624 and brought cumulative recorded spend to $0.0107931.
The 2,048-token setting is therefore accepted for the planned hidden-run
preflight. This outcome does not itself authorize hidden requests or reveal any
hidden labels or outcomes.
