# ADR 0021: Correct the Fireworks reasoning-output boundary

- **Status:** accepted before development admission collection
- **Date:** 2026-10-24

## Context

The first paid Fireworks transport smoke used the preregistered four-model
panel, medium reasoning effort, native JSON-schema output, no retries, and a
512-token output allowance. Authentication, exact returned model identity,
request accounting, and budget controls worked. All 20 responses ended at the
512-token limit, however, and all were classified `incomplete`; reasoning used
509--512 tokens per sampled response and no scorable JSON was returned. Total
recorded cost was $0.02272925.

The harness also reported `passed=true` because its pass flag represented only
that the planned number of trials had records. That was not a valid smoke gate.

## Decision

Before any 100-trial development admission collection:

1. Increase the shared output allowance from 512 to 4,096 tokens for all four
   models.
2. Use the minimum reasoning configuration each model supports: `low` for
   GPT-OSS and thinking-only GLM, and `none` for DeepSeek and Nemotron. The
   first corrected 2,048-token smoke retained `medium`; GPT-OSS and GLM each returned
   four valid responses and one truncation, while DeepSeek and Nemotron
   truncated all five. Fireworks documents that DeepSeek V4.1 promotes
   `medium` to its high tier, while `low` selects its lower tier, and that
   GPT-OSS produces shorter reasoning at low effort. A 4,096-token low-effort
   diagnostic then passed GPT-OSS and GLM but produced only two valid DeepSeek
   responses and no valid Nemotron responses, making further allowance growth
   incompatible with the low-cost design.
3. Keep prompts, schemas, model IDs, sampling settings, no-retry policy,
   admission threshold, and invalid-response policy unchanged.
4. Define a passing collection smoke as complete with every response directly
   schema-valid. Preserve `completed` separately from `passed`.
5. Repeat the same five-trial smoke for all four models. A subsequent
   low-effort 2,048-token diagnostic produced five valid responses for GPT-OSS
   and GLM, one for DeepSeek, and none for Nemotron, so the final shared ceiling
   is 4,096 tokens. The 100-trial development stage remains blocked until the
   corrected smoke passes.

## Rationale

The failure was infrastructure-level truncation rather than evidence about
decision quality. The project previously validated the same boundary in v1: a
2,048-token allowance produced direct structured output where 512 tokens did
not. Minimum supported reasoning better matches this project's inexpensive
demonstration goal and avoids silently giving DeepSeek a high-effort treatment.
The output allowance remains shared, but reasoning capability differs because
GPT-OSS and GLM cannot disable it. The primary estimand therefore compares the
exact frozen serving configurations, not equal internal compute. This is a
disclosed limitation. No hidden-v2 data were inspected or altered.

## Outcome

The final capability-aware configuration returned 20 direct schema-valid
responses in 20 final-setting trials, with exact returned model identities and
no retries. GPT-OSS and GLM used `low`; DeepSeek and Nemotron used `none`. The
GLM final verification reused five immutable cached valid records and made no
new provider calls. Cumulative recorded Fireworks spend across all smoke and
diagnostic settings was $0.114403055, well below the $5 project ceiling. The
four-model development admission collection is authorized from this frozen
configuration.

## Evidence

- `artifacts/v2-fireworks/smoke-v2/*/records/*.json`
- `docs/decisions/0007-live-output-limit.md`
- https://docs.fireworks.ai/guides/reasoning
