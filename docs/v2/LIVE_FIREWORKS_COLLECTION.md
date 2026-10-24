# Live Fireworks collection runbook

No paid Janus v2 request has been made. Run paid commands only from the frozen
revision, with `FIREWORKS_API_KEY` present in the process environment. Never put
the key in a command, config, log, or commit.

The four frozen roles are:

- `gpt_oss_120b`
- `deepseek_v4p1_flash`
- `glm_5p3_flash`
- `nemotron_lightning_3p5_30b_a3b`

Each paid command operates on one role so spending can be inspected between
models. It requires `--max-usd` and the exact stage-specific confirmation shown
by `--help`. The ceiling is cumulative for that role and stage. Cache records
make reruns resumable without repeating completed requests. A separate scan of
all v2 Fireworks cache records enforces the $5 project ceiling, including an
interrupted run that did not finish a manifest.

## 1. Smoke test each exact model

The initial 512-token transport smoke exhausted the output allowance on
reasoning for all 20 calls. A first 2,048-token rerun at medium effort still
truncated 12 of 20 calls and revealed that Fireworks promotes DeepSeek V4.1
`medium` to its high tier. A low-effort 2,048-token diagnostic passed for GPT
and GLM but still truncated DeepSeek and Nemotron. ADR 0021 therefore freezes a
corrected shared 4,096-token allowance with minimum supported reasoning before
development admission: `low` for GPT-OSS and GLM, and `none` for DeepSeek and
Nemotron. A smoke passes only when all five responses are directly schema-valid.

```bash
uv run janus v2-smoke \
  --model-role gpt_oss_120b \
  --max-usd 0.10 \
  --confirm-spend 'I CONFIRM FIREWORKS V2 SMOKE SPEND'
```

Repeat for the other three roles. Inspect validation status, returned model ID,
token counts, request IDs, latency, and cost before continuing.

## 2. Collect the 100-trial development admission set

```bash
uv run janus v2-collect-development \
  --model-role gpt_oss_120b \
  --max-usd 0.35 \
  --confirm-spend 'I CONFIRM FIREWORKS V2 DEVELOPMENT SPEND'
```

Repeat for all roles, raising a role's local ceiling only if the frozen maximum
allocation and $5 project ceiling permit it. Then evaluate the four manifests:

```bash
uv run janus v2-admission \
  artifacts/v2-fireworks/development-v2/gpt_oss_120b/manifests/run-manifest.json \
  artifacts/v2-fireworks/development-v2/deepseek_v4p1_flash/manifests/run-manifest.json \
  artifacts/v2-fireworks/development-v2/glm_5p3_flash/manifests/run-manifest.json \
  artifacts/v2-fireworks/development-v2/nemotron_lightning_3p5_30b_a3b/manifests/run-manifest.json
```

Do not change prompts, schemas, output allowance, reasoning effort, or retries
to rescue one model. A model must produce at least 98 direct valid responses
among the same 100 trials, with no budget stop.

## 3. Seal and collect hidden-v2

Commit the development response manifests and `model-admission.json`, run
`janus v2-gates`, then create and commit the hidden seed commitment and public
manifest using `v2-hidden-seed` and `v2-hidden-generate`. Only after those gates
pass, run one hidden collection per role:

```bash
uv run janus v2-collect-hidden \
  --model-role gpt_oss_120b \
  --max-usd 1.50 \
  --confirm-spend 'I CONFIRM FIREWORKS V2 HIDDEN SPEND'
```

The hidden design remains 40 ambiguity bundles and 300 trials per model. Commit
all four complete blinded response manifests before unblinding. An independent
reviewer is not required to publish this as a disclosed evaluation demo; an
independent auditor is required only if the post claims independent audit.
