# OpenAI development smoke test

## Observed first-run outcome

The first authorized run completed on 2026-10-19 for $0.0042324 and did not
access hidden evaluation data. All five primary responses were empty and all
five fixed repairs returned schema-valid JSON, producing ten provider calls.
Those repaired decisions are not scientifically usable because the repair saw
an empty primary response. Do not rerun the original smoke blindly.

The harness now records provider `status`, `incomplete_details`, and usage for
each primary and repair request. An `incomplete` response is a terminal status
and is never sent to syntax repair.

The one-request diagnostic completed on 2026-10-19 for $0.0010983. It returned
one direct schema-valid response with no repair, using 4,433 input tokens and
1,310 total output tokens, of which 1,136 were reasoning tokens. Cumulative
recorded spend is $0.0053307. The 2,048-token setting is therefore frozen for a
clean five-world rerun; see Decision 0007. Do not run that stage until a $0.02
cumulative local and provider hard limit is separately approved.

This run validates the live provider boundary using five development
observations. It does not access or score the hidden evaluation. The command is
deliberately impossible to start without both a process-local API key and an
exact spend confirmation.

## Implemented safeguards

- five fixed development observations only;
- `gpt-6-luna` through the Responses API;
- strict structured output and one budget-conditional syntax repair;
- `reasoning.effort=high`, no temperature/top-p override;
- 512-token output ceiling and default/Standard service tier;
- explicit-only prompt-cache mode with no breakpoints, preventing the
  higher-priced cache writes that implicit mode could create;
- local ceiling no greater than $0.01;
- one-byte-per-input-token bound plus 1,024 framing tokens per request;
- official SDK automatic retries disabled;
- stable logical-trial idempotency key;
- API response storage disabled with `store=false`;
- immutable response cache and timestamped run manifests;
- prior cached spend included when resuming; and
- no secret values in artifacts or manifests.

The platform hard limit is an independent backstop. OpenAI documents that
provider-side enforcement is not instantaneous, so Janus does not rely on it as
the only limit.

## 1. Install and run the free preflight

```bash
cd /Users/fawzmehfil/Desktop/market-intervention-eval
uv sync --all-groups
uv run janus openai-smoke-preflight --max-usd 0.01
```

Expected preflight facts:

- `network_calls`: `0`
- `paid_api_calls`: `0`
- `development_only`: `true`
- `trial_count`: `5`
- `planned_primary_request_upper_bound_usd`: `0.0098262`
- `fits_ceiling`: `true`

This command neither reads `OPENAI_API_KEY` nor contacts OpenAI.

## 2. Export the key without shell history

In the same zsh terminal that will run Janus:

```bash
read -s "OPENAI_API_KEY?Paste the Janus project API key: "
echo
export OPENAI_API_KEY
```

Paste the key at the prompt and press Return. Nothing is displayed and the key
is not included in the command history.

Verify presence without printing the secret:

```bash
test -n "$OPENAI_API_KEY" && echo "OPENAI_API_KEY is set"
```

Never use `echo "$OPENAI_API_KEY"`, paste the key into a command argument, or
store it in a tracked file. `.env` files are ignored as a defensive backstop,
but a temporary environment variable is preferred for this one-time smoke test.

## 3. Paid command—do not run until explicitly authorized

```bash
uv run janus openai-smoke \
  --max-usd 0.01 \
  --confirm-spend ONE_CENT_SMOKE \
  --output artifacts/openai-smoke-v1
```

Without the exact confirmation or a nonempty environment key, Janus exits
before constructing the SDK client or sending a request. A ceiling above $0.01
is rejected.

The process may stop before five completed trials if a repair or earlier usage
leaves insufficient conservative budget. That is expected hard-stop behavior,
not permission to raise the ceiling automatically.

## 4. Inspect the latest manifest

```bash
latest_manifest=$(ls -t artifacts/openai-smoke-v1/manifests/run-*.json | head -1)
jq '{passed, completed, provider_calls_this_invocation, cumulative_cost_usd, local_hard_ceiling_usd, returned_model_ids, status_counts, budget_stop, fatal_error_type}' "$latest_manifest"
```

Do not rerun blindly after a failure. Inspect the manifest and cache first; a
safe resume reuses completed logical trials without another provider call.

## 5. Remove the key from the terminal

```bash
unset OPENAI_API_KEY
test -z "${OPENAI_API_KEY-}" && echo "OPENAI_API_KEY removed"
```

Close the terminal afterward if it is no longer needed.

## 6. One-request 2,048-token diagnostic

Run the free preflight first. It includes the immutable $0.0042324 smoke spend
when checking the original cumulative one-cent ceiling:

```bash
uv run janus openai-diagnostic-preflight \
  --max-usd 0.01 \
  --prior-output artifacts/openai-smoke-v1 \
  --output artifacts/openai-diagnostic-2048-v1
```

The frozen preflight bound is $0.0027333 for one request and $0.0069657
cumulative. The diagnostic uses the first development observation, preserves
`reasoning.effort=high`, raises only `max_output_tokens` to 2,048, permits no
repair, and never reads hidden artifacts.

After exporting a new key as described in section 2, run:

```bash
uv run janus openai-diagnostic \
  --max-usd 0.01 \
  --confirm-spend ONE_CENT_DIAGNOSTIC \
  --prior-output artifacts/openai-smoke-v1 \
  --output artifacts/openai-diagnostic-2048-v1
```

Unset the key afterward. A passing diagnostic must contain one direct `valid`
response, one provider call, no repair, and cumulative cost below $0.01. If it
is still `incomplete`, stop and revise the output allowance under a new budget;
do not continue to the hidden evaluation.

This diagnostic has now passed. Its immutable manifest is under
`artifacts/openai-diagnostic-2048-v1/manifests/`; do not submit it again.
