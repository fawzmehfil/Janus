# OpenAI development smoke test

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
