# Hidden evaluation runbook

The frozen hidden evaluation contains 100 public, opaque worlds with three
planned model replicates per world: exactly 300 logical requests. The collector
uses `gpt-6-luna`, high reasoning, a 2,048-token output ceiling, strict
Structured Outputs, no SDK retries, no syntax repair, no provider storage, and
no prompt-cache writes.

The collector reads only `manifests/hidden-v1-public.json`. It does not read the
private seed, hidden family labels, scenario state, oracle files, or outcomes.
The resulting response files and run manifest must be committed before any
private artifact is opened or joined for scoring.

## Cost gate

Run the zero-network preflight without setting an API key:

```bash
uv run janus openai-hidden-preflight --max-usd 0.84
```

For the frozen 300-request plan, the preflight reports:

- recorded live spend before the hidden run: `$0.0107931`;
- conservative remaining primary-request bound: `$0.8200065`;
- conservative cumulative bound: `$0.8307996`;
- rounded cumulative local hard ceiling: `$0.84`; and
- expected confirmation: `HIDDEN_V1_CUMULATIVE_USD_0_84`.

The bound counts every input byte as a token and reserves all 2,048 output
tokens for every request. It is deliberately much higher than expected spend.
Using the corrected smoke's observed mean as a planning estimate gives roughly
`$0.33` incremental and `$0.34` cumulative spend, but that estimate is not a
guarantee. Official GPT-6 Luna Standard pricing is `$0.10` per million input
tokens and `$0.50` per million output tokens.

Do not run the paid command until the account/project provider limit is at least
the approved amount and the user has explicitly approved the cumulative local
cap. Provider spend-limit enforcement can lag, so the local gate remains the
controlling protection.

## Paid collection after approval

Use a fresh restricted project key. Enter it without echoing it, run the
collector, and remove it automatically when the subshell exits:

```bash
(
  read -rs "OPENAI_API_KEY?Paste the restricted OpenAI API key: "
  echo
  export OPENAI_API_KEY
  uv run janus openai-hidden \
    --max-usd 0.84 \
    --confirm-spend HIDDEN_V1_CUMULATIVE_USD_0_84 \
    --output artifacts/openai-hidden-v1
)
```

The command is resumable. Completed logical trials are never sent again, even
if the installed SDK version changes. Any provider exception stops the run with
no blind retry. Invalid, incomplete, refusal, empty, and timeout responses are
retained with the predeclared `do_nothing` fallback and remain in the analysis.

After collection, verify that the printed manifest says `"passed": true`, then
commit the entire hidden response cache and that manifest before unblinding.
Never paste an API key into chat, a shell command, a file, or a Git commit.

## Completed hidden-v1 collection

The collection completed on 2026-10-19 with 300 provider calls and cumulative
spend of `$0.3475366`. The 300 response artifacts and run manifest were sealed
in commit `95b6b9b` before unblinding. The final analysis is recorded in
`reports/hidden-v1-results.md` and its machine-readable counterpart is
`artifacts/hidden-v1-analysis/report.json`.
