# Contributing to Janus

Janus is both software and a frozen research measurement. Contributions are welcome, but
changes to the benchmark need stronger controls than ordinary application code.

## Start here

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy
```

The package requires Python 3.12 or newer. Keep generated run output under `artifacts/`,
which is ignored by default.

## Contribution types

### Software and documentation

Bug fixes, tests, documentation, provider-neutral infrastructure, and analysis tooling may
target the current implementation when they do not alter a frozen estimand or artifact.

### Benchmark changes

Any change to market mechanics, scenario generation, public observations, prompts, output
schemas, actions, welfare definitions, parsing, invalid-response handling, or primary
analysis requires:

1. a new versioned design amendment;
2. new frozen manifests and hashes;
3. rerunning the applicable local gates;
4. a fresh hidden set; and
5. a clear statement that results are not comparable without qualification.

Do not rewrite v1 or v2 artifacts in place.

### New results

New model results should disclose the exact endpoint, provider, serving mode, prompt/schema
hashes, sampling settings, output allowance, retry policy, pricing snapshot, collection date,
invalid responses, and total cost. Hidden outcomes must remain sealed until response
manifests are committed.

## Research invariants

Contributions must preserve:

- conservation and feasibility checks in the market engine;
- indexed random streams and common future disturbances across action branches;
- strict separation of public observations from hidden state;
- bundle-level statistical independence;
- intention-to-evaluate accounting for failures; and
- deterministic regeneration of public tables from frozen inputs.

## Pull requests

Keep pull requests focused and describe:

- what changed;
- whether the change can affect a frozen measurement;
- tests and local gates run;
- any new data, cost, or provider dependency; and
- whether generated artifacts are included.

Never commit API keys, `.env` files, local blog drafts, or unreviewed raw credentials.
