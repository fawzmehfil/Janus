# Examples

## Local end-to-end check

Run the provider-free v2 path:

```bash
uv run janus v2-dry-run --output artifacts/v2-dry-run
```

This exercises the provider-neutral harness and response parser without making an external
request or spending money.

## Regenerate frozen public results

```bash
uv run janus v2-publication
```

This reads the frozen comparative report and deterministically regenerates the public CSV
tables and SVG figures under `reports/v2-publication/`.

## Run the scientific test suite

```bash
uv run pytest
```

The suite includes analytic fixtures, property-based invariants, regressions, and integration
coverage. The exact live collection procedure is intentionally kept in
[`../docs/v2/LIVE_FIREWORKS_COLLECTION.md`](../docs/v2/LIVE_FIREWORKS_COLLECTION.md) because it
requires budget approval, a provider key, and frozen model configuration.
