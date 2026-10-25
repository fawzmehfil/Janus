# Janus v2 release checklist

## Before hidden generation

- [x] v1 release tag and artifacts unchanged.
- [x] Five objective-specific public-information baselines hash-frozen.
- [x] Nested bundle CV and learning curves reviewed.
- [x] Action-role and unique-optimum audit passes.
- [x] Matched causal contrast audit passes.
- [x] Structural OOD reversal audit passes with at least four non-overlapping pairs in one or more families; the post-failure scope amendment and five rejected seeds are disclosed in ADR 0023.
- [x] Precision allocation and minimum detectable effect frozen.
- [x] Preregistered robustness and falsification matrix passes locally.
- [x] Prompt and schema versions hash-frozen after an evidence-based desk review; lack of independent review disclosed.
- [x] Exact model IDs, endpoints, SDKs, decoding settings, output limits, and dated pricing frozen.
- [x] Each model passes at least 98 direct valid responses among 100 representative development trials.
- [x] Prospective per-model and $5 cumulative budgets frozen; each paid invocation still requires explicit confirmation.
- [x] Fresh hidden-v2 seed commitment committed against the frozen code revision.

## Before unblinding

- [x] Same hidden-v2 public manifest used for all four models.
- [x] No retries or model-specific prompt changes.
- [x] Raw responses and usage metadata immutable and hash-manifested.
- [x] Budget stops and all other failures retained.
- [x] Four blinded response manifests committed to version control.
- [x] Commit hash recorded in the unblinding command.

## Publication archive

- [x] Raw provider responses and response manifests.
- [x] Revealed private worlds, states, scenarios, seed, and oracle samples.
- [x] Public observations and exact model/pricing metadata.
- [x] Frozen baselines, prompts, schemas, configs, analysis code, tables, and figures.
- [x] Environment lock, container recipe, CI configuration, LICENSE, and CITATION.cff.
- [x] Local non-independent audit passes with no missing paths or hash failures.
- [ ] Optional: an independent auditor runs `uv run janus v2-audit ARCHIVE` from a separately controlled checkout before any claim of independent audit.
- [x] Claims remain limited to comparative behavior on Janus v2.

Build the archive only after unblinding and final analysis:

```bash
uv run janus v2-archive \
  --response-commit PRE_UNBLIND_COMMIT \
  --response-manifest PATH_TO_MANIFEST_1 \
  --response-manifest PATH_TO_MANIFEST_2 \
  --response-manifest PATH_TO_MANIFEST_3 \
  --response-manifest PATH_TO_MANIFEST_4 \
  --output artifacts/janus-v2-reproduction
```
