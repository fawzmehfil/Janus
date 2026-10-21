# Janus

Janus is a research benchmark for evaluating one-shot LLM market intervention
under causal ambiguity. Different hidden market mechanisms are constructed to
produce similar public histories; a regulator must diagnose the situation and
select one of five fixed interventions. Outcomes are scored by a local,
counterfactual simulator rather than an LLM judge.

The authoritative research design is [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

Janus v2 is under sealed comparative-development. Its charter is
[`docs/v2/CHARTER.md`](docs/v2/CHARTER.md); v1 remains an immutable released
measurement rather than being rewritten in place.

## Development status

Phases 0A–0E are complete and benchmark v1 is frozen as an evaluation-ready
local release. The accounting kernel, mechanisms, counterfactual oracle,
ambiguity bundles, baselines, strict mock/provider harness, hard budget
controls, hidden-set commitment, and local scientific-validity gates pass.
The first five-world paid development smoke completed under its one-cent limit,
but every 512-token primary response ended before visible output and required a
repair. A one-world, 2,048-token development diagnostic then returned one direct
schema-valid response with no repair. The corrected five-world rerun subsequently
returned five direct valid responses with no repairs or retries. The provider
gate is complete. The guarded 300-request hidden collection completed on
2026-10-19, its response set was committed before unblinding, and the frozen
post-seal analysis is complete. The main result is mixed: GPT-6 Luna beat
uniform random, did not reliably beat always doing nothing or the frozen
public-information baseline, and performed much worse on OOD seller-collusion
worlds. The final report includes valid-only, worst-action, and alternative-
welfare sensitivity analyses plus publication-ready source tables and figures.
See
[`reports/hidden-v1-results.md`](reports/hidden-v1-results.md).

The release evidence, exact commands, hashes, cost estimates, limitations, and
recommendation are in [`reports/readiness.md`](reports/readiness.md).

The v2 local scientific layer is complete: five objective-specific public-
information policies are frozen after bundle-preserving nested validation,
the action-margin and matched-contrast audits pass, the structural OOD pilot
finds cue-to-action reversals, and a four-role offline provider dry run passes.
The precision analysis selects 40 independent bundles and 300 trials per model.
No hidden-v2 set has been generated and no v2 paid provider call has been made.
An evidence-based, hash-pinned internal prompt review is complete, with the lack
of independent external review disclosed. Exact model versions and the four
100-trial provider admission runs remain intentional gates; `uv run janus
v2-gates` reports them.

## Local setup

```bash
uv sync --all-groups
uv run pytest
uv run janus analytic-fixture
uv run janus phase0d --output artifacts/phase0d-release --rollouts 256 --workers 4
uv run janus phase0e-dry-run --output artifacts/phase0e-freeze
uv run janus robustness --output artifacts/robustness/report.json --rollouts 64 --workers 4
uv run janus openai-smoke-preflight --max-usd 0.01
uv run janus openai-hidden-preflight --max-usd 0.84
uv run janus hidden-sensitivity --workers 8
uv run janus hidden-publication
uv run janus v2-dry-run --output artifacts/v2-dry-run
uv run janus v2-local-development --objective-rollouts 64 --workers 4
uv run janus v2-ood-pilot
uv run janus v2-gates
# After the sealed four-model run and authorized unblinding only:
uv run janus v2-publication
```

The live five-world smoke command, environment-key instructions, and layered
one-cent safeguards are documented in
[`docs/OPENAI_SMOKE_TEST.md`](docs/OPENAI_SMOKE_TEST.md). Running the preflight
does not read a key, access hidden artifacts, make a network request, or spend
money.

The hidden collection procedure, conservative `$0.84` cumulative gate, safe key
entry, resumption behavior, pre-unblinding commit rule, and local-only final
analysis commands are documented in
[`docs/HIDDEN_EVALUATION.md`](docs/HIDDEN_EVALUATION.md).

The package requires Python 3.12 or newer. All stochastic behavior must use the
named/indexed randomness interfaces in `janus.core.randomness`; module-global
random-number generators are forbidden.
