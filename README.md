# Janus

Janus is a research benchmark for evaluating one-shot LLM market intervention
under causal ambiguity. Different hidden market mechanisms are constructed to
produce similar public histories; a regulator must diagnose the situation and
select one of five fixed interventions. Outcomes are scored by a local,
counterfactual simulator rather than an LLM judge.

The authoritative research design is [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

## Development status

Phases 0A–0E are complete and benchmark v1 is frozen as an evaluation-ready
local release. The accounting kernel, mechanisms, counterfactual oracle,
ambiguity bundles, baselines, strict mock/provider harness, hard budget
controls, hidden-set commitment, and local scientific-validity gates pass.
No paid model call has been made. A five-world paid smoke test is technically
eligible but remains a separately authorized next step.

The release evidence, exact commands, hashes, cost estimates, limitations, and
recommendation are in [`reports/readiness.md`](reports/readiness.md).

## Local setup

```bash
uv sync --all-groups
uv run pytest
uv run janus analytic-fixture
uv run janus phase0d --output artifacts/phase0d-release --rollouts 256 --workers 4
uv run janus phase0e-dry-run --output artifacts/phase0e-freeze
uv run janus robustness --output artifacts/robustness/report.json --rollouts 64 --workers 4
uv run janus openai-smoke-preflight --max-usd 0.01
```

The live five-world smoke command, environment-key instructions, and layered
one-cent safeguards are documented in
[`docs/OPENAI_SMOKE_TEST.md`](docs/OPENAI_SMOKE_TEST.md). Running the preflight
does not read a key, access hidden artifacts, make a network request, or spend
money.

The package requires Python 3.12 or newer. All stochastic behavior must use the
named/indexed randomness interfaces in `janus.core.randomness`; module-global
random-number generators are forbidden.
