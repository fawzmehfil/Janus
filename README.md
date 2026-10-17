# Janus

Janus is a research benchmark for evaluating one-shot LLM market intervention
under causal ambiguity. Different hidden market mechanisms are constructed to
produce similar public histories; a regulator must diagnose the situation and
select one of five fixed interventions. Outcomes are scored by a local,
counterfactual simulator rather than an LLM judge.

The authoritative research design is [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

## Development status

Janus is being implemented phase-by-phase. Paid model calls are prohibited
until the local simulator, accounting, counterfactual evaluator, baselines,
validity gates, and budget controls have passed.

## Local setup

```bash
uv sync --all-groups
uv run pytest
uv run janus analytic-fixture
```

The package requires Python 3.12 or newer. All stochastic behavior must use the
named/indexed randomness interfaces in `janus.core.randomness`; module-global
random-number generators are forbidden.

