# Janus

Janus is a research benchmark for evaluating one-shot LLM market intervention
under causal ambiguity. Different hidden market mechanisms are constructed to
produce similar public histories; a regulator must diagnose the situation and
select one of five fixed interventions. Outcomes are scored by a local,
counterfactual simulator rather than an LLM judge.

The authoritative research design is [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

## Development status

Phases 0A–0D are complete. The accounting kernel, mechanisms, counterfactual
oracle, ambiguity bundles, baselines, and local validity gates pass. Phase 0E
is implementing the mock/provider harness and hard budget controls. Paid model
calls remain prohibited until that final local gate passes.

## Local setup

```bash
uv sync --all-groups
uv run pytest
uv run janus analytic-fixture
uv run janus phase0d --output artifacts/phase0d-release --rollouts 256 --workers 4
```

The package requires Python 3.12 or newer. All stochastic behavior must use the
named/indexed randomness interfaces in `janus.core.randomness`; module-global
random-number generators are forbidden.
