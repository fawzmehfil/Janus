# Janus benchmark v1 readiness report

**Release date:** 2026-10-17

**Outcome:** evaluation-ready local release

**Paid model/API calls:** 0

**Recommendation:** PROCEED only to an explicitly authorized five-development-world smoke test; do not yet run the 300-request hidden evaluation.

## Release boundary

Janus v1 implements the full local research path in `PROJECT_SPEC.md`: a
deterministic double-auction kernel, four separated causal mechanisms, five
interventions, paired counterfactual recovery, primary and secondary utilities,
epsilon-optimal full-information oracles, public-information baselines,
ambiguity matching, clustered uncertainty analysis, a strict cached and
budgeted model harness, and a committed hidden set.

“Evaluation-ready” means the local scientific and software gates pass, the
prompt/schema/hidden release are frozen, the provider request boundary can be
tested without network access, and the next request would be budget-checked
before dispatch. It does **not** mean that provider connectivity or live model
behavior has been tested. No credential was read and no paid request was made.

## Gate results

| Gate | Evidence | Result |
|---|---|:---:|
| analytic reference market | quantity 21, price 100, surplus 840 | pass |
| accounting and clearing invariants | unit/property/differential suite | pass |
| causal and intervention semantics | integration suite | pass |
| named RNG, replay, paired branches | determinism/regression suite | pass |
| Phase 0D scientific validity | all 15 frozen gates | pass |
| strict model response handling | valid, repair, refusal, timeout, empty/schema paths | pass |
| immutable cache semantics | same logical trial replays; planned replicates differ | pass |
| cost controls | conservative preflight stops before adapter; usage categories reconcile | pass |
| mock end-to-end release | 5 valid trials, $0.00, immediate replay adds 0 calls | pass |
| hidden construction | 25 bundles, 100 worlds, 4 causes per bundle | pass |
| hidden observational distance | maximum 1.191009 ≤ 1.25 | pass |
| public concealment | opaque IDs; no label/stratum/scenario/state/oracle fields | pass |
| alternate clearing | 4/4 representative primary actions unchanged | pass |
| normative sensitivity | consumer-protection objective changes 4/4 actions | disclosed |

The Phase 0D values, baseline ordering, convergence evidence, and failed
approaches are reported separately in [`phase0d_gate.md`](phase0d_gate.md).

## Frozen hidden release

Hidden-v1 was generated from code commit
`d22c2ca0b800066b54455919c9960faa2792ca4a`. The later alternate-clearing and
normative-objective paths preserve the frozen defaults and do not change that
release. Reproduction of hidden-v1 should check out the committed basis rather
than silently regenerate it from a later revision.

- 15 in-distribution bundles and 10 one-dimension-shifted OOD bundles;
- four worlds per bundle and 100 worlds total;
- 256 paired oracle rollouts per action and world;
- private directory mode `0700`; seed, private manifest, and oracle files mode
  `0600`;
- reveal policy: private seed/labels/oracles only after model response manifests
  are committed.

The first OOD rehearsal used 80 candidates per family and missed the frozen
distance gate at 1.275. The threshold was not weakened. The search pool was
raised to 160 per family, after which the maximum distance was 1.191009. This is
search-capacity calibration, not manual inclusion or outcome-based selection.

## Robustness evidence

At 64 paired rollouts for one matched bundle (four representative worlds), the
primary best action was identical under midpoint and marginal-ask pricing:

| Hidden family | Primary action under both clearing rules |
|---|---|
| supply shortage | `do_nothing` |
| demand increase | `do_nothing` |
| seller collusion | `collusion_investigation` |
| momentum demand | `transaction_tax` |

Producer-continuity, stability-aware, and fiscal-neutral objectives agreed with
the primary action in this bundle. Consumer protection, which doubles consumer
surplus weight, selected `supply_subsidy` for all four worlds. Therefore the
allocation result is robust to the tested clearing convention, while the policy
ranking is meaningfully sensitive to the normative objective. Claims must remain
conditional on the primary welfare function.

## Budget estimate before any paid call

Using the dated standard short-context Luna table ($0.10/M input and $0.50/M
output), the exact frozen prompt/schema bytes, a conventional four-bytes-per-
token planning estimate, and the 512-token output ceiling:

| Planned stage | Requests | Planning input estimate | Output-token ceiling | Estimated cost | If every request needs one repair |
|---|---:|---:|---:|---:|---:|
| smoke | 5 | 20,094 | 2,560 | $0.00329 | $0.00658 |
| one-model hidden run | 300 | 1,205,325 | 153,600 | $0.19733 | $0.39467 |

These are planning estimates, not invoices or mathematical upper bounds. The
live smoke runner therefore uses a separate hard preflight: it disables
prompt-cache writes, counts every input byte as an ordinary input token, adds
1,024 framing tokens per request, and reserves the full 512-token output
ceiling. The five primary requests total **$0.0098262** under that deliberately
high bound. A syntax repair is sent only if the
remaining local budget passes the same check; it may displace a later primary
request rather than cross $0.01.

The estimates exclude future provider price
changes, taxes, and any non-token fees, and assume the price snapshot remains
valid. Cached-input discounts are not credited. The project’s $10 development
and $100 cross-model ceilings remain hard stops, not targets. Revalidate the
official [GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna)
and [API pricing](https://developers.openai.com/api/docs/pricing) immediately
before authorization.

## Commands run

```bash
uv sync --all-groups
uv run janus analytic-fixture
uv run janus phase0d --output artifacts/phase0d-release --rollouts 256 --workers 4
uv run janus phase0e-dry-run --output artifacts/phase0e-freeze
uv run janus robustness --output artifacts/robustness/report.json --rollouts 64 --workers 4
uv run janus openai-smoke-preflight --max-usd 0.01
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
uv run pytest --cov=janus --cov-report=term-missing -q
git diff --check
```

Final verification on the current release candidate: Ruff formatting and lint
pass; strict mypy passes across 49 source files; 55 tests pass in both ordinary
and coverage runs; total statement coverage is 89%; and `git diff --check`
passes.
The two end-to-end research commands (`phase0d` and `robustness`) are validated
by their content-addressed local reports rather than rerun inside unit coverage.

## Frozen hashes

| Artifact | SHA-256 |
|---|---|
| Phase 0D config | `21e5055db50bb15d9e3d6290ac28ca806f7a426692638cfc86a6024a7e5c3bd2` |
| Phase 0D run manifest | `45167927e3af69d525603c83cc7fe0862f95fdbb0cf76eb6f12a2aa9e8bbdee1` |
| prompt `primary-v1` | `bdb0a5a83555f401dd9578ce8d57087448da0cb510ddf0b0d7476089da37e071` |
| decision schema | `0ab11ce4fa976fe80759ff5788b8d732c90431130560d6cb0a6e334b3cad01fe` |
| mock pricing snapshot | `db49f03f0e44980855c69d9d92e31454809fe49185195921b451b2c0e4d24db5` |
| Luna pricing snapshot | `3a4898af7a98e58c7dd04b0df937bd031dbfa06f1354933c5927c1fc3eabb3f1` |
| Phase 0E fresh manifest | `74e6147dda76804834b4879f9626b49d89c6c02f600a316f3fe2352ab2c2d751` |
| private seed commitment | `105ef4c17827ba9e90bf06f5c3eeab6979495fa7820890b86860f7b56dbeb51b` |
| commitment manifest | `a75b034f4d4cc50c5587c1b8c5684d59f1758ba94d9ac0289a02561c7d269c45` |
| public hidden manifest | `27fd5c4e399ec8b66413ac0ab1b3f1d467d068a7c463f35e98cde21653549cfb` |
| private hidden manifest | `298ad3d0c245071fbd8a76ed8892c9db22e8670a24eaa32d582dadd571e334e2` |
| robustness report | `458861bc44dede684b7fde8889644cbe8a22c3fa591c7be47fcb93d2ff973524` |

## Unresolved limitations

1. No live provider request, authentication result, rate-limit behavior, or
   returned-model snapshot has been exercised. The official SDK transport is
   installed and validated through injected no-network responses only.
2. Hidden private files are protected by local Unix permissions, not encryption,
   a separate evaluator host, or hardware-backed access control.
3. The fitted observation-only baseline is not a proven Bayes policy. The 13.10
   utility-unit information gap is conditional on the simulator and feature set.
4. Calibration uses only three of five actions as strict optima. Price ceilings
   and supply subsidies function mainly as harmful controls under the primary
   objective.
5. The overlap classifier is close to its frozen upper boundary (macro-F1 about
   0.76 versus 0.80). Causes are ambiguous, not indistinguishable.
6. The robustness audit covers four representative worlds at 64 rollouts. It is
   evidence against gross clearing-rule dependence, not a full robustness grid.
7. Normative sensitivity is real: a consumer-weighted objective changes the
   representative policy choices. The benchmark cannot establish a politically
   neutral “correct” intervention.
8. The market is small, synthetic, scripted, and single-cause. It supports
   controlled claims about reasoning under causal ambiguity, not claims about
   competence in real regulation or any specific economy.
9. No model outcomes exist yet, so there is no evidence that Luna—or any other
   model—beats the random, heuristic, or observation-only baselines.
10. Model aliases, availability, and prices can change. Selection, snapshot,
    settings, and price table must be revalidated and declared immediately before
    a live run.

## Recommendation

**Proceed, with a stop point.** The local construct is scientifically viable and
the evaluation plumbing is ready. If the owner explicitly authorizes spending,
run only the five-development-world Luna smoke test with a ceiling no higher than
$0.01 under the current price table. Diagnose transport, schema, and accounting
failures without examining hidden outcomes. Commit its response manifest before
considering the 300-request run. Stop and version a new benchmark if a semantic
prompt, action, objective, generator, or hidden-set change is required.
