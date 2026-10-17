# Phase 0D benchmark-validity gate

**Date:** 2026-10-17  
**Code commit at run start:** `7031792`  
**Paid model/API calls:** 0  
**Outcome:** PASS — proceed to Phase 0E; do not begin a paid smoke test yet.

## Release command

```bash
uv run janus phase0d \
  --output artifacts/phase0d-release \
  --rollouts 256 \
  --workers 4
```

The command generated 80 candidates per family for each split, selected 10
development bundles (40 worlds) and 15 calibration bundles (60 worlds), ran
256 paired future tapes for every action in every selected world, fitted all
non-oracle baselines on development only, and evaluated the frozen gates on
calibration. It then evaluated one representative world per family at 512 and
1,024 rollouts. Completed oracle files are content-addressed and reusable.

## Frozen gate results

| Gate | Observed | Criterion | Result |
|---|---:|---:|:---:|
| maximum pre-shock price deviation | 0.00548 | ≤ 0.05 | pass |
| maximum pre-shock volume deviation | 0.03571 | ≤ 0.15 | pass |
| maximum pre-shock half-window drift | 0.02857 | ≤ 0.05 | pass |
| maximum within-bundle distance | 1.09300 | ≤ 1.25 | pass |
| maximum standardized mean difference | 2.30466 | ≤ 2.50 | pass |
| mean cross-family nearest-neighbor distance | 0.78497 | ≤ 1.00 | pass |
| logistic macro-F1 | 0.76244 | 0.35–0.80 | pass |
| boosted-tree macro-F1 | 0.76563 | 0.35–0.80 | pass |
| distinct oracle-optimal actions | 3 | ≥ 3 | pass |
| maximum constant epsilon-optimal rate | 0.53333 | ≤ 0.60 | pass |
| uniquely epsilon-optimal worlds | 0.98333 | ≥ 0.50 | pass |
| action gap exceeds 2× paired MC SE | 0.98333 | ≥ 0.50 | pass |
| heuristic − random bootstrap lower bound | 325.35 | > 0 | pass |
| partial-information − heuristic lower bound | 93.41 | > 0 | pass |
| oracle − partial-information lower bound | 10.92 | ≥ 0 | pass |

The classifier range is a benchmark design criterion. It was revised once
before freeze from the provisional upper bound 0.70 to 0.80. Welfare-relevant
momentum dynamics made the causes more learnable, but held-out error remains
about 24%; the construct is neither random guessing nor nearly perfect cause
classification.

## Oracle action balance and uncertainty

Strict best-action counts across the 60 calibration worlds were:

| Action | Worlds |
|---|---:|
| `do_nothing` | 32 |
| `collusion_investigation` | 15 |
| `transaction_tax` | 13 |
| `price_ceiling` | 0 |
| `supply_subsidy` | 0 |

One additional momentum world tied `transaction_tax` with `do_nothing` under
the epsilon rule, so the explicitly retained tied-world count is one. The
maximum constant epsilon-optimal rate is therefore 32/60, or 53.3%.

Representative best actions were unchanged at 256, 512, and 1,024 rollouts:

| Family | Stable action | Maximum action-value mean delta versus 1,024 |
|---|---|---:|
| supply shortage | `do_nothing` | 1.10 |
| demand increase | `do_nothing` | 2.70 |
| seller collusion | `collusion_investigation` | 7.55 |
| momentum demand | `transaction_tax` | 4.59 |

## Baseline ordering

Calibration mean primary utilities were:

| Policy | Mean utility |
|---|---:|
| uniform random | 16,418.59 |
| development-tuned public heuristic | 16,758.15 |
| development-CV observation-only ridge | 16,865.26 |
| full-information oracle | 16,878.36 |

The estimated full-information gap is 13.10 utility units. All three adjacent
differences have nonnegative 95% ambiguity-bundle-clustered bootstrap lower
bounds; the first two are strictly positive. The partial-information ridge
regularization selected by bundle-preserving development CV was 100. The
projected 95% half-width for the oracle-minus-random effect with 25 hidden
bundles is 7.84 utility units. This supports a precise pilot comparison, not a
broad claim about real-world regulatory competence.

## Failed approaches retained

1. The first generator/default-intervention probe produced only
   `do_nothing` and `collusion_investigation`. Momentum state lost its return
   history at the decision boundary, and affected actors were always the first
   lexicographic IDs. Both were implementation defects, not evidence about the
   construct.
2. A horizon-scaled epsilon floor of 84 made nearly costless alternatives tie
   almost everywhere. The specification's analytic reference is explicitly
   840 **per period**, so the corrected 0.5% floor is 4.2; paired-difference
   uncertainty remains the other epsilon term.
3. The provisional 10% transaction tax overcorrected bounded momentum demand.
   A development-only sweep supported the frozen 2% magnitude.
4. A four-rollout end-to-end diagnostic passed 14 of 15 gates but reported an
   80% constant-policy epsilon rate. The gate was not weakened; the release run
   used 256 rollouts and passed at 53.3%.

All changes occurred before hidden-set generation and are recorded in Decision
0004. No calibration world was manually inserted or removed.

## Artifact integrity

The local release artifacts are intentionally ignored by Git because they
contain complete oracle samples. Their committed references are:

| Artifact | SHA-256 |
|---|---|
| `configs/phase0d.json` | `21e5055db50bb15d9e3d6290ac28ca806f7a426692638cfc86a6024a7e5c3bd2` |
| `artifacts/phase0d-release/run_manifest.json` | `45167927e3af69d525603c83cc7fe0862f95fdbb0cf76eb6f12a2aa9e8bbdee1` |
| `artifacts/phase0d-release/split_manifest.json` | `8bd4055375e42f0fe48b6bdb11c4dde20a4d609e46e97e51e97bdc024fe0a558` |
| `artifacts/phase0d-release/validity_report.json` | `c985785e81533daa1eed7f978d315bfd48878009fb0462f24c06ca8816dcc330` |
| `artifacts/phase0d-release/partial_policy.json` | `9639bb62479fdb172cd69ddd625eafd2ed9e9c15764061003ebe03e595f19aa5` |

The run manifest contains the SHA-256 digest of every per-world oracle file,
including all 512/1,024 convergence artifacts.

## Limitations and decision

- The upper overlap classifier score is close to its 0.80 ceiling; hidden-set
  diagnostics must be monitored without retuning.
- Only three of five actions are optimal in calibration. Price ceilings and
  supply subsidies remain meaningful harmful controls, not positive fixtures.
- The small, fixed-participant double auction is an analytic benchmark, not an
  empirical model of any specific market.
- The observation-only ridge is a strong fitted baseline, not a proven Bayes
  policy; the 13.10-unit information gap is an estimate conditional on this
  simulator and feature contract.
- Scenario and intervention magnitudes were calibrated for validity and must
  not be described as real economic estimates.

**Recommendation: proceed to Phase 0E.** The local causal-ambiguity construct
is viable, but no paid call is authorized until strict parsing, caching,
pricing, budget preflight, automatic stop, mock end-to-end replay, and hidden
manifest controls also pass.
