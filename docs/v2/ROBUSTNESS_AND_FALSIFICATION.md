# Janus v2 robustness and falsification matrix

Status: preregistered development evidence. Hidden-v2 outcomes have not been generated or inspected.

| Check | Development evidence | Status before hidden generation |
|---|---|---|
| Alternate clearing price | One calibration world per causal family, 64 paired rollouts, midpoint versus marginal-ask allocation/action comparison in `artifacts/robustness/report.json` | Pass |
| Increased oracle rollouts | One calibration world per family agrees at 256, 512, and 1,024 rollouts in `artifacts/phase0d-release/validity_report.json` | Pass |
| Tied and near-tied actions | Epsilon sets, action gaps, and paired Monte Carlo errors are reported for all 60 calibration worlds | Pass |
| Alternative welfare objectives | Five objectives have separately trained and frozen observation-only policies; direction and sign reversals are reported after unblinding | Pass design; outcomes pending |
| Observation ordering and numeric precision | Canonical JSON serialization and schema/hash regression tests freeze field order independence and numeric representation | Pass |
| Irrelevant identifiers | The public contract exposes no participant or scenario identifiers; opaque hidden-v2 IDs are assigned after selection | Pass by construction |
| Provider wrapper equivalence | Four-role zero-network replay requires identical semantic-input hashes and one checked-in schema | Pass |
| Hidden-label falsification | Welfare selection uses only the chosen action; diagnosis changes scores but cannot change the selected action's simulator outcome | Pass test |
| Baseline hidden-field access | Baseline features deserialize only the public observation contract; forbidden-key serialization and observation-only manifests are checked | Pass |
| Branch randomness | Existing paired-tape regression tests verify exogenous branch identity | Pass |
| Invalid fallback inclusion | Incomplete, refusal, timeout, rate, transport, empty, schema, and budget outcomes remain in the denominator and map to the worst action | Pass test |
| Negative-control semantics | Price-cap costs and subsidy transfer/financing behavior have explicit v2 microcase tests | Pass |
| Collection isolation | Collection accepts only the public manifest and records `private_hidden_artifacts_accessed=false`; unblinding is a separate four-manifest command | Pass test |

The alternative-welfare check is not expected to be invariant. A sign reversal is a result to elevate, not a failed test. The final archive must rerun this matrix in a clean environment and the independent auditor—not the implementation agent—must certify the result.
