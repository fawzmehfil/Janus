# Janus v2 development readiness

Date: 2026-10-19

## Outcome

The local scientific and engineering layer is complete. No paid v2 provider request was made, no exact current model version was selected, and hidden-v2 was not generated or unblinded.

Five public-information action-value policies are frozen under primary, consumer-protection, producer-continuity, stability-aware, and fiscal-neutral objectives. Bundle-preserving nested validation selected ridge with alpha 100 for each objective; boosted trees, polynomial ridge, and an extra-trees ensemble remained eligible candidates rather than decorative baselines. Each manifest records every fold, learning curve, training world, dependency version, and model hash.

The primary action audit classifies tax and collusion investigation as substantive, price ceiling and subsidy as negative controls, and do-nothing as an abstention control. Tax is uniquely epsilon-optimal in 21.7% of calibration worlds and investigation in 25.0%, both above the preregistered 10% floor. Across 60 calibration worlds, 98.3% have a unique statistically resolved best action; the strongest constant policy is epsilon-optimal in 53.3%. There are 28 close public-history pairs with different optimal actions, spanning every calibration bundle.

The revealed v1 release is explicitly reused as v2 development data. Its structural OOD pilot finds 11 same-mechanism, close-public-cue pairs whose optimal action reverses across ID and OOD, covering momentum demand and seller collusion. Fresh hidden-v2 generation must independently pass the same gate or abort.

The v1-informed allocation simulation compares the original three-replicate design, more bundles with two replicates, and a balanced split-replicate design. At about 300 trials per model, the selected 40-bundle/160-world split design has an expected 95% half-width of 53.5 utility units and an 80%-power minimum detectable effect of 76.5, versus 66.9 and 95.7 for v1's 25-bundle design. The model count never inflates the independent sample size.

## Engineering evidence

- Four model roles pass the offline native-payload dry run with identical semantic-input hashes and zero network calls.
- OpenAI, Anthropic, and Google adapters use native structured outputs, injected transports, zero harness retries, and complete failure/usage/latency metadata.
- The scored schema contains diagnosis, four probabilities, action, and confidence; rationale is not requested or scored.
- The final analysis provides paired bundle bootstrap intervals, Holm-corrected model comparisons, a REML mixed model, a deterministic variance cross-check, normalized and raw regret, behavioral/calibration diagnostics, risk-coverage, model and baseline economic vectors, and intention-to-evaluate failure preservation.
- `uv run janus v2-publication` deterministically regenerates CSV tables and an accessible vector primary-estimand figure from the sealed comparative report.
- The hidden generator requires a fresh committed seed, 40 bundles, 300 trials/model, the structural OOD gate, and all development gates.
- Reproduction archive and independent audit commands exist. CI, container recipe, license, citation metadata, and lockfile are present.

## Remaining gates

1. External subject-matter review of the frozen prompt wording.
2. Select exact dated model versions, endpoints, SDK/HTTP revisions, decoding parameters, and official prices.
3. Obtain explicit budgets, then run at least 100 representative development trials for each of the four roles.
4. Require at least 98% direct schema-valid outputs per provider and freeze the validated output allowance.
5. Commit the resulting admission manifest and exact model config.
6. Only then create and commit the fresh hidden-v2 seed commitment and generate hidden-v2.

Run `uv run janus v2-gates` for the machine-readable status. The expected result today is blocked on external prompt review, exact-model freeze, and four-provider admission; that is a safety property, not a failed build.
