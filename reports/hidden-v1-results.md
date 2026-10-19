# Janus hidden-v1 results

- **Evaluation date:** 2026-10-19
- **Model:** `gpt-6-luna`
- **Response commit before unblinding:** `95b6b9b`
- **Trials:** 100 worlds × 3 planned replicates = 300
**Inference:** 95% ambiguity-bundle-clustered bootstrap intervals, 10,000 samples

## Protocol and integrity

The paid collection completed all 300 planned requests without SDK retries or
repair calls. The response manifest and all 300 response artifacts were hashed,
scanned for key-shaped secrets, and committed before private labels or oracle
outcomes were opened. The post-seal analysis verified the public manifest,
private manifest, seed commitment, every response hash, and all 100 oracle-file
hashes.

The collection cost `$0.3367435` incrementally and `$0.3475366` cumulatively,
below the approved cumulative cap of `$0.84`.

Of 300 responses, 256 (85.3%) were directly schema-valid and 44 (14.7%) ended
`incomplete` because the 2,048-token output allowance was exhausted. There were
no refusals, timeouts, schema failures, repairs, or cache writes. Under the
preregistered intention-to-evaluate rule, every incomplete response remained in
the denominator and received the frozen `do_nothing` fallback.

## Primary results

| Metric | Estimate | 95% clustered interval |
|---|---:|---:|
| mean normalized full-information regret | 0.00695 | [0.00456, 0.00944] |
| epsilon-optimal action rate | 55.0% | [48.7%, 61.3%] |
| utility difference vs uniform random | +401.15 | [+387.40, +414.06] |
| utility difference vs always do nothing | +3.88 | [-4.82, +16.03] |
| utility difference vs frozen heuristic | -28.35 | [-82.15, +23.91] |
| utility difference vs partial-information ridge | -99.07 | [-139.21, -59.28] |

Median normalized regret was zero. Mean utility was 16,959.24, compared with
16,558.09 for uniform random, 16,955.36 for always doing nothing, 16,987.58 for
the frozen heuristic, 17,058.31 for the observation-only ridge, and 17,075.91
for the full-information oracle. The estimated oracle-minus-partial information
gap was 17.60 utility units, while the partial-minus-LLM decision gap was 99.07.
The model also underperformed the best constant hidden-set action,
`collusion_investigation`, whose mean utility was 17,033.44.

## Distribution shift and failure concentration

| Split/family | Mean normalized regret | Epsilon-optimal | Valid response rate |
|---|---:|---:|---:|
| ID | 0.00197 | 57.8% | 91.7% |
| OOD | 0.01440 | 50.8% | 75.8% |
| supply shortage | 0.00028 | 84.0% | 81.3% |
| demand increase | 0.00037 | 69.3% | 94.7% |
| momentum demand | 0.00071 | 60.0% | 96.0% |
| seller collusion | 0.02642 | 6.7% | 69.3% |

OOD minus ID normalized regret was 0.01243, with a 95% clustered interval of
[0.01104, 0.01373]. Harm was concentrated in seller-collusion worlds: the model
selected `collusion_investigation` only 3 times overall and selected
`do_nothing` 251 times. Across all trials its actions were 251 `do_nothing`, 44
`transaction_tax`, 3 `collusion_investigation`, 2 `price_ceiling`, and no
`supply_subsidy`.

## Diagnosis, confidence, and consistency

Diagnosis accuracy was exactly 25.0% when incomplete responses were retained as
incorrect, with a 95% clustered interval of [23.3%, 26.7%]. Valid-only accuracy
was 29.3%. The model labeled 250 of its 256 valid responses as
`momentum_demand`, so diagnosis performance did not exceed four-class chance in
the intention-to-evaluate analysis.

Confidence had a positive association with epsilon-optimality: clustered
correlation 0.357, 95% interval [0.239, 0.469]. Mean within-world action
agreement was 86.7%; 61% of worlds had unanimous actions across all three
replicates.

## Preregistered hypotheses

| Hypothesis | Result |
|---|---|
| H1: lower loss / higher utility than uniform random | supported |
| H2: higher utility than always do nothing | not supported |
| H3: OOD increases regret | supported |
| H4: confidence positively tracks epsilon-optimality | supported |
| H5: diagnosis accuracy exceeds chance | not supported |

## Interpretation

GPT-6 Luna produced a conservative policy that was far better than uniform
random but statistically indistinguishable from always doing nothing. It left a
large and precisely estimated decision gap relative to a frozen policy using
the same public information. Performance was strong on shortage cases and weak
on seller collusion, where both diagnosis collapse and under-intervention drove
the tail losses. These findings apply only to the frozen synthetic Janus-v1
benchmark, prompt, model alias, settings, and evaluation window; they are not
evidence about real-world regulatory competence.

The 14.7% incomplete rate is an operational limitation. The corrected
five-world smoke did not reveal it. Increasing the output allowance and rerunning
these hidden items would invalidate the one-use hidden-v1 protocol, so hidden-v1
is reported as collected. A higher allowance may be preregistered for a future
benchmark version.
