# Janus v2 evidence-based prompt review

Date: 2026-10-19  
Decision: approved for model-admission testing
Review type: AI-assisted internal desk review; not independent external review

## Scope and limitation

This review replaces the unavailable external subject-matter review as a release
gate. It checks internal fidelity, measurement clarity, and consistency with
published sources. It does not claim independent assurance or validate Janus as
a model of real-world regulation. Independent review remains desirable before
publication.

This substitution is consistent with the NIST AI RMF's treatment of independent
review as a way to improve testing and mitigate internal bias, rather than as a
universal prerequisite. The compensating controls here are explicit source
mapping, hash-pinned artifacts, automated semantic tests, preregistration, and a
prominent independence limitation.

## Evidence and findings

| Criterion | Evidence checked | Finding |
|---|---|---|
| Welfare language matches the ledger | `src/janus/core/ledger.py`, `src/janus/simulation/runner.py`, and `src/janus/core/config.py` | Pass after revision. The prompt now states the 20% financing burden and distinguishes transfers from real costs. This matches Canadian program-evaluation guidance and HM Treasury's Green Book treatment of transfers as offsetting social costs and benefits. |
| Intervention descriptions match simulator behavior | `src/janus/interventions/actions.py` and `src/janus/simulation/runner.py` | Pass after revision. Every rate, recurring cost, one-time cost, detection probability, false-positive consequence, fine, and intervention horizon is now stated. These are synthetic benchmark definitions, not empirical policy-effect claims. The collusion description is directionally consistent with OECD material on price fixing and supply restriction. |
| Diagnosis probabilities are meaningful | `src/janus/v2/contracts.py` and `src/janus/v2/statistics.py` | Pass. The four probabilities must sum to one, and the diagnosis must be an argmax. Brier score, log loss, and reliability analysis evaluate the probabilistic forecasts. Proper-scoring-rule literature supports eliciting and evaluating distributional forecasts this way. |
| Confidence means action optimality | `src/janus/oracle/counterfactual.py`, `src/janus/v2/contracts.py`, and `src/janus/v2/statistics.py` | Pass after revision. The prompt now defines confidence as a probability that the action satisfies the exact frozen epsilon-optimality rule. |
| Provider neutrality | `configs/prompts/primary-v2.txt`, `src/janus/v2/providers.py`, and `src/janus/v2/harness.py` | Pass. No provider, model family, expected action frequency, oracle result, baseline behavior, or hidden-set construction is named. A semantic-input hash checks identical prompt and schema bytes across adapters. |
| Minimal structured output | `configs/schemas/decision-v2.json` and `src/janus/v2/contracts.py` | Pass. The response asks only for fields needed by the preregistered behavioral and calibration analyses, rejects extra fields, and requests no rationale or chain of thought. |

## Material revisions made before approval

The original prompt did not disclose the quantitative investigation mechanics
and did not define epsilon-optimality. That made the action comparison
asymmetric and the confidence target underspecified. Prompt version
`primary-v2.1` fixes both issues and also exposes every policy cost and the
financing-cost parameter. No paid v2 request preceded these changes, so no
admission result was invalidated.

## Sources

Sources were checked on 2026-10-19.

- [NIST AI RMF Core, Measure](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/): repeatable, documented TEVV, uncertainty, benchmarks, and the value of independent review.
- [Government of Canada, Program Evaluation Methods](https://www.tbs-sct.canada.ca/cee/pubs/meth/pem-mep-eng.pdf): social cost-benefit analysis and the cancellation of transfer payments.
- [HM Treasury, The Green Book 2026](https://www.gov.uk/government/publications/the-green-book--2/the-green-book-2026): social cost-benefit accounting and treatment of taxes, subsidies, penalties, and fines as economic transfers.
- [OECD, Competition enforcement](https://www.oecd.org/en/topics/competition-enforcement.html): coordinated price increases and supply restrictions as anticompetitive conduct with welfare effects.
- [Gneiting and Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf): calibration and proper evaluation of probabilistic forecasts.
- [JSON Schema Core](https://json-schema.org/draft/2019-09/json-schema-core): the validation vocabulary used for the strict response contract.

## Approval boundary

Approval means the prompt and schema are sufficiently specified for the
four-model development admission test. It does not authorize spending,
hidden-v2 generation, hidden evaluation, unblinding, or real-policy claims.
