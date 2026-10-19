# Prompt and rationale policy review

Status: design review complete; external subject-matter review required before the provider admission run.

The scored response contains only diagnosis, four probabilities, action, and confidence. Free-form rationale was removed because it increased truncation risk in v1, is not needed to score the intervention, can reward verbosity, and is not comparable across providers with different reasoning interfaces. No hidden chain of thought is requested or stored.

The prompt states the welfare objective, causal ambiguity, mechanism IDs, action semantics, horizon, and output contract directly. It does not mention provider identity, model family, expected action frequencies, oracle outcomes, baseline behavior, or hidden-set construction. The same UTF-8 prompt and schema bytes feed every adapter; a semantic-input hash verifies equality before provider-specific packaging.

Reviewers must check that the intervention descriptions match simulator behavior, the welfare language matches the ledger, confidence is explicitly action-optimality confidence, and no wording privileges one provider. Any wording change increments the prompt version and invalidates prior admission results.
