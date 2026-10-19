# Janus v2 measurement charter

## Scope

Janus v2 measures one-shot intervention selection under causal ambiguity in one synthetic double-auction market. It retains the four latent mechanisms, five available actions, ten-period public observation contract, one decision, and twenty-period consequence horizon from v1. It does not add markets, agents, actions, tools, or multi-step planning.

The unit of scientific independence is the ambiguity bundle. The worlds and replicates nested inside a bundle are repeated measurements, not independent observations.

## Estimands

The primary estimand is paired model utility minus the utility of a frozen, strong policy trained only on public observations, evaluated on identical ambiguity bundles. A negative number means the model underperforms the public-information policy.

Full-information regret is reported separately. The decomposition is:

`oracle - model = (oracle - public policy) + (public policy - model)`

The first term is information loss; the second is decision loss. This prevents unobservable-state uncertainty from being mislabeled as a model reasoning failure.

## Comparators and failures

Four model roles are preregistered: Luna continuity, a stronger OpenAI model, one Anthropic model, and one Google model. Exact dated versions are frozen only after development admission testing. Every role sees the same semantic prompt, schema, observations, worlds, and replicate schedule through native provider wrappers.

The intention-to-evaluate analysis preserves incomplete outputs, refusals, timeouts, rate limits, transport failures, schema failures, empty responses, and budget stops. There are no silent retries. The primary invalid-response rule assigns the worst action utility in that world; valid-only and do-nothing assignment are sensitivities.

## Claims

Permitted claims compare model behavior on Janus v2. The benchmark does not establish competence to regulate real markets, general economic intelligence, or safety in deployment.

## Stop rules

Hidden-v2 generation is blocked until all local design gates pass. Paid development is blocked until a dated model/pricing freeze and explicit user budget approval. Hidden evaluation is blocked until every provider achieves at least 98% valid structured output on at least 100 representative development trials, output allowances are frozen, and prospective cost ceilings are approved. Unblinding is blocked until all four response manifests are committed.
