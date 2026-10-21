# ADR 0020: low-cost Fireworks open-model panel

## Decision

Janus v2 uses four open-weight models on the Fireworks standard-serverless
OpenAI-compatible API: `gpt-oss-120b`, `deepseek-v4p1-flash`,
`glm-5p3-flash`, and `nemotron-lightning-3p5-30b-a3b`. The exact IDs,
reasoning setting, output allowance, endpoint, metadata, and 2026-10-21 prices
are frozen in `configs/models/v2-candidates.json` and
`configs/pricing/v2-fireworks-2026-10-21.json`.

## Rationale

The panel is intended to demonstrate a rigorous evaluation of non-verifiable
economic decisions at very low cost, not to rank every frontier proprietary
model. A single serving provider removes provider-family interpretation while
preserving model-level variation in family, parameter scale, active parameter
count, architecture, and reasoning behavior.

`gpt-oss-20b` was in the proposed panel but was replaced before any paid v2
call because Fireworks did not list it as a serverless model on the retrieval
date. Nemotron Lightning 3.5 30B-A3B is the inexpensive serverless substitute.
The local project ceiling remains $5 even though expected use is $2–$3.

## Sources

- https://docs.fireworks.ai/serverless/pricing
- https://fireworks.ai/models?modelTypes=LLM%2CServerless&providers=OpenAI
- https://docs.fireworks.ai/guides/querying-text-models
- https://docs.fireworks.ai/structured-responses/structured-response-formatting
- https://docs.fireworks.ai/guides/reasoning

## Consequences

Every model receives the same prompt, schema, 512-token allowance, and medium
reasoning effort through one no-retry transport. Each must still pass 98 direct
valid responses among 100 development trials before hidden collection. The
panel can be changed only by a new preregistration amendment before paid data
collection.
