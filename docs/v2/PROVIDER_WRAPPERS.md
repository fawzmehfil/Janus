# Fireworks transport contract

The v2 adapter accepts an injected transport so payload and failure behavior can be tested without a network. The live panel uses Fireworks' OpenAI-compatible `/inference/v1/chat/completions` endpoint and native `response_format.type=json_schema`. `FIREWORKS_API_KEY` is read only by explicitly confirmed paid CLI commands. The base URL is configurable but must match the frozen serving configuration.

The transport records requested and returned model IDs, endpoint, response status, incomplete details, refusal, timeout, rate limit, transport error, request ID, input/cache/output/reasoning tokens, latency, rate-limit metadata, timestamp, and estimated and actual cost. Both transport and harness retries are zero. Cache-keyed resume never duplicates a completed request.

Official references:

- Fireworks text API: https://docs.fireworks.ai/guides/querying-text-models
- Fireworks structured outputs: https://docs.fireworks.ai/structured-responses/structured-response-formatting
- Fireworks reasoning settings: https://docs.fireworks.ai/guides/reasoning
- Fireworks serverless pricing: https://docs.fireworks.ai/serverless/pricing
