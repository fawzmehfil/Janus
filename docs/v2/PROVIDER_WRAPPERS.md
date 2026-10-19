# Provider wrapper contract

The v2 adapters accept an injected transport so all payload and failure behavior can be tested without a network. OpenAI uses Responses JSON Schema output; Anthropic uses Messages `output_config.format`; Google uses Interactions structured response format. Provider-specific payload shapes follow the official documentation current at implementation time and must be revalidated when exact dated model versions are frozen.

Every wrapper records requested and returned model IDs, SDK version, endpoint, response status, incomplete details, refusal, timeout, rate limit, transport error, provider request ID, input/cache/output/reasoning tokens, latency, rate-limit metadata, and timestamp. SDK retries must be zero. Janus itself performs no semantic or syntax retry in v2.

Official references:

- OpenAI structured outputs: https://platform.openai.com/docs/api-reference/responses
- Anthropic structured outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Google structured outputs: https://ai.google.dev/gemini-api/docs/structured-output
