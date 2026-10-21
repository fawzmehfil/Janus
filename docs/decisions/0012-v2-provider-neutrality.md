# ADR 0012: Native wrappers, identical semantic inputs

Status: accepted.

Each model receives identical prompt and schema bytes through the same frozen Fireworks JSON-schema transport. Semantic hashes, returned model IDs, usage, latency, request IDs, cost, and all failure states are recorded. Retries are disabled so models are compared on first-attempt behavior and cost.
