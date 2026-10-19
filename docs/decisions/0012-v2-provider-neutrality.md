# ADR 0012: Native wrappers, identical semantic inputs

Status: accepted.

Each provider receives identical prompt and schema bytes through its native structured-output API. The adapter layer changes transport representation only. Semantic hashes, returned model IDs, usage, latency, and all failure states are recorded. Retries are disabled so providers are compared on first-attempt behavior and cost.
