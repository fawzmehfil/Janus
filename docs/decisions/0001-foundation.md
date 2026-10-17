# Decision 0001: deterministic functional foundation

**Status:** Accepted  
**Phase:** 0A

Janus uses immutable dataclasses for market state and strict frozen Pydantic
models for configuration. Period transitions return a new state. Random draws
come from named NumPy streams or an indexed disturbance tape whose values are
pure functions of semantic keys.

This is intentionally more explicit than a module-global seeded generator. It
prevents one intervention branch from shifting another branch's future draws
when their control flow differs. Monetary taxes, subsidies, and fines are
represented by balanced cash entries; real and financing costs are recorded
separately and subtracted exactly once from social surplus.

The first implementation keeps participant order generation competitive and
deterministic. Behavioral strategies and interventions enter in Phase 0B.

