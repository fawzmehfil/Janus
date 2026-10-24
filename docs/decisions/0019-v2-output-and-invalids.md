# ADR 0019: Minimal output, zero repair, worst-action primary fallback

Status: accepted.

The shared output allowance and model-specific minimum reasoning settings are frozen candidate settings. The initial 512-token allowance and medium effort were amended by ADR 0021 after the pre-development smoke showed uniform reasoning-token exhaustion and provider-specific effort normalization. The final allowance is 4,096 tokens; GPT-OSS and thinking-only GLM use their minimum supported `low` effort, while DeepSeek and Nemotron use `none`. Each exact model must complete 100 representative development trials with at least 98% direct schema validity before hidden collection. The scored schema is minimal and no syntax repair is allowed. Transport and harness retry counts are zero.

All failures remain in the intention-to-evaluate denominator. The primary rule assigns an invalid trial the minimum primary utility among the five actions in that world. Valid-only and do-nothing assignment are preregistered sensitivities. This rule is intentionally conservative and is frozen before hidden generation.
