# Prospective cost envelope

No exact model versions or prices are frozen yet, so this report gives auditable token coefficients rather than a misleading point estimate.

The selected design uses 300 trials per model and four models: 1,200 hidden requests. The admission gate requires at least 100 representative trials per model: at least 400 development requests. A separate five-request-per-model operational smoke would add 20 requests. With a conservative planning allowance of 5,000 uncached input tokens and 512 billed output/reasoning tokens per request:

| Stage | Requests | Input tokens | Output tokens |
|---|---:|---:|---:|
| Five-request/provider smoke | 20 | 100,000 | 10,240 |
| Minimum admission | 400 | 2,000,000 | 204,800 |
| Hidden evaluation | 1,200 | 6,000,000 | 614,400 |
| Combined minimum | 1,620 | 8,100,000 | 829,440 |

For each exact model, cost is `(input tokens × input $/M + output tokens × output $/M) / 1,000,000`, plus any separately billed cache writes. The provider-specific preflight must use current official rates, returned token accounting, taxes if applicable, and a local hard ceiling before any live call. Cache writes are disabled unless their economics and semantics are explicitly frozen.

At illustrative rates spanning $0.10–$10 per million input tokens and $0.50–$50 per million output tokens, aggregate stage costs would span approximately:

| Stage | Illustrative low | Illustrative high |
|---|---:|---:|
| Smoke | $0.015 | $1.512 |
| Minimum admission | $0.302 | $30.240 |
| Hidden evaluation | $0.907 | $90.720 |
| Combined | $1.225 | $122.472 |

These ranges are neither quotes nor approvals. Immediately before each paid stage, freeze official dated prices and exact output/reasoning billing, compute a provider-specific estimate from the coefficients above, and require an explicit stage and cumulative cap. The local conservative maximum exposure is the sum of those approved hard ceilings; no paid stage has a default authorization. The final estimate should be materially narrower after exact dated model IDs and output allowances pass admission.
