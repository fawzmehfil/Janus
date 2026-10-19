# ADR 0017: Forty-bundle split-replicate allocation

Status: accepted.

The v1 pilot variance decomposition compares 25 bundles × 4 worlds × 3 calls, 38 × 4 × 2, and a 40-bundle split allocation. At approximately 300 calls/model, the split allocation has the narrowest simulated interval. V2 therefore gives every one of 160 worlds one call, gives three members of every bundle a second call, and gives the fourth member a second call in twenty seed-selected bundles. This totals 300 calls/model while keeping bundles—not calls—as the independent unit.

The expected 95% half-width is 53.5 utility units and the two-sided 80%-power minimum detectable model difference is 76.5. Confirmatory conclusions narrower than that effect are not supported by the design.
