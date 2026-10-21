# ADR 0013: Ambiguity bundles are independent

Status: accepted.

Bundles are the sampling unit. Worlds and replicates are nested repeated measures. Primary uncertainty uses bundle-clustered resampling. Model comparisons are paired on bundles and Holm-corrected. The hierarchical report separates model, model family, mechanism family, OOD, bundle, world, and replicate variation and preregisters model-by-family and model-by-OOD interactions. Parameter counts, dense-versus-MoE architecture, reasoning mode, and serving configuration are reported as descriptive model attributes, not separately identified causal effects.
