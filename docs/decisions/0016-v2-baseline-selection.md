# ADR 0016: Nested selection of the public-information reference

Status: accepted.

Eligible classes are ridge action-value regression, histogram gradient boosting, polynomial ridge, and extra trees. All use only the frozen public summary. Selection uses nested leave-one-bundle-out validation; no world crosses a training/validation boundary without its bundle. Learning curves compare three and six training bundles and declare stabilization only when the absolute utility change is no more than 0.25% of the twenty-period analytic reference surplus. The chosen model is fit on all development bundles, checked on calibration through the pre-existing above-heuristic gate, serialized with its dependency version, and hash-pinned. The procedure is repeated for every welfare objective.
