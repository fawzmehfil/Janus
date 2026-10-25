# Public assets

This directory contains stable visual assets for the repository and public research
communication. The figures present frozen Janus v2 results; they do not change the analysis.

File | Meaning
--- | ---
`figures/janus-v2-utility-interval.*` | Bundle-mean utility difference and 95% intervals
`figures/janus-v2-accuracy-utility.*` | Near-optimal action rate versus utility difference
`figures/janus-v2-ambiguity-pair.*` | One matched pair with similar public histories

PNG files are sized for social and README use. SVG files are the preferred source for web,
print, or further layout work. [`figures/manifest.json`](figures/manifest.json) records the
frozen analysis hash and a SHA-256 digest for every exported file.

The underlying frozen numerical tables live in [`../reports/v2-publication`](../reports/v2-publication).
The matched-pair image is an illustrative hidden-set case, not an average effect.
