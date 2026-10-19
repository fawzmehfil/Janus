# ADR 0014: Four-manifest seal and independent audit

Status: accepted.

A fresh CSPRNG seed is committed against the frozen v2 revision before hidden generation. All providers use the resulting public manifest. The four response manifests and raw-response hashes must be committed before private states or oracles are revealed. The repository provides a deterministic audit command but does not self-certify independence; an external auditor must run it from a separately controlled checkout.
