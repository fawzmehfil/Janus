# Decision 0008: Hidden collection gate

- **Status:** accepted for preflight; paid run awaits explicit spending approval
- **Date:** 2026-10-19

The corrected five-world smoke returned five direct schema-valid responses with
the frozen prompt, schema, GPT-6 Luna, high reasoning, and a 2,048-token output
ceiling. This validates the operational setting but does not itself authorize
the 300-request hidden collection.

The hidden collector therefore freezes:

1. 100 opaque public worlds and three replicates per world;
2. exactly 300 primary requests and no syntax repairs;
3. no SDK retries, provider storage, or prompt-cache writes;
4. a one-cent-granularity cumulative cap no greater than `$1.00`;
5. a recommended `$0.84` cumulative cap, covering the exact conservative
   `$0.8307996` preflight bound;
6. a cap-specific confirmation phrase before any authenticated request;
7. resumability by logical trial ID, independent of SDK-version cache changes;
8. response-manifest commitment before unblinding; and
9. read access only to the public hidden manifest during collection.

The no-repair policy is valid under the charter's “at most one repair” rule and
is supported by the corrected smoke's five direct valid responses. It keeps the
provider-call count fixed, prevents repair traffic from displacing planned
worlds, and preserves every invalid response for intention-to-evaluate scoring.

