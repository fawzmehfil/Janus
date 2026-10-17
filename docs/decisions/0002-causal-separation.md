# Decision 0002: causal mechanisms alter separate primitives

**Status:** Accepted  
**Phase:** 0B

The four hidden families are implemented through an explicit `MarketPrimitives`
boundary. Supply shortage alters marginal costs and physical offer limits;
genuine demand increase alters true buyer values; collusion alters seller
markup and withholding strategy without changing costs; momentum demand alters
submitted bids without changing true values.

Interventions resolve into exchange constraints, participant-facing transfer
parameters, real costs, and—only for a successfully detected collusive
scenario—disabled coordination. Audit outcomes are indexed disturbance-tape
draws and are not exposed in the public observation schema.

This representation makes causal-separation tests direct and prevents labels
from becoming decorative aliases for the same price process.

