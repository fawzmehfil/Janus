# Decision 0003: paired counterfactual recovery and epsilon-optimal sets

**Status:** Accepted  
**Phase:** 0C

Every action branch starts from an immutable clone of the same decision state.
Rollout `k` for every action uses the same indexed disturbance tape; buyer-value
noise, seller-cost noise, tie order, and investigation outcomes are keyed by
semantic identity rather than draw order. Action-dependent control flow cannot
shift unrelated future randomness.

The full-information oracle stores every action-value sample. Its primary
choice maximizes Monte Carlo mean utility, while action accuracy uses an
epsilon-optimal set. The epsilon for an action is the larger of 0.5% of the
analytic per-period attainable-surplus reference and twice the standard error
of paired differences from the estimated best action.

Investigation fines and one-time investigation costs apply only in the first
recovery period; detected collusive coordination remains disabled for the
entire horizon. Recurring price-ceiling, tax, and subsidy administration and
transfer semantics apply each period.
