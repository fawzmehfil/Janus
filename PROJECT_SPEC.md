# Janus — Project Specification and Research Charter

**Status:** Source of truth; pre-implementation research design  
**Version:** 0.1.0  
**Date:** 2026-10-16  
**Project type:** Synthetic, research-oriented LLM decision evaluation  
**Current phase:** Design only; no simulator or paid model calls are authorized yet

## 0. Authority and change control

This document is the authoritative specification for Janus. It replaces the initial prose brief as the operational source of truth. The brief remains useful provenance, but implementation decisions, experimental claims, and release gates are governed here.

The terms **MUST**, **SHOULD**, and **MAY** are normative:

- **MUST** means required for a valid release.
- **SHOULD** means expected unless a documented reason justifies a deviation.
- **MAY** means optional and out of the critical path.

Before the hidden evaluation is generated, changes may be made through ordinary version control with a short decision record. Once the benchmark is frozen, any change to market mechanics, observations, prompts, action semantics, utility, scenario distributions, parsing, or primary analyses MUST increment the benchmark version and invalidate/regenerate the hidden set. Cosmetic report changes do not invalidate it.

No paid API call may be made until the local simulator, counterfactual evaluator, baselines, invariants, and budget enforcement pass their release gates.

---

## 1. Executive definition

Janus tests whether an LLM acting once as a regulator can choose a welfare-improving intervention in a synthetic single-commodity market when different hidden causes can produce similar public histories. The name evokes one visible market with multiple hidden faces.

The regulator receives only a versioned table of public, aggregate observations. It returns one structured diagnosis and one action. Scripted local participants—not LLMs—then respond to the action. Every allowed action is evaluated from the same cloned decision state under paired future disturbances. No LLM judge and no human grader determines correctness.

The benchmark's core question is:

> Given only public evidence, can an LLM select an intervention with low economic regret when the same apparent market disruption may arise from a real supply shock, a real demand shock, seller collusion, or momentum-driven bidding—and does its performance survive strategic response and pre-specified distribution shift?

This is an evaluation of decision-making inside a stylized simulator. It is **not** evidence that a model should regulate a real market, and it MUST NOT be presented as a policy recommendation for any real economy.

### Primary contribution

The intended contribution is a small, inspectable benchmark that combines:

1. a controlled market with analytically checkable accounting;
2. latent causal mechanisms with measured observational overlap;
3. one-shot policy choice under partial observability;
4. local counterfactual evaluation of every allowed action;
5. explicit separation of information limits from decision failures; and
6. tail-risk, calibration, and distribution-shift analysis without an LLM judge.

The project succeeds only if the environment distinguishes random, constant, heuristic, partial-information, LLM, and full-information policies. A polished harness around an uninformative simulator is a failed project.

---

## 2. Expert analysis before implementation

This section applies the requested “deeper, not merely bigger” standard. It states what a strong research engineer or experimental economist would add, which raw materials they would inspect, which methods they would use, and what independent evidence would establish correctness.

### 2.1 What a top professional would do that the initial brief did not explicitly require

#### Define the estimand before the simulator

The object being estimated is not generic “economic intelligence.” It is the expected loss of a model policy over a declared synthetic scenario distribution:

$$
\mathcal{R}(\pi)=
\mathbb{E}_{B,Z,S,O,Y}\left[Q(S,a^*_{\mathrm{FI}})-Q(S,\pi(O))\right],
$$

where $B$ is an ambiguity bundle, $Z$ is the hidden cause, $S$ is the full decision state, $O=g(S)$ is the public observation, $Y$ represents future disturbances, $Q(S,a)$ is expected post-decision regulator utility, and $a^*_{\mathrm{FI}}$ is the best action with full state information.

This quantity is useful, but it mixes two sources of loss:

- **information loss:** the best achievable policy using $O$ may be worse than a full-information oracle;
- **decision loss:** the model may be worse than a strong policy that uses the same $O$.

Accordingly, the project MUST report both full-information regret and performance relative to a partial-information optimization baseline. Full-information regret alone MUST NOT be described as a model reasoning failure.

#### Treat identifiability as a measured property

“Superficially similar histories” is not an implementation detail; it is the benchmark's central construct. Scenario families MUST be calibrated so their observation distributions overlap, and that overlap MUST be audited with held-out diagnostic classifiers and nearest-neighbor checks.

If labels are nearly perfectly predictable, the task is ordinary classification with decorative economics. If labels are completely indistinguishable and the optimal action differs, diagnosis accuracy is bounded by construction and regret includes unavoidable ambiguity. The target is a documented mixture of learnable and irreducibly ambiguous cases.

#### Separate diagnosis from decision quality

A wrong diagnosis can lead to a robust action, and a correct diagnosis can lead to a harmful action. Diagnosis accuracy is therefore secondary. Primary claims concern expected utility and regret. The report MUST include the diagnosis-to-action confusion matrix and examples of:

- wrong diagnosis, good action;
- correct diagnosis, bad action;
- high confidence, harmful action; and
- conservative uncertainty, robust action.

#### Make the regulator's objective explicit

An action cannot be “correct” without a welfare criterion. Taxes, subsidies, and fines are transfers, not automatically destroyed resources. Price stability, producer survival, consumer protection, fiscal cost, and allocative efficiency can conflict. The benchmark MUST define a primary economic objective and pre-register alternative normative objectives. It MUST report the metric vector even when it also reports regret.

#### Engineer counterfactual comparability

Each action MUST be evaluated from an exact clone of the decision state using the same exogenous disturbance tape. Random-number consumption MUST be indexed by period, entity, and disturbance type so one action cannot shift later draws merely by taking a different code path. This common-random-number design reduces variance and makes action differences attributable to the intervention.

#### Distinguish model repetitions from independent scenarios

Three model calls on one scenario measure sampling variability; they do not create three independent market scenarios. Inference MUST cluster at the scenario/ambiguity-bundle level. Claims based on 100 scenarios × 3 calls MUST NOT report $n=300$ as if all observations were independent.

#### Use negative controls and falsification tests

The simulator MUST include tests in which:

- changing only the hidden label has no effect;
- a zero-magnitude intervention equals “do nothing”;
- transfers cancel in total welfare when distortion and real costs are disabled;
- future random tapes are identical across action branches;
- shuffling diagnosis labels destroys diagnosis accuracy but not mechanically computed welfare; and
- the oracle cannot access future realizations, only their declared distribution.

#### Pre-register the hidden evaluation

Before hidden generation, freeze and hash the environment version, generator distribution, action magnitudes, observation serializer, prompt, schema, utility definitions, analysis code, model settings, and planned comparisons. Store a commitment manifest. Hidden outcomes MUST not be used for prompt editing, threshold tuning, heuristic design, or scenario acceptance.

### 2.2 Raw material an expert would inspect first

The work should begin from primary artifacts rather than summaries:

1. The full papers, appendices, prompts, parameter tables, released code, configs, and trajectories for the three named foundations.
2. The original experimental-market specifications behind the double-auction methodology, especially Smith's valuation/cost schedules and Gode–Sunder constrained zero-intelligence controls.
3. Exact order, trade, cash, inventory, valuation, cost, and random-draw logs from this simulator—not only aggregate charts.
4. The frozen scenario-generator code and parameter distributions, including rejected candidate scenarios and acceptance reasons.
5. The observation payload actually sent to each model, byte for byte.
6. Raw API responses, parsed decisions, refusals, schema errors, usage fields, model IDs, provider request IDs, timestamps, and contemporaneous price tables.
7. Counterfactual action-value samples and their paired differences, not only the selected oracle action.
8. The analysis notebook/script that produces every table and figure from immutable artifacts.

The three project foundations are real but support different claims:

- [Agent Bazaar](https://arxiv.org/abs/2605.17698) frames economic alignment through market stability, integrity, welfare, and profitability, and demonstrates systemic failure modes under partial observability. It evaluates LLM market participants; this project instead evaluates an LLM regulator over scripted participants.
- [Competitive Market Behavior of LLMs](https://arxiv.org/abs/2609.02580) uses a controlled double auction with theoretical and human benchmarks, repeated seeds, quantifiable efficiency, and released code. This project borrows its preference for clean, falsifiable market mechanics, not its use of LLM traders.
- [LLM Economist](https://arxiv.org/abs/2507.15815) places an LLM planner above an economic environment and makes the planner's social-welfare objective explicit. This project narrows that idea to a single decision and replaces endogenous LLM workers with reproducible scripted agents.

Additional primary anchors are [Smith (1962), “An Experimental Study of Competitive Market Behavior”](https://doi.org/10.1086/258609) and [Gode & Sunder (1993), “Allocative Efficiency of Markets with Zero-Intelligence Traders”](https://faculty.sites.iastate.edu/tesfatsi/archive/tesfatsi/ZITraders.GodeSunder.JPE1993.pdf).

### 2.3 Methods and tools an expert would use

#### Economic and simulation methods

- A periodic sealed-bid double auction with explicit valuations, marginal costs, inventory, and deterministic clearing rules.
- A partially observable stochastic decision formulation with a latent cause and a single regulator action.
- Paired counterfactual rollouts with common random numbers.
- A full-information Monte Carlo oracle and an observation-only fitted policy baseline.
- Constrained scenario generation followed by overlap, difficulty, and oracle-action-balance audits.
- Analytic micro-markets whose equilibrium quantity, price interval, surplus, and intervention effects can be calculated by hand.
- Welfare accounting that distinguishes transfers from real resource costs.

#### Statistical methods

- Paired differences on identical scenarios for policy/model comparisons.
- Hierarchical or cluster bootstrap intervals resampling ambiguity bundles first, with model repetitions nested inside worlds.
- Multiclass Brier score and log loss for diagnosis probabilities, plus reliability plots with uncertainty.
- Agreement rate and action entropy across repeated model calls.
- Conditional value at risk (CVaR) for tail harm; raw sample maxima only as descriptive statistics.
- Sensitivity analysis over normative welfare assumptions and counterfactual rollout counts.
- Power/precision analysis after calibration; hidden-set size is justified by interval width, not by treating repetitions as extra scenarios.

#### Engineering tools

- Python with a locked environment and a `src/` package layout.
- NumPy random generators with named `SeedSequence` substreams or an equivalent indexed disturbance-tape abstraction.
- Pydantic or JSON Schema for configs, observations, decisions, manifests, and stored results.
- Pytest plus property-based tests for invariants and stateful transitions.
- Parquet for typed tabular trajectories and JSONL for immutable request/response records.
- A single command-line experiment runner; no notebook-only execution path.
- Content hashes for configs, prompts, code revision, model settings, and artifact manifests.

### 2.4 What would independently show the result is right

The result is credible only when evidence other than author judgment supports it:

1. **Analytic agreement:** hand-solvable toy markets match the engine exactly.
2. **Invariant tests:** conservation, budget, inventory, feasibility, and no-lookahead properties hold over randomized traces.
3. **Differential tests:** an intentionally slow brute-force clearing implementation agrees with the production clearing engine on small markets.
4. **Replayability:** a saved state, config, and seed manifest reproduce the same trajectory and metrics bit for bit on the supported platform.
5. **Branch equality:** cloned counterfactual branches share every exogenous draw and differ only through action-mediated transitions.
6. **Baseline ordering:** the full-information oracle weakly dominates every fixed action in estimated utility; a useful heuristic beats uniform random on calibration data; no single constant policy trivializes the benchmark.
7. **Signal validity:** known beneficial and harmful intervention micro-cases receive the expected regret signs.
8. **Difficulty audit:** held-out classifiers demonstrate nontrivial but imperfect latent-cause predictability.
9. **Blinded analysis:** hidden labels and oracle outcomes remain unavailable until model responses are committed.
10. **Artifact audit:** an independent person can regenerate the paper tables from the release bundle without API access.
11. **Sensitivity stability:** headline direction does not depend on one arbitrary welfare weight, one oracle seed count, or one random hidden-set draw.
12. **Failure transparency:** invalid outputs, refusals, ties, exclusions, crashes, and budget stops remain in denominators under predeclared rules.

---

## 3. Research claims, hypotheses, and limits

### 3.1 Confirmatory questions

1. Does an LLM achieve lower expected full-information regret than uniform random and constant-action baselines?
2. How much of the gap to the full-information oracle is explained by partial observability versus model decision quality?
3. Does performance degrade under pre-specified parameter and behavioral distribution shifts?
4. Are stated diagnosis probabilities and action confidence calibrated to observed correctness?
5. Does average performance conceal concentrated downside in the worst scenarios?

### 3.2 Pre-registered hypotheses

- **H1:** The primary LLM policy has lower mean normalized regret than uniform random.
- **H2:** The primary LLM policy has lower mean normalized regret than always-do-nothing.
- **H3:** OOD scenarios increase mean regret and lower-tail utility relative to in-distribution scenarios.
- **H4:** Higher stated action confidence is associated with a higher probability that the chosen action is epsilon-optimal.
- **H5:** The model's diagnosis accuracy exceeds chance, but diagnosis accuracy explains only part of action regret.

All other findings—including per-family rankings, prompt-ablation effects, and model-family narratives—are exploratory unless separately pre-registered before hidden evaluation.

### 3.3 Permitted conclusion

A valid conclusion may say that a named, pinned model under a named prompt and sampling configuration performed a certain way on version X of this synthetic benchmark distribution.

It may not claim:

- competence to regulate real markets;
- causal knowledge of real-world inflation, collusion, shortages, or speculation;
- superiority of a provider's models in general;
- human-level economic judgment without a human benchmark; or
- robustness beyond the tested simulator and shift axes.

---

## 4. Formal task definition

### 4.1 Decision process

An episode has a latent scenario $Z\in\mathcal{Z}$, full market state $S_t$, public observation $O_t=g(S_t)$, scripted participant actions, exogenous disturbance tape $\omega$, and one regulator decision at time $t_d$.

The latent families are:

$$
\mathcal{Z}=\{\text{supply shortage},\text{demand increase},\text{seller collusion},\text{momentum demand}\}.
$$

Their stable diagnosis IDs are `supply_shortage`, `demand_increase`, `seller_collusion`, and `momentum_demand`.

The regulator receives the serialized public history $O_{0:t_d}$ and chooses one action $a\in\mathcal{A}$. The action persists through the recovery horizon. There is no dialogue and no second decision in the MVP.

### 4.2 Episode timing

Phase 0 defaults are:

- $T_{pre}=20$ stable pre-shock periods;
- shock begins at $t=20$;
- $T_{obs}=10$ periods after shock onset;
- decision at $t_d=30$; and
- $T_{rec}=20$ post-decision periods.

These values are provisional until calibration, but once frozen they MUST be identical across evaluated policies. The model is told that it is acting after an observation window, but not the hidden cause or exact onset mechanics.

### 4.3 Unit of analysis

- A **world** is one parameterized latent scenario and pre-decision state.
- A **bundle** is a matched set containing one world from each latent family with similar public histories.
- A **model replicate** is one independent inference request for one world under the same prompt/settings.
- A **counterfactual rollout** is one local future simulation for one state-action pair and one future disturbance tape.

The scenario or bundle—not the model replicate—is the primary independent statistical unit.

---

## 5. Phase 0 market model

### 5.1 Deliberate abstraction

The market is a discrete-time, single-commodity, periodic sealed-bid double auction. This institution is chosen because allocation, feasibility, and surplus are transparent. It is not meant to mimic a specific real market.

At each period:

1. consumers submit unit bids;
2. sellers submit unit asks up to available inventory/capacity;
3. the exchange sorts bids descending and asks ascending;
4. feasible trades are selected; and
5. inventory, cash, public statistics, and participant internal states update.

No participant makes an LLM call.

### 5.2 Buyers

Buyer $i$ has a true private value $v_{i,t}$, submitted bid $b_{i,t}$, risk/urgency coefficient $r_i$, and scripted behavior type. True value and submitted bid are distinct by design.

Base consumers draw values from a configured distribution around a stable demand schedule. A constrained bid obeys a configurable rule such as

$$
b_{i,t}=\max(0, v_{i,t}+u_{i,t}+m_{i,t}),
$$

where $u_{i,t}$ is an urgency/risk adjustment and $m_{i,t}$ is a behavioral distortion. Standard competitive buyers have small zero-mean noise. Momentum buyers use recent public returns in $m_{i,t}$ while their true $v_{i,t}$ does not rise correspondingly.

This separation is essential: a genuine demand shock raises value, while momentum demand raises willingness to bid without creating equal fundamental surplus.

### 5.3 Sellers

Seller $j$ has unit-level marginal costs $c_{j,k,t}$, inventory $I_{j,t}$, replenishment capacity $K_{j,t}$, cash $C_{j,t}$, and markup/withholding rules. Each available unit produces an ask

$$
a_{j,k,t}=c_{j,k,t}+\mu_{j,t}+\epsilon_{j,k,t}.
$$

Competitive sellers use a small markup. Adaptive profit-maximizing sellers update markups from sales and inventory. Collusive sellers coordinate a shared markup floor and may withhold a fraction of otherwise available units. Bankruptcy occurs only under an explicit cash-flow rule and carries a documented real exit cost; it is not inferred merely from low profit.

### 5.4 Clearing

Let $b_{(1)}\ge\cdots$ be eligible bids and $a_{(1)}\le\cdots$ eligible asks after intervention-specific adjustments. The traded quantity is

$$
q_t=\max\{q:b_{(q),t}\ge a_{(q),t}\}.
$$

Absent a price cap, all matched units trade at the uniform midpoint price

$$
p_t=\frac{b_{(q_t),t}+a_{(q_t),t}}{2}.
$$

If no trade occurs, price is missing—not silently carried forward. A separate last-price field may be shown to participants and the regulator. Ties are resolved with a pre-sampled permutation from the disturbance tape.

The exact clearing convention is a construct choice. Phase 0 MUST include an ablation showing that headline results are not solely an artifact of midpoint pricing; a pay-as-bid or alternative valid uniform-price rule is an acceptable robustness check after the primary engine is stable.

### 5.5 Baseline calibration fixture

The provisional no-shock market has 40 potential one-unit buyers and eight sellers with five one-unit marginal-cost slots each. Before behavior noise and participant assignment, the ordered schedules are

$$
v_{(i)}=140-2(i-1),\quad i=1,\ldots,40,
$$

$$
c_{(k)}=60+2(k-1),\quad k=1,\ldots,40.
$$

They imply $q^*=21$, $p^*=100$, and attainable per-period surplus 840 under the midpoint rule: the 21st value and cost both equal 100 while the 22nd pair does not cross. This fixture supplies an exact pre-shock reference and a hand-checkable accounting target. Production slots are assigned to sellers through a seeded balanced mapping; scenario concentration changes those capacity shares without changing the accounting rules.

The stochastic benchmark perturbs this fixture with bounded arrival, value, cost, and order noise. Pre-shock acceptance tests require the perturbed process to remain stationary around the analytic reference. Initial inventory is provisionally one period of seller capacity and initial cash is ten times the analytic reference price; calibration may change these values before freeze.

### 5.6 Cash, inventory, and replenishment

- Buyers pay transaction price plus buyer-side taxes.
- Sellers receive transaction price plus per-unit supply subsidy.
- Taxes, subsidies, and fines enter a government account.
- Inventory cannot be negative.
- Replenishment occurs after trading according to seller capacity and marginal-cost schedules.
- Unsold nonperishable inventory carries forward with a real holding cost.
- Participant cash may be negative only if the configuration provides explicit credit; otherwise an infeasible order is rejected.

Every flow MUST be recorded in a double-entry-style ledger sufficient to reconstruct consumer surplus, producer surplus, government balance, and real resource costs.

---

## 6. Hidden causal families

Each family changes a different primitive. Merely assigning different labels to the same price process is forbidden.

### 6.1 Genuine supply shortage

An exogenous shock raises marginal costs, reduces replenishment capacity, or destroys inventory. Family parameters include cost shift, capacity loss, duration, affected-seller share, and recovery rate. A pure hard-capacity shortage may make subsidies ineffective; the family SHOULD include a documented mix of cost and capacity shocks so the optimal action is not hard-coded.

### 6.2 Temporary genuine demand increase

Buyer arrival intensity and/or true values rise temporarily. Because $v_{i,t}$ rises, additional trades may create real surplus even at higher prices. Parameters include arrival shift, valuation shift, duration, demand elasticity, and consumer risk heterogeneity.

### 6.3 Seller collusion

A coalition coordinates markups and optional supply withholding while costs and true demand remain near baseline. Parameters include coalition share, market concentration, common markup, withholding fraction, detection discipline, and defection probability. The public history does not reveal coalition membership or strategy labels.

### 6.4 Momentum-driven demand

A share of buyers adds a trend-following term to submitted bids based on lagged returns and/or volume, without a matching rise in true consumption value. Parameters include speculator prevalence, trend window, feedback coefficient, position limit, and decay. Phase 0 may treat these participants as behaviorally distorted final buyers; a later resale/inventory model is out of scope unless needed by validation.

### 6.5 Provisional parameter domain

Development candidates use bounded, interpretable ranges rather than unconstrained randomization. Initial ranges are:

| Parameter | Provisional development range |
|---|---:|
| potential buyers per period | 30–60 |
| active sellers | 5–12 |
| shock duration | 5–20 periods |
| cost or value shift | 10%–40% of $p_0$ |
| physical capacity loss | 10%–40% |
| aggregate inventory cover | 0.5–2.0 baseline periods |
| target seller HHI | 0.10–0.35 |
| urgency/risk bid adjustment | 0%–20% of $p_0$ |
| colluding seller share | 25%–75% |
| collusive markup | 10%–40% of $p_0$ |
| collusive withholding | 0%–40% of capacity |
| momentum-buyer share | 10%–50% |
| momentum bid distortion cap | 10%–50% of $p_0$ |

Supply and demand elasticity are controlled through the slopes and dispersion of the value/cost schedules rather than entered as decorative labels. Candidate parameters SHOULD be drawn with stratified or Latin-hypercube coverage, then filtered only by frozen feasibility and difficulty criteria. Correlated combinations that cannot sustain a stable pre-shock market are rejected with a recorded reason.

Calibration may narrow these ranges once to obtain stable pre-shock behavior, meaningful action diversity, and the desired observation overlap. OOD ranges are then declared outside the final development envelope while remaining within simulator safety bounds. The final config—not this prose table—records the exact distributions and inclusivity of endpoints.

### 6.6 Observational overlap construction

Development, calibration, and hidden scenarios are organized into four-world ambiguity bundles. Candidate worlds are generated independently within each family, then matched on a predeclared public-summary vector containing:

- normalized price path and return path;
- volume and unfilled-demand path;
- aggregate available supply and inventory;
- volatility and spread, if defined;
- concentration and active-firm count; and
- pre-decision bankruptcies and prior interventions.

Matching MUST use development/calibration data only to set tolerances. Hidden candidates are accepted by the frozen algorithm without manual cherry-picking.

Overlap audits MUST include:

- standardized mean differences by family;
- cross-family nearest-neighbor distance;
- a multinomial logistic baseline;
- a nonlinear tree/boosting diagnostic classifier; and
- label accuracy, macro-F1, and log loss on held-out calibration worlds.

The acceptance target is nontrivial but imperfect predictability. Provisional calibration guidance is macro-F1 between 0.35 and 0.70 for four balanced labels; the final numeric gate MUST be frozen based on calibration before hidden generation. The benchmark report MUST state that this threshold is a design criterion, not a natural property of real markets.

---

## 7. Public observation contract

The regulator may receive only public or aggregate fields:

- period index;
- transaction price or explicit no-trade marker;
- transaction volume;
- aggregate available supply;
- unfilled submitted demand;
- aggregate inventories;
- best bid, best ask, and spread when the mechanism defines them;
- market concentration (HHI and active-firm count);
- firm exits/bankruptcies;
- recent realized volatility;
- government intervention currently in force; and
- a pre-shock reference price, volume, and inventory scale.

The observation MUST NOT expose:

- hidden family or family-specific parameter names;
- true private values;
- marginal costs;
- participant type or strategy;
- coalition membership;
- future disturbances;
- counterfactual outcomes; or
- oracle choices.

All numeric observations SHOULD be expressed both in interpretable units and relative to pre-shock baselines where appropriate. Serialization order, rounding, missing-value representation, and explanatory text are part of the benchmark and MUST be versioned and hashed.

The model receives the same information content regardless of provider. Provider-specific wrappers MAY differ only where an API requires them, and those differences MUST be documented.

---

## 8. Intervention contract

Phase 0 has exactly five mutually exclusive actions. Magnitudes are fixed relative to pre-shock reference values so scenarios with different price scales remain comparable.

The stable action IDs are `do_nothing`, `price_ceiling`, `transaction_tax`, `supply_subsidy`, and `collusion_investigation`.

### 8.1 Do nothing

No new rule is applied. Existing market dynamics continue.

### 8.2 Price ceiling

A maximum transaction price of $1.10\,p_0$ applies for the recovery horizon, where $p_0$ is the median pre-shock transaction price. The capped traded quantity is $q_t=\max\{q:a_{(q),t}\le\min(b_{(q),t},\bar p)\}$. Feasible matched units trade at

$$
p_t=\min\left(\frac{b_{(q_t),t}+a_{(q_t),t}}{2},\bar p\right),
$$

with $a_{(q_t),t}\le p_t\le b_{(q_t),t}$ and $p_t\le\bar p$. Excess demand is rationed by the pre-sampled tie order. Enforcement has a small real per-period administrative cost.

### 8.3 Buyer-side transaction tax

A 10% ad valorem tax applies to buyers. If the seller receives $p$, the buyer pays $1.10p$. Buyers adapt by converting their gross willingness to pay into an eligible bid. Tax revenue is a transfer to government; only behavioral distortion and real administration costs reduce total surplus.

### 8.4 Supply subsidy

Sellers receive $0.15p_0$ per unit sold. Their eligible asks reflect the subsidy, and higher-cost marginal units may enter when capacity permits. The subsidy cannot create units beyond physical capacity. Fiscal transfers cancel in unweighted social surplus except for the predeclared marginal cost of public funds and real administration costs.

### 8.5 Investigate and penalize possible collusion

The regulator initiates a market-wide investigation with a real administrative and compliance cost. Detection uses a pre-sampled audit outcome with provisional sensitivity 0.85 and false-positive probability 0.05. If a collusive coalition is detected, its coordinated markup/withholding rule is disabled for the recovery horizon and a fine is levied. Fines are transfers; investigation and false-positive disruption are real costs. The model never sees the audit draw.

### 8.6 Calibration and freezing of magnitudes

The provisional numbers above exist to make Phase 0 executable; they are not economic estimates. They MAY be changed during development only to satisfy validation criteria such as action diversity, non-dominance, and plausible intervention trade-offs. Once calibration begins, changes require a logged design decision. Before hidden generation, all values MUST be frozen.

Potential later actions—circuit breakers, disclosure rules, position limits, inventory limits, and entry subsidies—are explicitly out of the MVP. Adding them creates a new benchmark version.

---

## 9. Economic accounting and oracle objective

### 9.1 Period accounting

For the set of matched trades $\mathcal{M}_t$ in period $t$:

$$
CS_t=\sum_{(i,j,k)\in\mathcal{M}_t}\left(v_{i,t}-\text{buyer payment}_{i,t}\right),
$$

$$
PS_t=\sum_{(i,j,k)\in\mathcal{M}_t}\left(\text{seller receipt}_{j,t}+\text{subsidy}_{j,t}-c_{j,k,t}\right)-\text{fines paid}_t,
$$

$$
G_t=\text{taxes}_t+\text{fines}_t-\text{subsidies}_t.
$$

Let $C^{\mathrm{real}}_t$ collect explicit real administration, compliance, holding, exit, and adjustment costs. Unweighted net social surplus is $CS_t+PS_t+G_t-C^{\mathrm{real}}_t$. Pure monetary transfers cancel because taxes and fines reduce private surplus when they raise the government balance, while subsidies do the reverse. Lost mutually beneficial trades appear as lower realized surplus relative to feasible alternatives; they are not added again as an arbitrary “shortage penalty.”

### 9.2 Primary oracle utility

The primary regulator utility is discounted net social surplus over the recovery horizon:

$$
U_{\mathrm{primary}}(S,a;\omega)=
\sum_{t=t_d}^{T}
\delta^{t-t_d}
\left[CS_t+PS_t+G_t-C^{\mathrm{real}}_t-C^{\mathrm{funds}}_t\right].
$$

Phase 0 uses $\delta=1$. The financing term is $C^{\mathrm{funds}}_t=\lambda_F\max(0,-G_t)$: a marginal excess burden of public funds $\lambda_F=0.20$ applied only to net government outlays, with sensitivity values $\{0,0.20,0.50\}$.

This objective intentionally does not add separate price-volatility, volume, or recovery-time penalties that would double-count welfare consequences already induced through participant values and costs.

### 9.3 Secondary normative objectives

The release MUST also recompute oracle choices and regret under:

- **consumer-protection:** distributional weight 2 on consumer surplus and 1 on producer surplus;
- **producer-continuity:** explicit additional real exit cost and weight on survival;
- **stability-aware:** a predeclared penalty on normalized post-decision log-price volatility; and
- **fiscal-neutral:** $\lambda_F=0$.

Headline conclusions are robust only if their direction survives these alternatives. Otherwise, dependence on normative choice is itself a primary finding.

### 9.4 Full-information counterfactual oracle

At the decision point, clone the complete state. For every action, estimate

$$
Q(S,a)=\mathbb{E}_{\omega}[U(S,a;\omega)]
$$

using $K_{oracle}=256$ paired future disturbance tapes by default. The same tape index is used for all actions. Select the action with the highest estimated $Q$.

An action is **epsilon-optimal** if its estimated value is within

$$
\epsilon=\max(0.5\%\text{ of reference attainable surplus},\;2\,SE(\Delta Q))
$$

of the best action. Ties MUST be represented as an optimal set; deterministic alphabetical tie-breaking may be used for storage but not for action-accuracy scoring.

Oracle Monte Carlo size MUST be validated by rerunning a representative calibration subset at 2× and 4× $K_{oracle}$. Oracle action changes and regret deltas are reported. The oracle uses the state and disturbance distribution, never the realized future of the scored trajectory.

### 9.5 Partial-information optimization baseline

A strong observation-only baseline MUST be trained entirely on local simulator data from the development distribution. It receives the exact serialized public observation, or numeric features parsed only from that same rounded payload, and predicts each action's counterfactual value. Candidate methods are regularized linear regression and gradient-boosted trees; the simpler method is preferred when performance is comparable.

Model selection uses development data; calibration estimates its out-of-sample performance; the fitted method is frozen before the hidden set. It may not access hidden state or labels at inference.

Report the approximate decomposition:

$$
\text{information gap}=U(a^*_{FI})-U(a_{PI}),
$$

$$
\text{decision gap}=U(a_{PI})-U(a_{LLM}).
$$

Because $a_{PI}$ is fitted rather than a proven Bayes-optimal policy, “information gap” MUST be labeled an estimate/lower bound on the attainable-observation gap, not an exact decomposition.

---

## 10. Model task and output schema

### 10.1 Prompt content

The frozen prompt MUST state:

- the regulator role and synthetic nature of the task;
- the public observation table and units;
- exact allowed actions and operational semantics;
- the primary objective in plain language;
- the four possible diagnosis labels;
- that hidden causes may be observationally ambiguous;
- that one action will be applied for the recovery horizon; and
- the strict output schema.

It MUST NOT include exemplars drawn from hidden scenarios, family-specific lexical cues, private state, or counterfactual values.

The primary benchmark uses one zero-shot prompt. A structured-economic-analysis prompt variant MAY be studied on development/calibration data, but it becomes a separate pre-registered condition if evaluated on hidden data.

### 10.2 Strict structured response

Use provider-supported structured outputs when available and validate the same JSON Schema locally for every provider:

```json
{
  "diagnosis": "supply_shortage",
  "diagnosis_probabilities": {
    "supply_shortage": 0.55,
    "demand_increase": 0.20,
    "seller_collusion": 0.15,
    "momentum_demand": 0.10
  },
  "action": "supply_subsidy",
  "confidence": 0.72,
  "rationale": "Aggregate supply and inventory fell before volume contracted; a subsidy may activate higher-cost capacity without imposing broad rationing."
}
```

Rules:

- diagnosis and action are enums;
- all four diagnosis probabilities are required, lie in ([0,1]), and sum to 1 within (10^{-6});
- `diagnosis` equals an argmax probability, allowing ties;
- `confidence` is the claimed probability that the selected action is epsilon-optimal in the hidden world;
- rationale is at most 100 words; and
- no free-form text may appear outside the object.

The rationale is stored for qualitative failure analysis but is never graded by an LLM. Hidden reasoning traces are neither requested nor stored. Reasoning-token counts may be logged when the provider exposes them.

### 10.3 Invalid responses

Provider refusal, schema failure, timeout, and empty output are distinct statuses. At most one repair request is allowed, using a fixed schema-repair prompt containing no outcome feedback. The intention-to-evaluate analysis applies a predeclared `do_nothing` fallback to an unrepaired response and reports the invalid rate; sensitivity analyses score invalid output as the worst allowed action for that world and report valid-only performance. All denominators are mandatory.

---

## 11. Baselines

Baselines MUST be implemented and validated before any paid model call:

1. **Uniform random:** samples each of the five actions with probability 0.2 from a seeded policy RNG.
2. **Every constant action:** five separate policies, including always-do-nothing. This is more interpretable than an undefined “always intervene aggressively” baseline.
3. **Handwritten heuristic:** a frozen rule using only public observations; thresholds tuned on development only.
4. **Constrained zero-intelligence diagnostic/control:** where applicable, verifies that market rules—not sophisticated participant optimization—can produce sensible allocation.
5. **Observation-only optimization baseline:** fitted action-value predictor described above.
6. **Full-information oracle:** upper bound using the complete decision state.

Optional later baselines include a small open-weight language model and an ablated LLM that sees only the last period, but these do not block Phase 0.

Benchmark validity requires:

- more than one action is oracle-optimal on calibration scenarios;
- no constant action attains near-oracle performance everywhere;
- random, heuristic, partial-information, and full-information policies are measurably separated; and
- the heuristic is not given hidden information through feature engineering.

If these conditions fail, fix the environment or narrow the claims; do not proceed to cross-model evaluation.

---

## 12. Scenario splits and distribution shift

### 12.1 Bundle-preserving split

The original counts map cleanly to ambiguity bundles:

- **Development:** 10 bundles × 4 causes = 40 worlds.
- **Calibration:** 15 bundles × 4 causes = 60 worlds.
- **Hidden final:** 25 bundles × 4 causes = 100 worlds.

Every split is balanced by cause. Bundle members stay in the same split. These counts define a credible pilot, not automatically a high-powered paper. After calibration, conduct a precision analysis using bundle-level variance. If pre-registered confidence-width targets cannot be met with 25 hidden bundles, increase the number of bundles before freeze or explicitly limit the claims.

### 12.2 Hidden-set strata

The hidden set contains predeclared in-distribution and OOD bundles. Provisional allocation is 15 ID bundles and 10 OOD bundles.

OOD tests change one dimension at a time where feasible:

- severity/duration values outside the development interval but inside safe simulator bounds;
- novel combinations of elasticity and concentration;
- participant-count changes while preserving observation format;
- stronger momentum or collusion response; and
- inventory/capacity regimes not seen during prompt iteration.

OOD status is not shown to the model. A shift MUST NOT alter field names, add obvious lexical cues, or silently change the regulator objective.

### 12.3 Generation and concealment

Before hidden generation:

1. freeze code, configs, prompt, schemas, and analysis plan;
2. record their hashes and git commit;
3. draw hidden generator seeds from an auditable process;
4. commit a hash of the seed manifest; and
5. generate and encrypt or access-restrict hidden state, labels, and oracle results.

Model response files are committed before hidden labels/outcomes are joined for analysis. The hidden set is used once for the versioned final result. Debugging on a hidden failure creates a future benchmark version, not an amended score.

---

## 13. Metrics

### 13.1 Primary metrics

- Mean normalized full-information regret.
- Median normalized regret.
- Probability of selecting an epsilon-optimal action.
- Mean utility difference versus random and always-do-nothing, using paired scenarios.
- CVaR at the worst 10% of utility outcomes.

Regret normalization uses pre-shock attainable surplus over the recovery horizon, a scenario-local positive scale frozen before seeing model outputs. Sample-dependent min-max normalization is forbidden.

### 13.2 Economic metric vector

Always report separately:

- net total surplus;
- consumer surplus;
- producer surplus;
- government net balance and real administration cost;
- price level and log-price volatility;
- efficient demand left unfilled;
- transaction volume;
- inventory and stockout periods;
- firm exits/bankruptcies;
- market concentration;
- recovery time under a predeclared threshold; and
- harm from unnecessary intervention relative to do-nothing.

No single composite “economic alignment score” is the sole result.

### 13.3 Diagnostic and behavioral metrics

- diagnosis accuracy and macro-F1;
- diagnosis confusion matrix;
- multiclass Brier score and clipped log loss;
- action-confidence Brier score against epsilon-optimality;
- reliability diagrams;
- within-world action agreement across model replicates;
- action entropy and diagnosis-probability variance; and
- invalid/refusal rate.

Expected calibration error may be reported only with binning and uncertainty disclosed; it is not a primary metric at this sample size.

### 13.4 Tail and worst-case reporting

Report maximum regret descriptively, but use CVaR/worst-decile mean as the principal tail statistic because a single sample maximum is unstable. Include the five highest-regret trajectories per model as auditable case studies with all public inputs and local counterfactual outcomes.

### 13.5 Statistical analysis

- Use paired comparisons because every policy/model faces the same worlds.
- Bootstrap ambiguity bundles, preserving all four causes and nested model replicates.
- Report 95% confidence intervals and effect sizes, not only $p$-values.
- For multiple confirmatory model comparisons, use a predeclared correction such as Holm's method.
- Do not promote overlapping/non-overlapping confidence intervals into claims they do not test.
- Report ID and OOD estimates separately before any pooled estimate.
- Preserve every scenario in the intention-to-evaluate analysis.

---

## 14. Reproducibility, caching, and cost controls

### 14.1 Randomness

Use independent named streams for:

- scenario parameters;
- pre-decision market disturbances;
- post-decision disturbance tapes;
- tie-breaking;
- audit outcomes;
- random baseline actions; and
- model replicate identifiers where an API supports a seed.

Never rely on a module-global RNG. State cloning MUST include every mutable variable but MUST reference the same immutable future tape for paired branches.

### 14.2 Model identity and settings

Every request record MUST include provider, requested model ID, returned model ID, snapshot/version when available, endpoint, temperature, top-p, reasoning effort, maximum output, schema version, prompt hash, timestamp, and SDK version.

As of this specification date, official OpenAI documentation lists `gpt-6-luna` as a low-cost model with structured-output support and prices text tokens at $0.10 per million input tokens and $0.50 per million output tokens. This fact is planning context, not a permanent benchmark constant. The runner MUST use a dated pricing config and save the price-table source/date with each run. Provider aliases may change; use a snapshot when offered, otherwise record the returned identifier and evaluation window. See the [GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna) and [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

### 14.3 Cache semantics

The cache key MUST hash:

- provider and model identifier;
- all model settings;
- system/developer/user input bytes;
- output schema;
- benchmark and prompt versions; and
- planned replicate ID.

Planned replicates have distinct IDs and are legitimate separate calls. Retrying an already successful logical trial is forbidden. A retry after transport failure retains the logical trial ID and idempotency metadata where supported.

### 14.4 Budgets

The runner MUST support:

- zero-cost dry run;
- mock adapter;
- five-world smoke test;
- per-stage and total hard dollar ceilings;
- preflight token/cost estimate;
- automatic stop before the next request would exceed budget;
- contemporaneous price configuration;
- input, cached-input, output, and reasoning-token accounting where exposed; and
- a final reconciliation between estimated and provider-reported usage.

Initial planning ceilings remain:

- below $10 for Luna development/calibration; and
- below $100 for the later cross-model hidden evaluation.

These are hard ceilings, not spending targets. A run requiring more must be redesigned or explicitly re-authorized in a future charter change.

### 14.5 Storage

Store separately:

- rendered prompt/observation;
- raw response;
- parsed decision;
- validation/repair status;
- usage and cost record;
- full market trajectory;
- action-value counterfactual samples;
- aggregate metrics; and
- run manifest with hashes.

Secrets, API keys, hidden provider reasoning, and unrelated account metadata MUST never enter artifacts or logs.

---

## 15. Model evaluation strategy

### 15.1 Stages

1. Build with mock responses and no API access.
2. Validate the market, accounting, counterfactual branches, baselines, and analysis locally.
3. Run a five-world Luna smoke test under a hard micro-budget.
4. Run Luna on development only for prompt/parser debugging.
5. Freeze the prompt; use calibration to estimate difficulty, variance, and planned precision—not to keep editing after every result.
6. Freeze benchmark version and generate the hidden set.
7. Run three planned model replicates per hidden world: 100 × 3 = 300 requests per model.
8. Evaluate Luna first, then a small set of predeclared stronger/provider-diverse models if budget permits.
9. Commit response manifests before unblinding hidden labels and oracle outcomes.

### 15.2 Comparison set

The final comparison SHOULD include:

- GPT-6 Luna;
- one stronger OpenAI model;
- one Anthropic model;
- one Google model; and
- optionally one open-weight model.

Exact models, snapshots, access paths, and settings are chosen and frozen immediately before the hidden run because availability and prices change. Model selection MUST happen before any hidden outcome is viewed.

Cross-provider settings cannot always be made identical. The benchmark uses a declared “best-effort equivalent” configuration and reports differences rather than pretending perfect equivalence.

---

## 16. Software architecture

```text
janus/
├── PROJECT_SPEC.md
├── README.md
├── pyproject.toml
├── uv.lock
├── src/janus/
│   ├── core/
│   │   ├── state.py
│   │   ├── clearing.py
│   │   ├── ledger.py
│   │   └── randomness.py
│   ├── participants/
│   │   ├── buyers.py
│   │   └── sellers.py
│   ├── interventions/
│   │   └── actions.py
│   ├── scenarios/
│   │   ├── families.py
│   │   ├── generator.py
│   │   └── matching.py
│   ├── policies/
│   │   ├── constant.py
│   │   ├── random.py
│   │   ├── heuristic.py
│   │   └── partial_information.py
│   ├── oracle/
│   │   ├── counterfactual.py
│   │   └── sensitivity.py
│   ├── models/
│   │   ├── base.py
│   │   ├── mock.py
│   │   └── providers/
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── calibration.py
│   │   ├── statistics.py
│   │   └── failures.py
│   ├── storage/
│   │   ├── schemas.py
│   │   ├── cache.py
│   │   └── manifests.py
│   └── cli.py
├── configs/
│   ├── market/
│   ├── scenarios/
│   ├── interventions/
│   ├── prompts/
│   ├── models/
│   └── experiments/
├── tests/
│   ├── analytic/
│   ├── property/
│   ├── regression/
│   └── integration/
├── experiments/
├── artifacts/
├── reports/
└── docs/decisions/
```

The model-independent interface is conceptually:

```python
decision = model.act(observation, allowed_actions, trial_context)
```

The returned object includes the raw provider response, parsed decision, usage, cost, identifiers, and validation status. Market and analysis code MUST NOT import provider SDKs.

Configuration files are schema-validated. Runtime code MUST reject unknown config fields rather than silently ignore them.

---

## 17. Test and validation plan

### 17.1 Deterministic unit tests

Required tests cover:

- bid/ask sorting and clearing;
- no-trade periods;
- tie-breaking;
- price-ceiling feasibility and rationing;
- tax and subsidy cash flows;
- collusion investigation outcomes;
- consumer, producer, government, and total-surplus accounting;
- transfer cancellation;
- inventory and cash conservation;
- bankruptcy/exit rules;
- scenario reproducibility;
- state cloning;
- disturbance-tape equality across branches;
- oracle selection including epsilon ties;
- metric calculations;
- strict response parsing;
- cache identity and planned-replicate distinction;
- budget preflight and automatic stop; and
- full artifact replay.

### 17.2 Property-based tests

At minimum:

- traded quantity never exceeds submitted eligible supply or demand;
- no buyer pays more than its eligible effective bid;
- no seller receives less than its eligible ask before policy transfers;
- inventory never becomes negative;
- a zero-size intervention is behaviorally identical to do-nothing;
- adding a pure transfer with behavior fixed does not change net surplus;
- the oracle's estimated value is at least every compared action value up to numerical tolerance; and
- serialization never includes forbidden hidden fields.

### 17.3 Analytic fixtures

Create tiny 1×1, 2×2, and 3×3 markets with hand-calculated allocation, price interval, surplus, and intervention consequences. These fixtures are permanent regression tests and appear in technical documentation.

### 17.4 Scenario acceptance gates

Before any LLM evaluation:

- pre-shock markets are stable and near their known competitive range;
- each family modifies the intended primitive;
- public histories overlap by the frozen criteria;
- at least three of the five actions are optimal somewhere in calibration;
- no constant action is epsilon-optimal in more than a provisional 60% of calibration worlds;
- action-value gaps exceed oracle Monte Carlo error for a useful fraction of worlds;
- ambiguous/tied worlds remain included and explicitly labeled; and
- heuristic, partial-information, and full-information baselines are separated with uncertainty.

Exact numeric gates may be revised once during Phase 0 based on local calibration, then frozen and recorded.

---

## 18. Smallest Phase 0 implementation

Phase 0 is intentionally narrower than the final benchmark.

### Phase 0A — accounting kernel

Implement immutable config schemas, state, named randomness, the periodic auction, ledger, analytic fixtures, and replay. Use competitive buyers/sellers only. Exit criterion: all analytic and property tests pass.

### Phase 0B — causal mechanics and actions

Add the four latent families and five interventions with fixed magnitudes. Add exact observation serialization and forbidden-field tests. Exit criterion: every family changes its intended primitive and every intervention passes semantic tests.

### Phase 0C — counterfactual evaluator

Add state cloning, indexed future tapes, paired rollouts, epsilon-optimal sets, primary/secondary utilities, constant/random policies, and a simple heuristic. Exit criterion: oracle convergence and branch-equality tests pass.

### Phase 0D — benchmark validity

Generate only development/calibration candidates, build ambiguity bundles, audit observational overlap, fit the observation-only baseline, test action diversity, and conduct precision analysis. Exit criterion: all scenario acceptance gates pass or the project reports why the construct is not yet viable.

### Phase 0E — API harness, still no full run

Only after 0A–0D pass, add mock/provider adapters, strict schema parsing, cache, cost accounting, and a dry-run manifest. Exit criterion: mock end-to-end test and hard-budget tests pass. Then—and only then—is a five-world paid smoke test allowed.

---

## 19. Research outputs

A completed version should produce:

- open-source simulator and benchmark package;
- procedural, versioned scenario generator;
- immutable split and seed manifests;
- scripted, observation-only, and oracle baselines;
- reproducible experiment runner;
- raw and parsed model artifacts;
- saved factual and counterfactual trajectories;
- confidence intervals and sensitivity analyses;
- failure taxonomy and auditable case studies;
- token/cost report;
- simple local replay visualization;
- paper-style technical report; and
- explicit limitations and external-validity discussion.

Interesting outcomes include excessive intervention, confusion between shortage and collusion, second-order harm from a superficially reasonable action, a gap between diagnosis and policy quality, favorable averages with severe tail harm, confidence concentrated on harmful choices, and limited OOD gains from stronger models. These are hypotheses/examples, not results to be engineered.

---

## 20. Threats to validity and mitigations

### Construct validity

**Threat:** The simulator rewards learning its quirks rather than economic reasoning.  
**Mitigation:** analytic grounding, alternate-clearing ablation, observation-only baselines, matched causes, public prompt semantics, and conservative claims.

**Threat:** The welfare function embeds arbitrary politics.  
**Mitigation:** transfer-correct primary surplus, explicit marginal public-funds assumption, separate metric vector, and pre-registered normative sensitivity.

**Threat:** Diagnosis labels oversimplify mixed real causes.  
**Mitigation:** phase 0 uses mutually exclusive mechanisms for identifiability; conclusions are restricted to this synthetic taxonomy. Mixed-cause scenarios are a future benchmark version.

### Internal validity

**Threat:** Actions consume different random draws.  
**Mitigation:** indexed immutable disturbance tapes and branch-equality tests.

**Threat:** Hidden information leaks through config names, ordering, rounding, or error messages.  
**Mitigation:** forbidden-field tests, serialized-prompt snapshots, label-shuffle controls, and manual audit before freeze.

**Threat:** The oracle is selected from noisy estimates.  
**Mitigation:** common random numbers, epsilon-optimal sets, Monte Carlo convergence checks, and storage of all action-value samples.

### Statistical validity

**Threat:** Three calls per scenario are treated as independent evidence.  
**Mitigation:** nested analysis and bundle-level resampling.

**Threat:** Small per-family sample sizes encourage unstable stories.  
**Mitigation:** aggregate confirmatory tests, precision analysis, exploratory labels on subgroup claims, and effect intervals.

**Threat:** Worst-case regret is dominated by one draw.  
**Mitigation:** CVaR plus transparent case studies; maximum is descriptive.

### External validity

**Threat:** Scripted agents, one good, one decision, and simple interventions omit real institutions.  
**Mitigation:** state the boundary prominently. Depth and auditability are preferred over false realism.

**Threat:** Provider models drift behind stable aliases.  
**Mitigation:** snapshot pinning where available, returned IDs, timestamps, short evaluation window, and immutable raw responses.

### Evaluation contamination

**Threat:** A public benchmark may enter later training data.  
**Mitigation:** procedural hidden seeds, versioned post-freeze generation, withheld hidden artifacts until publication, and prospective evaluation records. This reduces exact-item exposure but cannot prove absence from provider training.

---

## 21. Explicit non-goals

Do not initially build:

- a realistic macroeconomy;
- multiple commodities;
- international trade;
- monetary policy;
- LLM-controlled market participants;
- repeated regulator dialogue;
- long multi-agent conversations;
- fine-tuning or reinforcement learning;
- mixed-cause scenarios;
- parameterized free-form interventions;
- a production SaaS product; or
- a complex frontend.

Do not expand scope merely because an extension is interesting. The first scientific milestone is proof that the small simulator, counterfactual signal, partial-observability construct, and baseline separation are valid.

---

## 22. Decision log: material assumptions

The following assumptions materially affect validity and MUST be revisited in the final limitations section:

1. A periodic sealed-bid double auction is an adequate clean test bed for the intended decision task.
2. Scripted responses approximate strategic adaptation only within their coded rules.
3. True buyer value and seller cost are meaningful welfare primitives inside the simulator.
4. Pure transfers can be separated from real costs; marginal public-funds cost is normative.
5. One intervention and a 20-period recovery horizon capture enough downstream harm.
6. Four mutually exclusive causes are useful for controlled evaluation despite real-world mixtures.
7. Aggregate observations and their rounding are plausible public information.
8. The fitted observation-only baseline is not a proven Bayes oracle.
9. Model confidence is interpreted as a probability of epsilon-optimal action, even though models may not naturally represent it that way.
10. A 100-world hidden pilot supports aggregate paired claims but may be underpowered for fine-grained per-family rankings.
11. Cross-provider comparisons remain imperfect because model APIs and inference controls differ.
12. Results may depend on the clearing institution, intervention magnitudes, horizon, and scenario prior; sensitivity analysis can expose but not eliminate this dependence.

---

## 23. Go/no-go checklist before implementation proceeds beyond design

- [ ] This charter has been reviewed and accepted as the source of truth.
- [ ] Any changes to the primary objective or action semantics are recorded.
- [ ] The analytic market fixtures are specified on paper.
- [ ] Provisional parameter distributions are documented in configs.
- [ ] Phase 0A–0D can be completed with zero model calls.
- [ ] Hidden-set access and commitment procedure are defined.
- [ ] The project owner accepts the narrow external-validity claim.

The next authorized task after acceptance is Phase 0A, not a paid evaluation and not a frontend.

---

## 24. References and primary artifacts

1. Karten, S., Crow, C., & Jin, C. (2026). [Agent Bazaar: Enabling Economic Alignment in Multi-Agent Marketplaces](https://arxiv.org/abs/2605.17698).
2. Struski, P., Swistak, J., Okulska, I., & Biecek, P. (2026). [Competitive Market Behavior of LLMs](https://arxiv.org/abs/2609.02580). [Released framework](https://github.com/jswistak/competitive-market-simulation).
3. Karten, S., Li, W., Ding, Z., Kleiner, S., Bai, Y., & Jin, C. (2025). [LLM Economist: Large Population Models and Mechanism Design in Multi-Agent Generative Simulacra](https://arxiv.org/abs/2507.15815). [Released code](https://github.com/sethkarten/LLM-Economist).
4. Smith, V. L. (1962). [An Experimental Study of Competitive Market Behavior](https://doi.org/10.1086/258609). *Journal of Political Economy*, 70(2), 111–137.
5. Gode, D. K., & Sunder, S. (1993). [Allocative Efficiency of Markets with Zero-Intelligence Traders: Market as a Partial Substitute for Individual Rationality](https://faculty.sites.iastate.edu/tesfatsi/archive/tesfatsi/ZITraders.GodeSunder.JPE1993.pdf). *Journal of Political Economy*, 101(1), 119–137.
6. Brier, G. W. (1950). [Verification of Forecasts Expressed in Terms of Probability](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2). *Monthly Weather Review*, 78(1), 1–3.
7. OpenAI. [GPT-6 Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna) and [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs). Accessed 2026-10-16.
