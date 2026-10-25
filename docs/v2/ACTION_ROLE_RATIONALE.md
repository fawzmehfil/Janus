# Action roles and economic microcases

The labels below apply only to the primary transfer-correct welfare objective. They are never shown to evaluated models.

- `transaction_tax` is substantive for momentum demand: when submitted bids exceed true consumption values, a small buyer tax can suppress inefficient marginal trades. The simulator test checks tax incidence, government revenue, and transfer balance.
- `collusion_investigation` is substantive for seller collusion: when coordination is detected, disabling markups/withholding can restore efficient trades, net of investigation and compliance costs. Tests cover true positives, false positives, fines, and branch-stable detection.
- `price_ceiling` is a negative control: a price is a transfer, not social surplus. A binding cap can ration efficient trades and always carries an administration cost. Tests verify the cap and real cost rather than forcing it to win.
- `supply_subsidy` is a negative control under the frozen primary parameters: the payment is a transfer, public outlays incur financing cost, and a development sweep found no uniquely optimal calibration case. Tests verify seller receipts, government outlay, transfer balance, and financing cost.
- `do_nothing` is an abstention control. It introduces no cap, flow, investigation, or policy cost and is correctly optimal when intervention costs exceed recoverable distortions.

Price ceiling and subsidy remain valid selectable actions and all harm counts in regret. They may behave differently under alternative normative objectives, which use their own frozen public-information policies. Janus v2 is not described as a balanced five-way correct-answer task.
