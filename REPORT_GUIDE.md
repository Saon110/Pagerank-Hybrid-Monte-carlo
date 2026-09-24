# Report Figures & Tables — PageRank (Power / LU / Monte Carlo)

Dataset: `web-Stanford.txt` — 281,903 nodes, 2,312,497 edges, 172 dangling nodes.
Damping factor: 0.85.

## Table 1 — Dataset summary

| Metric | Value |
|---|---|
| Nodes | 281,903 |
| Edges | 2,312,497 |
| Dangling nodes | 172 |
| Damping factor | 0.85 |

## Figure 1 — Convergence of Power Iteration

Line plot, x = iteration number, y = difference between successive rank vectors (log scale on y).
Data: every `Iteration N | difference = ...` line in `results/power.txt` (146 iterations, converges below 1e-12).
Shows the expected linear/geometric convergence rate of power iteration.

## Table 2 — Top-10 PageRank pages (ground truth: Power Iteration)

| Rank | Node | PageRank |
|---|---|---|
| 1 | 89073 | 1.130285e-02 |
| 2 | 226411 | 9.267831e-03 |
| 3 | 241454 | 8.297272e-03 |
| 4 | 262860 | 3.023117e-03 |
| 5 | 134832 | 3.001279e-03 |
| 6 | 234704 | 2.571731e-03 |
| 7 | 136821 | 2.453714e-03 |
| 8 | 68889 | 2.430791e-03 |
| 9 | 105607 | 2.391046e-03 |
| 10 | 69358 | 2.364014e-03 |

LU reproduces the same values to ~9 decimal places — worth one sentence noting the two exact methods agree, confirming correctness.

## Table 3 — Top-5 pages across all methods

| Rank | Power | LU | MC-Cyclic | MC-Dangling | MC-Random |
|---|---|---|---|---|---|
| 1 | 89073 | 89073 | 89073 | 89073 | 89073 |
| 2 | 226411 | 226411 | 226411 | 226411 | 226411 |
| 3 | 241454 | 241454 | 241454 | 241454 | 241454 |
| 4 | 262860 | 262860 | 134832 | 262860 | 134832 |
| 5 | 134832 | 134832 | 262860 | 134832 | 262860 |

All five methods agree on the top 3 pages exactly, and only swap the order of ranks 4–5 — a good visual proof that the Monte Carlo estimators converge to the same answer as the exact methods.

## Figure 2 — L1 error of each Monte Carlo method vs. ground truth

Bar chart, one bar per method, y = L1 error (from `results/evaluation_m1.txt`):

| Method | L1 error |
|---|---|
| MC Endpoint – Cyclic Start | 0.554 |
| MC Complete Path – Dangling Stop | 0.196 |
| MC Complete Path – Random Start | 0.271 |

Headline result: **Complete Path (Dangling Stop)** is the most accurate of the three.

## Table 4 — Error metrics per Monte Carlo method

| Method | Mean abs. error | Max abs. error | Mean rel. error | Max rel. error | L1 error |
|---|---|---|---|---|---|
| MC Endpoint – Cyclic | 1.965e-06 | 1.927e-04 | 1.283 | 18.10 | 0.554 |
| MC Complete Path – Dangling Stop | 6.941e-07 | 1.572e-04 | 0.305 | 4.32 | 0.196 |
| MC Complete Path – Random Start | 9.619e-07 | 1.954e-04 | 0.574 | 6.00 | 0.271 |

## Table 5 — Relative error at rank 1 / 10 / 100 / 1000

| Method | Rank 1 | Rank 10 | Rank 100 | Rank 1000 |
|---|---|---|---|---|
| MC Endpoint – Cyclic | 0.95% | 1.41% | 0.42% | 8.25% |
| MC Complete Path – Dangling Stop | 1.39% | 0.15% | 3.46% | 15.61% |
| MC Complete Path – Random Start | 1.73% | 2.41% | 3.64% | 25.37% |

Error grows noticeably for lower-ranked (less important) pages — a good point to discuss: Monte Carlo needs many more samples to accurately resolve low-probability pages.

## Figure 3 — Runtime comparison of Monte Carlo methods

Bar chart, y = seconds, one run each with `runs_per_node = 1` / `num_walks = N`:

| Method | Runtime (s) |
|---|---|
| MC Endpoint – Cyclic | 1.31 |
| MC Complete Path – Dangling Stop | 1.94 |
| MC Complete Path – Random Start | 1.99 |

## Note on QR

`results/qr.txt` is empty — Sparse QR was never actually executed (it needs `libsuitesparse-dev` + `sparseqr`, not installed in this environment). Either install the dependency and run `python3 main.py qr`, or note in the report that QR was implemented but not benchmarked on this dataset.
