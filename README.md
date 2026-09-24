# PageRank — Setup & Run

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

For Sparse QR:

```bash
sudo apt update
sudo apt install libsuitesparse-dev
pip install sparseqr
```

Note: on large web graphs (like `web-Stanford.txt`), Sparse LU/QR can suffer
severe fill-in from a few hub nodes and may not finish in reasonable time —
see `REPORT_GUIDE.md` for details. Power Iteration is used as ground truth
everywhere in this project instead.

## 2. Project layout

```
pagerank/
    graph.py              graph loading, transition matrix, top-k helpers
    exact/                 power / lu / qr
    monte_carlo/            walks.py + one module per MC algorithm
    hybrid/                  MC warm-start + Power Iteration
main.py                    CLI entry point for every method
make_all_figures.py        generates every report figure + a full error-metrics table
data/                       input graphs (SNAP edge-list format)
results/                    saved rank vectors (.npy) and run logs (.txt)
figures/                    generated plots (.png)
```

Adding a new method: drop a module in `pagerank/exact/`, `pagerank/monte_carlo/`,
or `pagerank/hybrid/`, and register it in the `METHODS` dict in `main.py`.

## 3. Run a method

```bash
python3 main.py power
python3 main.py lu
python3 main.py qr
python3 main.py mc-endpoint-random
python3 main.py mc-endpoint-cyclic
python3 main.py mc-complete-path
python3 main.py mc-complete-dangling
python3 main.py mc-complete-random
python3 main.py hybrid-power-mc
```

Useful flags: `--dataset <path>`, `--damping <float>`, `--param <int>`
(runs per node for fixed-start MC methods, or number of walks for
random-start ones), `--seed <int>`.

## 4. Run everything

```bash
./run_all_experiments.sh
```

Runs Power Iteration (ground truth), all Monte Carlo variants, and the
Hybrid method, then generates the full comparison figures.

## 5. Generate report figures + error-metrics table

Requires `results/power_rank.npy` to already exist (`python3 main.py power`).

```bash
python3 make_all_figures.py
```

Saves accuracy/runtime comparison charts and the paper-style confidence
interval plots into `figures/`, and prints a full error-metrics table for
every method vs. ground truth.
