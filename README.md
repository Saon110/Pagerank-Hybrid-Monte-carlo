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

## 2. Project layout

```
pagerank/
    graph.py            graph loading, transition matrix, top-k helpers
    exact/               power / lu / qr
    monte_carlo/          walks.py + one module per MC algorithm
main.py                  CLI entry point for every method
evaluate_mc.py            compares MC methods against the power-iteration ground truth
data/                     input graphs (SNAP edge-list format)
results/                  saved rank vectors (.npy) and run logs (.txt)
```

Adding a new method: drop a module in `pagerank/exact/` or `pagerank/monte_carlo/`
and register it in the `METHODS` dict in `main.py`.

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
```

Useful flags: `--dataset <path>`, `--damping <float>`, `--param <int>`
(runs per node for fixed-start MC methods, or number of walks for
random-start ones), `--seed <int>`.

## 4. Run everything + evaluate

```bash
./run_all_experiments.sh
```

This runs the ground truth, all Monte Carlo variants, and prints their
error against power iteration.

## 5. Evaluate manually

```bash
python3 evaluate_mc.py
```
