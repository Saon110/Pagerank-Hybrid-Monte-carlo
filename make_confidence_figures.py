"""
Reproduces the paper's PI-vs-MC confidence interval figures:
for a few "important" nodes (by exact rank), plot how the Monte Carlo
estimate (mean + 95% confidence interval) moves towards the exact
PageRank value as the number of walks per node (m) increases.

Ground truth = Power Iteration's result, loaded from results/power_rank.npy
(run `python3 main.py power` first if it doesn't exist yet). We use this
instead of re-running Sparse LU here because LU suffers from severe
fill-in on this graph (a few hub nodes make the factor nearly dense),
making it impractical to recompute on demand. REPORT_GUIDE.md already
shows Power and LU agree to ~9 decimal places on this dataset, so Power's
result is a valid stand-in for "exact".

MC method = Complete Path, stopping at dangling nodes (Algorithm 4).

Runs MC repeatedly (in parallel, one process per CPU core) for
m = 1..10, TRIALS independent repeats each, to build the confidence
interval at every m.
"""

import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pagerank.graph import load_edges, adjacency_list
from pagerank.monte_carlo.complete_path_dangling import mc_complete_path_dangling


DATASET = "data/web-Stanford.txt"
GROUND_TRUTH_FILE = "results/power_rank.npy"
DAMPING = 0.85
M_VALUES = list(range(1, 11))
TRIALS = 10
TARGET_RANKS = [1, 10, 100, 1000]
FIGURES_DIR = "figures"

_GRAPH = None


def init_worker(graph):
    global _GRAPH
    _GRAPH = graph


def run_one(m_trial):
    m, trial = m_trial
    random.seed(1000 * m + trial)
    return mc_complete_path_dangling(_GRAPH, runs_per_node=m, damping=DAMPING)


def main():
    print("Loading graph...")
    nodes, source_idx, destination_idx, out_degree = load_edges(DATASET)
    n = len(nodes)
    graph = adjacency_list(source_idx, destination_idx, n)

    print(f"Loading ground truth from {GROUND_TRUTH_FILE} ...")
    if not os.path.exists(GROUND_TRUTH_FILE):
        raise FileNotFoundError(
            f"{GROUND_TRUTH_FILE} not found. Run `python3 main.py power` first."
        )
    exact_rank = np.load(GROUND_TRUTH_FILE)

    ranking = np.argsort(exact_rank)[::-1]
    target_index = {r: ranking[r - 1] for r in TARGET_RANKS}

    jobs = [(m, trial) for m in M_VALUES for trial in range(TRIALS)]

    print(
        f"\nRunning MC Complete Path (Dangling Stop) for "
        f"m = 1..{M_VALUES[-1]}, {TRIALS} trials each "
        f"({len(jobs)} runs total, in parallel)..."
    )

    start = time.perf_counter()
    with ProcessPoolExecutor(initializer=init_worker, initargs=(graph,)) as pool:
        results = list(pool.map(run_one, jobs))
    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.1f} seconds.")

    # Collect, per target rank and per m, the list of MC estimates across trials.
    samples = {r: {m: [] for m in M_VALUES} for r in TARGET_RANKS}
    for (m, _trial), rank in zip(jobs, results):
        for r, idx in target_index.items():
            samples[r][m].append(rank[idx])

    os.makedirs(FIGURES_DIR, exist_ok=True)

    for r in TARGET_RANKS:
        idx = target_index[r]
        exact_value = exact_rank[idx]

        means, upper, lower = [], [], []
        for m in M_VALUES:
            values = np.array(samples[r][m])
            mean = values.mean()
            ci = 1.96 * values.std(ddof=1) / np.sqrt(TRIALS)
            means.append(mean)
            upper.append(mean + ci)
            lower.append(mean - ci)

        plt.figure(figsize=(5, 4))
        plt.plot(M_VALUES, means, "rs", label="MC complete path (dangling stop)")
        plt.plot(M_VALUES, upper, "r^--", label="MC 95% CI (upper)")
        plt.plot(M_VALUES, lower, "rv--", label="MC 95% CI (lower)")
        plt.axhline(exact_value, color="b", label="Power Iteration (exact)")
        plt.xlabel("no. of walks per node (m)")
        plt.ylabel("PageRank")
        plt.title(f"PI vs. MC: node ranked #{r}")
        plt.legend(fontsize=7)
        plt.tight_layout()

        out_path = f"{FIGURES_DIR}/pi_{r}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
