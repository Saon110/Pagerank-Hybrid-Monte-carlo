"""
# Master script: generates every figure and measurement used in
# REPORT_GUIDE.md, for every method: Power Iteration, LU, QR, the three
# Monte Carlo variants (endpoint-cyclic, complete-path-dangling,
# complete-path-random) and the hybrid MC + Power method.

# Ground truth = results/power_rank.npy (Power Iteration). It is also
# re-run here to time it, but run `python3 main.py power` first to create
# the ground-truth file if it doesn't exist yet.

# Note: Sparse QR on this graph is slow (QR fill-in is much heavier than
# LU), so that step dominates the total runtime.

# Produces, all under figures/:
#   - accuracy_comparison.png   L1 error of every method vs ground truth
#   - runtime_comparison.png    wall-clock time of every method
#   - pi_1.png, pi_10.png, pi_100.png, pi_1000.png
#         Monte Carlo mean + 95% confidence interval vs number of walks
#         per node (m = 1..10), against the exact value, for the nodes
#         ranked #1, #10, #100, #1000 by ground truth.

# Also prints a full error-metrics table (mean/max absolute & relative
# error, L1 error) for every method, to copy into the report.


import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pagerank.graph import load_edges, adjacency_list, transition_matrix
from pagerank.exact.power import pagerank_power
from pagerank.exact.lu import pagerank_lu
from pagerank.monte_carlo.endpoint_cyclic import mc_endpoint_cyclic
from pagerank.monte_carlo.complete_path_dangling import mc_complete_path_dangling
from pagerank.monte_carlo.complete_path_random import mc_complete_path_random


DATASET = "data/web-Stanford.txt"
GROUND_TRUTH_FILE = "results/power_rank.npy"
DAMPING = 0.85
FIGURES_DIR = "figures"

# Confidence-interval sweep settings (paper's Fig. 3-6 style)
M_VALUES = list(range(1, 11))
TRIALS = 10
TARGET_RANKS = [1, 10, 100, 1000]


def run_hybrid(graph, out_degree, damping, param):
    initial_rank = mc_complete_path_dangling(graph, param, damping=damping)
    rank, _ = pagerank_power(graph, damping=damping, initial_rank=initial_rank, verbose=False)
    return rank


def run_lu(P, out_degree, damping, param):
    return pagerank_lu(P, out_degree, damping=damping)


def run_qr(P, out_degree, damping, param):
    # Imported lazily: sparseqr needs libsuitesparse-dev and shouldn't be
    # required to run every other method.
    from pagerank.exact.qr import pagerank_qr

    return pagerank_qr(P, out_degree, damping=damping)


# name -> (uses, runner(data, out_degree, damping, param), default_param(n) or None)
# "uses" is "adjacency" (needs the graph as an adjacency list) or
# "matrix" (needs the sparse transition matrix P).
METHODS = {
    "power": (
        "adjacency",
        lambda g, od, damping, param: pagerank_power(g, damping=damping)[0],
        None,
    ),
    "lu": (
        "matrix",
        run_lu,
        None,
    ),
    "qr": (
        "matrix",
        run_qr,
        None,
    ),
    "mc-endpoint-cyclic": (
        "adjacency",
        lambda g, od, damping, param: mc_endpoint_cyclic(g, param, damping=damping),
        lambda n: 1,
    ),
    "mc-complete-dangling": (
        "adjacency",
        lambda g, od, damping, param: mc_complete_path_dangling(g, param, damping=damping),
        lambda n: 1,
    ),
    "mc-complete-random": (
        "adjacency",
        lambda g, od, damping, param: mc_complete_path_random(g, param, damping=damping),
        lambda n: n,
    ),
    "hybrid-power-mc": (
        "adjacency",
        run_hybrid,
        lambda n: 1,
    ),
}

_GRAPH = None


def init_worker(graph):
    global _GRAPH
    _GRAPH = graph


def run_mc_trial(m_trial):
    m, trial = m_trial
    random.seed(1000 * m + trial)
    return mc_complete_path_dangling(_GRAPH, runs_per_node=m, damping=DAMPING)


def error_metrics(estimate, exact):
    abs_err = np.abs(estimate - exact)
    rel_err = abs_err / np.maximum(exact, 1e-300)
    return {
        "mean_abs": abs_err.mean(),
        "max_abs": abs_err.max(),
        "mean_rel": rel_err.mean(),
        "max_rel": rel_err.max(),
        "l1": abs_err.sum(),
    }


def main():
    print("Loading graph...")
    nodes, source_idx, destination_idx, out_degree = load_edges(DATASET)
    n = len(nodes)
    graph = adjacency_list(source_idx, destination_idx, n)
    P = transition_matrix(source_idx, destination_idx, out_degree, n)

    print(f"Loading ground truth from {GROUND_TRUTH_FILE} ...")
    if not os.path.exists(GROUND_TRUTH_FILE):
        raise FileNotFoundError(
            f"{GROUND_TRUTH_FILE} not found. Run `python3 main.py power` first."
        )
    exact_rank = np.load(GROUND_TRUTH_FILE)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # -----------------------------------------------------------------
    # Part 1: run every method once, time it, measure error vs ground truth
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Running every method once: timing + accuracy vs ground truth")
    print("=" * 70)

    runtimes = {}
    metrics = {}

    for name, (uses, runner, default_param) in METHODS.items():
        param = default_param(n) if default_param is not None else None
        data = graph if uses == "adjacency" else P

        print(f"\nRunning {name} ...")
        start = time.perf_counter()
        rank = runner(data, out_degree, DAMPING, param)
        elapsed = time.perf_counter() - start

        runtimes[name] = elapsed
        metrics[name] = error_metrics(rank, exact_rank)

        print(f"{name}: {elapsed:.2f}s, L1 error = {metrics[name]['l1']:.4e}")

    print("\nFull error-metrics table:")
    print(f"{'Method':<24}{'Mean Abs':>12}{'Max Abs':>12}{'Mean Rel':>12}{'Max Rel':>12}{'L1':>12}")
    for name, m in metrics.items():
        print(
            f"{name:<24}{m['mean_abs']:>12.3e}{m['max_abs']:>12.3e}"
            f"{m['mean_rel']:>12.3e}{m['max_rel']:>12.3e}{m['l1']:>12.3e}"
        )

    # Figure: accuracy (L1 error) comparison
    plt.figure(figsize=(8, 5))
    names = list(metrics.keys())
    plt.bar(names, [metrics[name]["l1"] for name in names], color="darkorange")
    plt.ylabel("L1 error vs ground truth")
    plt.title("Accuracy comparison across methods\n(ground truth = Power Iteration)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/accuracy_comparison.png", dpi=150)
    plt.close()
    print(f"\nSaved {FIGURES_DIR}/accuracy_comparison.png")

    # Figure: runtime comparison
    plt.figure(figsize=(8, 5))
    plt.bar(names, [runtimes[name] for name in names], color="steelblue")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime comparison across methods")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/runtime_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {FIGURES_DIR}/runtime_comparison.png")

    # -----------------------------------------------------------------
    # Part 2: MC confidence-interval sweep (paper's Fig. 3-6 style)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Monte Carlo confidence-interval sweep (Complete Path - Dangling Stop)")
    print("=" * 70)

    ranking = np.argsort(exact_rank)[::-1]
    target_index = {r: ranking[r - 1] for r in TARGET_RANKS}

    jobs = [(m, trial) for m in M_VALUES for trial in range(TRIALS)]
    print(
        f"\nRunning m = 1..{M_VALUES[-1]}, {TRIALS} trials each "
        f"({len(jobs)} runs total, in parallel)..."
    )

    start = time.perf_counter()
    with ProcessPoolExecutor(initializer=init_worker, initargs=(graph,)) as pool:
        results = list(pool.map(run_mc_trial, jobs))
    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.1f} seconds.")

    samples = {r: {m: [] for m in M_VALUES} for r in TARGET_RANKS}
    for (m, _trial), rank in zip(jobs, results):
        for r, idx in target_index.items():
            samples[r][m].append(rank[idx])

    # Baseline: Power Iteration's own value after exactly m iterations
    # (not run to convergence), so we can see both methods approach the
    # exact answer together -- same as the paper's Fig. 3-6.
    print("\nRunning Power Iteration baseline for m = 1..10 iterations...")
    pi_at_m = {r: [] for r in TARGET_RANKS}
    for m in M_VALUES:
        rank_m, _ = pagerank_power(graph, damping=DAMPING, max_iterations=m, verbose=False)
        for r, idx in target_index.items():
            pi_at_m[r].append(rank_m[idx])

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
        plt.plot(M_VALUES, means, "s", color="red", label="MC complete path (dangling stop)")
        plt.plot(M_VALUES, upper, "^--", color="orange", label="MC 95% CI (upper)")
        plt.plot(M_VALUES, lower, "v--", color="green", label="MC 95% CI (lower)")
        plt.plot(M_VALUES, pi_at_m[r], "D-", color="black", label="Power Iteration (after m iterations)")
        plt.axhline(exact_value, color="blue", label="Power Iteration (converged, exact)")
        plt.xlabel("no. of walks per node / iterations (m)")
        plt.ylabel("PageRank")
        plt.title(f"PI vs. MC: node ranked #{r}")
        plt.legend(fontsize=7)
        plt.tight_layout()

        out_path = f"{FIGURES_DIR}/pi_{r}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    print("\nAll figures generated in ./figures/")


if __name__ == "__main__":
    main()

"""
