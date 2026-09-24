import argparse
import random
import time

import numpy as np

from pagerank.graph import load_edges, adjacency_list, transition_matrix, print_top_k
from pagerank.exact.power import pagerank_power
from pagerank.exact.lu import pagerank_lu
from pagerank.monte_carlo.endpoint_random import mc_endpoint_random
from pagerank.monte_carlo.endpoint_cyclic import mc_endpoint_cyclic
from pagerank.monte_carlo.complete_path import mc_complete_path
from pagerank.monte_carlo.complete_path_dangling import mc_complete_path_dangling
from pagerank.monte_carlo.complete_path_random import mc_complete_path_random


def run_qr(P, out_degree, damping, param):
    # Imported lazily: sparseqr needs libsuitesparse-dev and shouldn't be
    # required to run every other method.
    from pagerank.exact.qr import pagerank_qr

    return pagerank_qr(P, out_degree, damping=damping)


# Each method declares:
#   uses: "adjacency" (needs the graph as an adjacency list) or "matrix" (needs sparse P)
#   run: a function(graph_or_P, out_degree, damping, param) -> rank
#   save: results/<name>_rank.npy, or None
# `param` is runs_per_node for cyclic/fixed-start methods, or num_walks for random-start methods.
METHODS = {
    "power": {
        "uses": "adjacency",
        "run": lambda g, out_degree, damping, param: pagerank_power(g, damping=damping),
        "save": "results/power_rank.npy",
    },
    "lu": {
        "uses": "matrix",
        "run": lambda P, out_degree, damping, param: pagerank_lu(P, out_degree, damping=damping),
        "save": "results/lu_rank.npy",
    },
    "qr": {
        "uses": "matrix",
        "run": run_qr,
        "save": "results/qr_rank.npy",
    },
    "mc-endpoint-random": {
        "uses": "adjacency",
        "run": lambda g, out_degree, damping, param: mc_endpoint_random(g, param, damping=damping),
        "default_param": lambda n: n,
        "save": "results/mc_endpoint_random_rank.npy",
    },
    "mc-endpoint-cyclic": {
        "uses": "adjacency",
        "run": lambda g, out_degree, damping, param: mc_endpoint_cyclic(g, param, damping=damping),
        "default_param": lambda n: 1,
        "save": "results/mc_endpoint_cyclic_rank.npy",
    },
    "mc-complete-path": {
        "uses": "adjacency",
        "run": lambda g, out_degree, damping, param: mc_complete_path(g, param, damping=damping),
        "default_param": lambda n: 1,
        "save": "results/mc_complete_path_rank.npy",
    },
    "mc-complete-dangling": {
        "uses": "adjacency",
        "run": lambda g, out_degree, damping, param: mc_complete_path_dangling(g, param, damping=damping),
        "default_param": lambda n: 1,
        "save": "results/mc_complete_dangling_rank.npy",
    },
    "mc-complete-random": {
        "uses": "adjacency",
        "run": lambda g, out_degree, damping, param: mc_complete_path_random(g, param, damping=damping),
        "default_param": lambda n: n,
        "save": "results/mc_complete_random_rank.npy",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a PageRank method.")

    parser.add_argument("method", choices=sorted(METHODS))
    parser.add_argument("--dataset", default="data/web-Stanford.txt")
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument(
        "--param",
        type=int,
        default=None,
        help="runs_per_node (fixed-start MC) or num_walks (random-start MC); "
             "defaults to 1 or the node count depending on the method.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)

    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    spec = METHODS[args.method]

    print(f"Loading graph from {args.dataset} ...")
    nodes, source_idx, destination_idx, out_degree = load_edges(args.dataset)
    n = len(nodes)

    print(f"Number of nodes: {n:,}")
    print(f"Number of edges: {len(source_idx):,}")

    if spec["uses"] == "adjacency":
        data = adjacency_list(source_idx, destination_idx, n)
    else:
        data = transition_matrix(source_idx, destination_idx, out_degree, n)

    param = args.param
    if param is None:
        param = spec["default_param"](n) if "default_param" in spec else None

    print(f"\nRunning method: {args.method}")
    if param is not None:
        print(f"Param (runs_per_node / num_walks): {param:,}")

    start_time = time.perf_counter()
    rank = spec["run"](data, out_degree, args.damping, param)
    elapsed = time.perf_counter() - start_time

    print(f"\nRuntime: {elapsed:.4f} seconds")
    print(f"Sum of PageRank: {rank.sum():.12f}")

    print_top_k(rank, nodes, k=args.top_k)

    if spec["save"]:
        np.save(spec["save"], rank)
        print(f"\nSaved PageRank vector to {spec['save']}")


if __name__ == "__main__":
    main()
