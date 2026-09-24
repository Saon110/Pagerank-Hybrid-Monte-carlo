import time
import numpy as np

from mc_common import (
    load_graph,
    random_walk,
    print_top_k
)


DATASET = "data/web-Stanford.txt"
DAMPING = 0.85


def mc_complete_path(
    graph,
    runs_per_node=1,
    damping=0.85
):
    """
    Algorithm 3:
    MC complete path.

    Uses the P transition matrix, meaning dangling
    pages transition uniformly to a random page.
    """

    n = len(graph)

    visit_count = np.zeros(
        n,
        dtype=np.float64
    )

    total_walks = n * runs_per_node

    for start in range(n):

        for _ in range(runs_per_node):

            for node in random_walk(
                graph,
                start,
                damping
            ):
                visit_count[node] += 1

    rank = (
        (1.0 - damping)
        / total_walks
        * visit_count
    )

    return rank


def main():

    print("Loading graph...")

    nodes, graph = load_graph(DATASET)

    print(f"Number of nodes: {len(nodes):,}")
    print(f"Number of edges: {sum(len(x) for x in graph):,}")

    runs_per_node = 1

    print()
    print("Algorithm 3")
    print("MC complete path")
    print(f"Runs per node: {runs_per_node:,}")

    start_time = time.perf_counter()

    rank = mc_complete_path(
        graph,
        runs_per_node,
        DAMPING
    )

    elapsed = time.perf_counter() - start_time

    print(f"\nRuntime: {elapsed:.4f} seconds")
    print(f"Sum of PageRank: {rank.sum():.12f}")

    print_top_k(rank, nodes)


if __name__ == "__main__":
    main()