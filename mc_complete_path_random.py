import random
import time
import numpy as np

from mc_common import (
    load_graph,
    random_walk_stop_dangling,
    print_top_k
)


DATASET = "data/web-Stanford.txt"
DAMPING = 0.85


def mc_complete_path_random(
    graph,
    num_walks,
    damping=0.85
):
    """
    Algorithm 5:
    MC complete path with random start.
    """

    n = len(graph)

    visit_count = np.zeros(
        n,
        dtype=np.int64
    )

    total_visits = 0

    for _ in range(num_walks):

        # Random starting page
        start = random.randrange(n)

        for node in random_walk_stop_dangling(
            graph,
            start,
            damping
        ):

            visit_count[node] += 1
            total_visits += 1

    rank = (
        visit_count.astype(np.float64)
        / total_visits
    )

    return rank


def main():

    print("Loading graph...")

    nodes, graph = load_graph(DATASET)

    print(f"Number of nodes: {len(nodes):,}")
    print(f"Number of edges: {sum(len(x) for x in graph):,}")

    # To make the number of random walks comparable
    # with m=1 cyclic experiments:
    num_walks = len(nodes)

    print()
    print("Algorithm 5")
    print("MC complete path with random start")
    print(f"Number of walks: {num_walks:,}")

    start_time = time.perf_counter()

    rank = mc_complete_path_random(
        graph,
        num_walks,
        DAMPING
    )

    elapsed = time.perf_counter() - start_time

    print(f"\nRuntime: {elapsed:.4f} seconds")
    print(f"Sum of PageRank: {rank.sum():.12f}")

    print_top_k(rank, nodes)


if __name__ == "__main__":
    main()