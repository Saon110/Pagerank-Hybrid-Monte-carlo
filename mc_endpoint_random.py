import random
import time
import numpy as np

from mc_common import (
    load_graph,
    random_walk,
    print_top_k
)


DATASET = "data/web-Stanford.txt"
DAMPING = 0.85


def mc_endpoint_random(graph, num_walks, damping=0.85):
    """
    Algorithm 1:
    MC end-point with random start.
    """

    n = len(graph)

    endpoint_count = np.zeros(n, dtype=np.int64)

    for _ in range(num_walks):

        # Random starting page
        start = random.randrange(n)

        # Follow the random walk
        endpoint = None

        for node in random_walk(
            graph,
            start,
            damping
        ):
            endpoint = node

        endpoint_count[endpoint] += 1

    rank = endpoint_count / num_walks

    return rank


def main():

    print("Loading graph...")

    nodes, graph = load_graph(DATASET)

    print(f"Number of nodes: {len(nodes):,}")
    print(f"Number of edges: {sum(len(x) for x in graph):,}")

    num_walks = len(nodes)

    print()
    print("Algorithm 1")
    print("MC endpoint with random start")
    print(f"Number of walks: {num_walks:,}")

    start_time = time.perf_counter()

    rank = mc_endpoint_random(
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