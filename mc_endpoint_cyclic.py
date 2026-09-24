import time
import numpy as np


from mc_common import (
    load_graph,
    random_walk,
    print_top_k
)


DATASET = "data/web-Stanford.txt"
DAMPING = 0.85


def mc_endpoint_cyclic(
    graph,
    runs_per_node=1,
    damping=0.85
):
    """
    Algorithm 2:
    MC end-point with cyclic start.

    Exactly 'runs_per_node' walks are started
    from every page.
    """

    n = len(graph)

    endpoint_count = np.zeros(
        n,
        dtype=np.int64
    )

    total_walks = n * runs_per_node

    for start in range(n):

        for _ in range(runs_per_node):

            endpoint = None

            for node in random_walk(
                graph,
                start,
                damping
            ):
                endpoint = node

            endpoint_count[endpoint] += 1

    rank = endpoint_count / total_walks

    return rank


def main():

    print("Loading graph...")

    nodes, graph = load_graph(DATASET)

    print(f"Number of nodes: {len(nodes):,}")
    print(f"Number of edges: {sum(len(x) for x in graph):,}")

    # Paper's "one iteration" experiment:
    # m = 1
    runs_per_node = 1

    print()
    print("Algorithm 2")
    print("MC endpoint with cyclic start")
    print(f"Runs per node: {runs_per_node}")
    print(
        f"Total walks: "
        f"{len(nodes) * runs_per_node:,}"
    )

    start_time = time.perf_counter()

    rank = mc_endpoint_cyclic(
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