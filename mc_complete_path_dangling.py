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


def mc_complete_path_dangling(
    graph,
    runs_per_node=1,
    damping=0.85
):
    """
    Algorithm 4:
    MC complete path stopping at dangling nodes.

    Exactly 'runs_per_node' walks start from every page.

    A walk terminates when:
        - geometric stopping occurs, or
        - a dangling page is reached.
    """

    n = len(graph)

    visit_count = np.zeros(
        n,
        dtype=np.int64
    )

    total_visits = 0

    for start in range(n):

        for _ in range(runs_per_node):

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

    random.seed(42)

    print("Loading graph...")

    nodes, graph = load_graph(DATASET)

    print(f"Number of nodes: {len(nodes):,}")
    print(f"Number of edges: {sum(len(x) for x in graph):,}")

    # m = 1 corresponds to one walk per page.
    runs_per_node = 1

    print()
    print("Algorithm 4")
    print("MC complete path stopping at dangling nodes")
    print(f"Runs per node: {runs_per_node:,}")

    start_time = time.perf_counter()

    rank = mc_complete_path_dangling(
        graph,
        runs_per_node,
        DAMPING
    )

    elapsed = time.perf_counter() - start_time

    print(f"\nRuntime: {elapsed:.4f} seconds")
    print(f"Sum of PageRank: {rank.sum():.12f}")

    print_top_k(rank, nodes)

    np.save(
        "results/mc_complete_dangling_rank.npy",
        rank
    )

    print(
        "\nSaved PageRank vector to "
        "results/mc_complete_dangling_rank.npy"
    )


if __name__ == "__main__":
    main()