import numpy as np


def pagerank_power(graph, damping=0.85, tolerance=1e-12, max_iterations=200):
    """PageRank via power iteration on the adjacency list."""

    n = len(graph)
    rank = np.ones(n) / n
    out_degree = np.array([len(neighbors) for neighbors in graph])

    for iteration in range(max_iterations):
        new_rank = np.full(n, (1 - damping) / n)

        for i in range(n):
            if out_degree[i] == 0:
                continue

            contribution = damping * rank[i] / out_degree[i]
            for j in graph[i]:
                new_rank[j] += contribution

        # Dangling nodes distribute their mass uniformly
        dangling_mass = rank[out_degree == 0].sum()
        new_rank += damping * dangling_mass / n

        difference = np.sum(np.abs(new_rank - rank))
        print(f"Iteration {iteration + 1:3d} | difference = {difference:.12e}")

        rank = new_rank

        if difference < tolerance:
            print(f"\nConverged after {iteration + 1} iterations.")
            break

    return rank
