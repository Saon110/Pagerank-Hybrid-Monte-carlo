import numpy as np


def pagerank_power(
    graph,
    damping=0.85,
    tolerance=1e-12,
    max_iterations=200,
    initial_rank=None,
    verbose=True,
):
    """
    PageRank via power iteration on the adjacency list.

    By default it starts from the uniform vector (1/n everywhere). Pass
    `initial_rank` (e.g. a Monte Carlo estimate) to "warm start" from a
    better guess instead, which can converge in fewer iterations.

    Returns (rank, iterations_used).
    """

    n = len(graph)

    if initial_rank is None:
        rank = np.ones(n) / n
    else:
        rank = np.array(initial_rank, dtype=np.float64)
        rank /= rank.sum()

    out_degree = np.array([len(neighbors) for neighbors in graph])

    iterations_used = max_iterations

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
        if verbose:
            print(f"Iteration {iteration + 1:3d} | difference = {difference:.12e}")

        rank = new_rank

        if difference < tolerance:
            iterations_used = iteration + 1
            if verbose:
                print(f"\nConverged after {iterations_used} iterations.")
            break

    return rank, iterations_used
