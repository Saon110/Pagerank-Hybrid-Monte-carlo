import numpy as np

from .walks import random_walk


def mc_endpoint_cyclic(graph, runs_per_node=1, damping=0.85):
    """Algorithm 2: MC end-point with cyclic start (runs_per_node walks from every page)."""

    n = len(graph)
    endpoint_count = np.zeros(n, dtype=np.int64)
    total_walks = n * runs_per_node

    for start in range(n):
        for _ in range(runs_per_node):
            endpoint = start
            for node in random_walk(graph, start, damping):
                endpoint = node

            endpoint_count[endpoint] += 1

    return endpoint_count / total_walks
