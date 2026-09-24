import numpy as np

from .walks import random_walk


def mc_complete_path(graph, runs_per_node=1, damping=0.85):
    """Algorithm 3: MC complete path (dangling nodes teleport uniformly, via random_walk)."""

    n = len(graph)
    visit_count = np.zeros(n, dtype=np.float64)
    total_walks = n * runs_per_node

    for start in range(n):
        for _ in range(runs_per_node):
            for node in random_walk(graph, start, damping):
                visit_count[node] += 1

    return (1.0 - damping) / total_walks * visit_count
