import random

import numpy as np

from .walks import random_walk_stop_dangling


def mc_complete_path_random(graph, num_walks, damping=0.85):
    """Algorithm 5: MC complete path with random start, stopping at dangling nodes."""

    n = len(graph)
    visit_count = np.zeros(n, dtype=np.int64)
    total_visits = 0

    for _ in range(num_walks):
        start = random.randrange(n)

        for node in random_walk_stop_dangling(graph, start, damping):
            visit_count[node] += 1
            total_visits += 1

    return visit_count.astype(np.float64) / total_visits
