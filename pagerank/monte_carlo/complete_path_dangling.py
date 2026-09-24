import numpy as np

from .walks import random_walk_stop_dangling


def mc_complete_path_dangling(graph, runs_per_node=1, damping=0.85):
    """
    Algorithm 4: MC complete path stopping at dangling nodes.

    Exactly `runs_per_node` walks start from every page; a walk terminates
    on geometric stopping or upon reaching a dangling page.
    """

    n = len(graph)
    visit_count = np.zeros(n, dtype=np.int64)
    total_visits = 0

    for start in range(n):
        for _ in range(runs_per_node):
            for node in random_walk_stop_dangling(graph, start, damping):
                visit_count[node] += 1
                total_visits += 1

    return visit_count.astype(np.float64) / total_visits
