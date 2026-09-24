import random

import numpy as np

from .walks import random_walk


def mc_endpoint_random(graph, num_walks, damping=0.85):
    """Algorithm 1: MC end-point with random start."""

    n = len(graph)
    endpoint_count = np.zeros(n, dtype=np.int64)

    for _ in range(num_walks):
        start = random.randrange(n)

        endpoint = start
        for node in random_walk(graph, start, damping):
            endpoint = node

        endpoint_count[endpoint] += 1

    return endpoint_count / num_walks
