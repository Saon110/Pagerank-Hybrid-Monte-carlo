import random


def random_walk(graph, start, damping=0.85):
    """
    Random walk corresponding to the paper's X_t process.

    The starting page is always visited. At each step, terminate with
    probability 1-damping, otherwise follow a link. Dangling pages
    distribute uniformly over all pages.
    """

    n = len(graph)
    current = start

    while True:
        yield current

        if random.random() >= damping:
            break

        neighbors = graph[current]

        if not neighbors:
            current = random.randrange(n)
        else:
            current = random.choice(neighbors)


def random_walk_stop_dangling(graph, start, damping=0.85):
    """
    Random walk corresponding to the paper's Y_t process.

    Terminates on geometric stopping (probability 1-damping) or when
    it reaches a dangling node.
    """

    current = start

    while True:
        yield current

        if not graph[current]:
            break

        if random.random() >= damping:
            break

        current = random.choice(graph[current])
