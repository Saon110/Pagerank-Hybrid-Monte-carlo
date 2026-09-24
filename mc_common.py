import random
import numpy as np


DAMPING = 0.85


def load_graph(filename):
    """
    Load the SNAP edge-list graph.

    Returns:
        nodes: sorted original node IDs
        graph: adjacency list using internal indices
    """

    edges = []
    nodes = set()

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            line = line.strip()

            if not line:
                continue

            source, destination = map(int, line.split())

            edges.append((source, destination))
            nodes.add(source)
            nodes.add(destination)

    nodes = sorted(nodes)

    node_to_index = {
        node: i for i, node in enumerate(nodes)
    }

    graph = [[] for _ in range(len(nodes))]

    for source, destination in edges:
        source_idx = node_to_index[source]
        destination_idx = node_to_index[destination]

        graph[source_idx].append(destination_idx)

    return nodes, graph


def random_walk(graph, start, damping=DAMPING):
    """
    Random walk corresponding to the paper's X_t process.

    The starting page is always visited.

    At each step:
        probability 1-damping -> terminate
        probability damping   -> follow a link

    For dangling nodes, P distributes the probability
    uniformly over all pages.
    """

    n = len(graph)

    current = start

    while True:

        # Current page is visited
        yield current

        # Termination probability = 1-c
        if random.random() >= damping:
            break

        neighbors = graph[current]

        # Dangling page:
        # P distributes uniformly over all pages.
        if not neighbors:
            current = random.randrange(n)

        else:
            current = random.choice(neighbors)


def random_walk_stop_dangling(graph, start, damping=DAMPING):
    """
    Random walk corresponding to the paper's Y_t process.

    The starting page is always visited.

    The walk terminates when:
        1. geometric termination occurs, or
        2. it reaches a dangling node.

    This is the walk used by Algorithm 4 / Algorithm 5.
    """

    current = start

    while True:

        # Visit current page
        yield current

        # Stop if current page is dangling
        if not graph[current]:
            break

        # Termination probability = 1-c
        if random.random() >= damping:
            break

        # Follow a real hyperlink
        current = random.choice(graph[current])


def top_k(rank, nodes, k=20):
    """
    Return top-k PageRank entries.
    """

    indices = np.argsort(rank)[::-1][:k]

    return [
        (nodes[i], rank[i])
        for i in indices
    ]


def print_top_k(rank, nodes, k=20):
    print("\n" + "=" * 60)
    print(f"TOP {k}")
    print("=" * 60)

    for position, (node, value) in enumerate(
        top_k(rank, nodes, k),
        start=1
    ):
        print(
            f"{position:3d}. "
            f"Node {node:8d} "
            f"PageRank = {value:.12e}"
        )