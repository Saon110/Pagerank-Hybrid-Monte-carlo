import numpy as np


def load_graph(filename):
    """
    Load a SNAP-format directed graph.

    Each non-comment line has:
        source destination
    """

    edges = []
    nodes = set()

    with open(filename, "r") as f:
        for line in f:

            # Skip comments
            if line.startswith("#"):
                continue

            line = line.strip()

            if not line:
                continue

            source, destination = map(
                int,
                line.split()
            )

            edges.append((source, destination))

            nodes.add(source)
            nodes.add(destination)

    # Give every node a compact index:
    # original node ID -> 0, 1, 2, ...
    nodes = sorted(nodes)

    node_to_index = {
        node: i
        for i, node in enumerate(nodes)
    }

    n = len(nodes)

    # Adjacency list
    graph = [[] for _ in range(n)]

    for source, destination in edges:

        source_idx = node_to_index[source]
        destination_idx = node_to_index[destination]

        graph[source_idx].append(destination_idx)

    return nodes, graph


def pagerank(
    graph,
    damping=0.85,
    tolerance=1e-8,
    max_iterations=100
):
    """
    Calculate PageRank using power iteration.
    """

    n = len(graph)

    # Initial PageRank:
    # every page has equal probability
    rank = np.ones(n) / n

    # Number of outgoing links from each page
    out_degree = np.array(
        [len(neighbors) for neighbors in graph]
    )

    for iteration in range(max_iterations):

        # Teleportation contribution
        new_rank = np.ones(n) * (
            (1 - damping) / n
        )

        # Distribute PageRank through links
        for i in range(n):

            # Dangling node
            if out_degree[i] == 0:
                continue

            contribution = (
                damping
                * rank[i]
                / out_degree[i]
            )

            for j in graph[i]:
                new_rank[j] += contribution

        # Handle dangling nodes
        dangling_mass = rank[
            out_degree == 0
        ].sum()

        new_rank += (
            damping
            * dangling_mass
            / n
        )

        # Check convergence
        difference = np.sum(
            np.abs(new_rank - rank)
        )

        print(
            f"Iteration {iteration + 1:3d} "
            f"| difference = {difference:.12e}"
        )

        rank = new_rank

        if difference < tolerance:

            print(
                f"\nConverged after "
                f"{iteration + 1} iterations."
            )

            break

    return rank


def main():

    # Dataset location
    dataset = "data/web-Stanford.txt"

    print("Loading graph...")
    print(f"Dataset: {dataset}\n")

    nodes, graph = load_graph(dataset)

    print(f"Number of nodes: {len(nodes):,}")

    number_of_edges = sum(
        len(neighbors)
        for neighbors in graph
    )

    print(
        f"Number of edges: "
        f"{number_of_edges:,}"
    )

    print("\nRunning PageRank...\n")

    ranks = pagerank(graph)

    # Sort nodes by PageRank
    ranking = np.argsort(ranks)[::-1]

    print("\n" + "=" * 60)
    print("TOP 20 PAGES")
    print("=" * 60)

    for position, index in enumerate(
        ranking[:20],
        start=1
    ):

        print(
            f"{position:2d}. "
            f"Node {nodes[index]:8d} "
            f"PageRank = {ranks[index]:.12e}"
        )


if __name__ == "__main__":
    main()