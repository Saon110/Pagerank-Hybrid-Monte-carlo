import numpy as np
import scipy.sparse as sp


def load_edges(filename):
    """
    Parse a SNAP edge-list file.

    Returns:
        nodes: sorted original node IDs
        source_idx, destination_idx: zero-based edge endpoints
        out_degree: out-degree of every node
    """

    sources = []
    destinations = []

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            line = line.strip()
            if not line:
                continue

            source, destination = map(int, line.split())
            sources.append(source)
            destinations.append(destination)

    sources = np.asarray(sources, dtype=np.int64)
    destinations = np.asarray(destinations, dtype=np.int64)

    nodes = np.unique(np.concatenate((sources, destinations)))

    source_idx = np.searchsorted(nodes, sources)
    destination_idx = np.searchsorted(nodes, destinations)

    out_degree = np.bincount(source_idx, minlength=len(nodes))

    return nodes, source_idx, destination_idx, out_degree


def adjacency_list(source_idx, destination_idx, n):
    """Build graph[i] = list of pages that page i links to."""

    graph = [[] for _ in range(n)]

    for i, j in zip(source_idx, destination_idx):
        graph[i].append(int(j))

    return graph


def transition_matrix(source_idx, destination_idx, out_degree, n):
    """Build the sparse column-stochastic transition matrix P, where P[j, i] = 1/out_degree(i)."""

    weights = np.zeros(len(source_idx), dtype=np.float64)

    valid = out_degree[source_idx] > 0
    weights[valid] = 1.0 / out_degree[source_idx[valid]]

    P = sp.coo_matrix(
        (weights, (destination_idx, source_idx)),
        shape=(n, n),
        dtype=np.float64,
    ).tocsc()

    return P


def build_system(P, damping):
    """Build A = I - damping * P for the PageRank linear system A r = b."""

    n = P.shape[0]
    I = sp.eye(n, format="csc", dtype=np.float64)

    return (I - damping * P).tocsc()


def top_k(rank, nodes, k=20):
    indices = np.argsort(rank)[::-1][:k]
    return [(nodes[i], rank[i]) for i in indices]


def print_top_k(rank, nodes, k=20):
    print("\n" + "=" * 60)
    print(f"TOP {k}")
    print("=" * 60)

    for position, (node, value) in enumerate(top_k(rank, nodes, k), start=1):
        print(f"{position:3d}. Node {node:8d} PageRank = {value:.12e}")
