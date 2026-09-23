import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


DATASET = "data/web-Stanford.txt"
DAMPING = 0.85


def load_graph(filename):
    """
    Load the SNAP edge-list and construct a sparse
    column-stochastic transition matrix P.

    If there is an edge i -> j:
        P[j, i] = 1 / out_degree(i)
    """

    print("Loading dataset...")

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

    sources = np.asarray(sources, dtype=np.int32)
    destinations = np.asarray(destinations, dtype=np.int32)

    # Get all node IDs appearing in the dataset
    nodes = np.unique(np.concatenate((sources, destinations)))

    n = len(nodes)
    m = len(sources)

    print(f"Number of nodes: {n:,}")
    print(f"Number of edges: {m:,}")

    # Convert original node IDs to 0...N-1
    source_idx = np.searchsorted(nodes, sources)
    destination_idx = np.searchsorted(nodes, destinations)

    # Calculate out-degree of every node
    out_degree = np.bincount(
        source_idx,
        minlength=n
    )

    # Weight of every outgoing edge
    weights = np.zeros(m, dtype=np.float64)

    non_dangling = out_degree[source_idx] > 0

    weights[non_dangling] = (
        1.0 / out_degree[source_idx[non_dangling]]
    )

    # P[j, i] = probability of going from i to j
    P = sp.coo_matrix(
        (
            weights,
            (destination_idx, source_idx)
        ),
        shape=(n, n),
        dtype=np.float64
    ).tocsc()

    return nodes, P, out_degree


def pagerank_lu(P, out_degree, damping=0.85):
    """
    Compute PageRank by solving:

        (I - dP) r = b

    using sparse LU decomposition.

    Dangling nodes are handled by redistributing
    their probability uniformly.
    """

    n = P.shape[0]

    print("\nBuilding PageRank linear system...")

    # Identify dangling nodes
    dangling = (out_degree == 0)

    number_dangling = np.sum(dangling)

    print(f"Dangling nodes: {number_dangling:,}")

    # ---------------------------------------------------------
    # Important:
    #
    # Standard PageRank treats a dangling node as linking
    # uniformly to every node.
    #
    # Therefore:
    #
    # P_corrected = P + (1/N) * dangling_vector * 1^T
    #
    # This technically introduces a dense component.
    #
    # We handle it using a low-rank correction instead of
    # explicitly creating a dense matrix.
    # ---------------------------------------------------------

    # First solve the system corresponding to:
    #
    # (I - dP) r = teleportation + dangling contribution
    #
    # We can write the final solution using a rank-1 correction.

    I = sp.eye(n, format="csc")

    A = I - damping * P

    b = np.ones(n, dtype=np.float64) * ((1.0 - damping) / n)

    print("Performing sparse LU decomposition...")

    lu = splu(A)

    print("LU decomposition completed.")

    # ---------------------------------------------------------
    # Handle dangling nodes.
    #
    # PageRank satisfies:
    #
    # r = d P r
    #     + d * dangling_mass / N
    #     + (1-d)/N
    #
    # The dangling_mass depends on r, so:
    #
    # r = A^-1 [teleportation + d/N * dangling_mass * 1]
    #
    # Let:
    #
    # x = A^-1 teleportation
    # y = A^-1 (d/N * 1)
    #
    # Then:
    #
    # r = x + dangling_mass * y
    #
    # And:
    #
    # dangling_mass = dangling^T r
    #
    # Therefore:
    #
    # dangling_mass =
    #     dangling^T x /
    #     (1 - dangling^T y)
    # ---------------------------------------------------------

    print("Solving linear system...")

    x = lu.solve(b)

    # Right-hand side for dangling contribution
    dangling_rhs = np.ones(n, dtype=np.float64) * (damping / n)

    y = lu.solve(dangling_rhs)

    # Calculate dangling mass
    denominator = 1.0 - np.sum(y[dangling])

    if denominator == 0:
        raise RuntimeError(
            "Numerical problem: dangling correction denominator is zero."
        )

    dangling_mass = np.sum(x[dangling]) / denominator

    # Final PageRank
    rank = x + dangling_mass * y

    # Normalize to protect against floating-point error
    rank /= rank.sum()

    return rank


def main():

    print("=" * 70)
    print("PageRank using Sparse LU Decomposition")
    print("=" * 70)

    nodes, P, out_degree = load_graph(DATASET)

    print(f"\nSparse matrix:")
    print(f"Shape: {P.shape}")
    print(f"Non-zero entries: {P.nnz:,}")

    print("\nRunning PageRank with sparse LU...")

    rank = pagerank_lu(
        P,
        out_degree,
        damping=DAMPING
    )

    print("\nPageRank calculation completed.")

    print(f"Sum of PageRank values: {rank.sum():.15f}")

    # Find top 20 pages
    ranking = np.argsort(rank)[::-1]

    print("\n" + "=" * 70)
    print("TOP 20 PAGES")
    print("=" * 70)

    for position, index in enumerate(ranking[:20], start=1):

        print(
            f"{position:2d}. "
            f"Node {nodes[index]:8d} "
            f"PageRank = {rank[index]:.12e}"
        )


if __name__ == "__main__":
    main()