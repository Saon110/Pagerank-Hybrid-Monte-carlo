import time

import numpy as np
import scipy.sparse as sp
import sparseqr


# ============================================================
# Configuration
# ============================================================

DATASET = "data/web-Stanford.txt"

DAMPING = 0.85

# Sparse QR ordering.
# SuiteSparseQR can use different orderings to reduce fill-in.
# 4 = default/best ordering.
QR_ORDERING = 4


# ============================================================
# Load SNAP graph
# ============================================================

def load_graph(filename):
    """
    Load a SNAP directed edge-list.

    Each line has:

        source destination

    Returns:
        nodes          Original node IDs
        source_idx     Zero-based source indices
        destination_idx Zero-based destination indices
        out_degree     Out-degree of each node
    """

    print("Loading graph...")

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

    # All node IDs
    nodes = np.unique(
        np.concatenate((sources, destinations))
    )

    n = len(nodes)
    m = len(sources)

    print(f"Number of nodes: {n:,}")
    print(f"Number of edges: {m:,}")

    # Map original node IDs to 0 ... N-1
    source_idx = np.searchsorted(nodes, sources)
    destination_idx = np.searchsorted(nodes, destinations)

    # Out-degree
    out_degree = np.bincount(
        source_idx,
        minlength=n
    )

    return (
        nodes,
        source_idx,
        destination_idx,
        out_degree
    )


# ============================================================
# Build PageRank transition matrix
# ============================================================

def build_transition_matrix(
    source_idx,
    destination_idx,
    out_degree,
    n
):
    """
    Construct sparse column-stochastic transition matrix P.

    For an edge:

        i -> j

    we store:

        P[j, i] = 1 / out_degree(i)

    This means:

        P @ rank

    propagates PageRank through outgoing links.
    """

    print("\nBuilding sparse transition matrix...")

    m = len(source_idx)

    weights = np.zeros(
        m,
        dtype=np.float64
    )

    valid = out_degree[source_idx] > 0

    weights[valid] = (
        1.0 /
        out_degree[source_idx[valid]]
    )

    P = sp.csc_matrix(
        (
            weights,
            (
                destination_idx,
                source_idx
            )
        ),
        shape=(n, n),
        dtype=np.float64
    )

    P.sum_duplicates()

    print(f"Matrix shape: {P.shape}")
    print(f"Non-zero entries: {P.nnz:,}")

    return P


# ============================================================
# Build PageRank linear system
# ============================================================

def build_system(P, out_degree, damping):
    """
    Construct:

        A = I - dP

    for the PageRank equation:

        A r = b

    Dangling nodes are handled separately because their
    transition probability is distributed uniformly.
    """

    n = P.shape[0]

    print("\nBuilding PageRank matrix A = I - dP...")

    # IMPORTANT:
    #
    # DO NOT use:
    #
    #     np.eye(n)
    #
    # because that creates a dense matrix.
    #
    # sp.eye() creates the identity directly as sparse.

    I = sp.eye(
        n,
        format="csc",
        dtype=np.float64
    )

    A = I - damping * P

    A = A.tocsc()

    print(f"A shape: {A.shape}")
    print(f"A non-zero entries: {A.nnz:,}")

    return A


# ============================================================
# PageRank using Sparse QR
# ============================================================

def pagerank_qr(
    A,
    out_degree,
    damping=0.85
):
    """
    Solve PageRank using SuiteSparseQR.

    The PageRank equation is:

        r = d P r
            + d * dangling_mass / N
            + (1-d) / N

    Rearranged:

        (I - dP) r
        =
        (1-d)/N * 1
        +
        d * dangling_mass/N * 1

    We solve the two systems:

        A x = teleportation

    and

        A y = dangling contribution

    where:

        A = I - dP

    Then:

        r = x + dangling_mass * y

    and dangling_mass is solved analytically.
    """

    n = A.shape[0]

    dangling = (
        out_degree == 0
    )

    number_dangling = int(
        np.sum(dangling)
    )

    print(
        f"Dangling nodes: "
        f"{number_dangling:,}"
    )

    # --------------------------------------------------------
    # Teleportation vector
    # --------------------------------------------------------

    b = np.full(
        n,
        (1.0 - damping) / n,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Dangling-node RHS
    # --------------------------------------------------------

    dangling_rhs = np.full(
        n,
        damping / n,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Sparse QR solve
    # --------------------------------------------------------

    print("\nStarting SuiteSparseQR...")

    start = time.perf_counter()

    # sparseqr.solve performs the sparse QR-based solve.
    #
    # A x = b

    x = sparseqr.solve(
        A,
        b,
        ordering=QR_ORDERING
    )

    qr_time_1 = (
        time.perf_counter() - start
    )

    print(
        f"First QR solve completed "
        f"in {qr_time_1:.3f} seconds."
    )

    # --------------------------------------------------------
    # Second solve
    #
    # A y = dangling_rhs
    # --------------------------------------------------------

    start = time.perf_counter()

    y = sparseqr.solve(
        A,
        dangling_rhs,
        ordering=QR_ORDERING
    )

    qr_time_2 = (
        time.perf_counter() - start
    )

    print(
        f"Second QR solve completed "
        f"in {qr_time_2:.3f} seconds."
    )

    # --------------------------------------------------------
    # Calculate dangling mass
    # --------------------------------------------------------

    numerator = np.sum(
        x[dangling]
    )

    denominator = (
        1.0 -
        np.sum(y[dangling])
    )

    if abs(denominator) < 1e-14:

        raise RuntimeError(
            "Dangling-node correction became "
            "numerically unstable."
        )

    dangling_mass = (
        numerator /
        denominator
    )

    print(
        f"Dangling mass: "
        f"{dangling_mass:.15e}"
    )

    # --------------------------------------------------------
    # Final PageRank
    # --------------------------------------------------------

    rank = (
        x +
        dangling_mass * y
    )

    # Normalize
    rank_sum = np.sum(rank)

    rank /= rank_sum

    return rank


# ============================================================
# Verify PageRank
# ============================================================

def verify_rank(
    rank,
    P,
    out_degree,
    damping
):
    """
    Verify:

        r =
        d P r
        + d dangling_mass/N
        + (1-d)/N
    """

    n = len(rank)

    dangling = (
        out_degree == 0
    )

    dangling_mass = np.sum(
        rank[dangling]
    )

    expected = (
        damping * (P @ rank)
        +
        damping * dangling_mass / n
        +
        (1.0 - damping) / n
    )

    residual = np.linalg.norm(
        rank - expected,
        ord=1
    )

    print("\nPageRank verification")
    print("---------------------")

    print(
        f"Sum of PageRank: "
        f"{rank.sum():.15f}"
    )

    print(
        f"L1 residual: "
        f"{residual:.15e}"
    )

    return residual


# ============================================================
# Display top pages
# ============================================================

def display_top_pages(
    nodes,
    rank,
    k=20
):

    ranking = np.argsort(
        rank
    )[::-1]

    print("\n")
    print("=" * 70)
    print(f"TOP {k} PAGES")
    print("=" * 70)

    for position, index in enumerate(
        ranking[:k],
        start=1
    ):

        print(
            f"{position:2d}. "
            f"Node {nodes[index]:8d} "
            f"PageRank = "
            f"{rank[index]:.12e}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PageRank using Sparse QR Decomposition")
    print("=" * 70)

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    (
        nodes,
        source_idx,
        destination_idx,
        out_degree
    ) = load_graph(DATASET)

    n = len(nodes)

    # --------------------------------------------------------
    # Build P
    # --------------------------------------------------------

    P = build_transition_matrix(
        source_idx,
        destination_idx,
        out_degree,
        n
    )

    # --------------------------------------------------------
    # Build A
    # --------------------------------------------------------

    A = build_system(
        P,
        out_degree,
        DAMPING
    )

    # --------------------------------------------------------
    # QR PageRank
    # --------------------------------------------------------

    print("\nComputing PageRank...")

    rank = pagerank_qr(
        A,
        out_degree,
        damping=DAMPING
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_rank(
        rank,
        P,
        out_degree,
        DAMPING
    )

    # --------------------------------------------------------
    # Top pages
    # --------------------------------------------------------

    display_top_pages(
        nodes,
        rank,
        k=20
    )

    # --------------------------------------------------------
    # Total time
    # --------------------------------------------------------

    total_time = (
        time.perf_counter() -
        total_start
    )

    print("\n")
    print("=" * 70)
    print(
        f"Total runtime: "
        f"{total_time:.3f} seconds"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()