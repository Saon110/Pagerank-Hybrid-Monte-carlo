import numpy as np
from scipy.sparse.linalg import splu

from pagerank.graph import build_system


def pagerank_lu(P, out_degree, damping=0.85):
    """
    Solve (I - dP) r = b via sparse LU decomposition.

    Dangling nodes are handled with a rank-1 correction so we never
    materialize the dense "dangling -> uniform" component of P:

        r = x + dangling_mass * y
        x = A^-1 * teleportation
        y = A^-1 * (d/N * 1)
        dangling_mass = dangling . x / (1 - dangling . y)
    """

    n = P.shape[0]
    dangling = (out_degree == 0)

    print(f"Dangling nodes: {np.sum(dangling):,}")

    A = build_system(P, damping)
    b = np.full(n, (1.0 - damping) / n)

    print("Performing sparse LU decomposition...")
    lu = splu(A)

    x = lu.solve(b)
    dangling_rhs = np.full(n, damping / n)
    y = lu.solve(dangling_rhs)

    denominator = 1.0 - np.sum(y[dangling])
    if denominator == 0:
        raise RuntimeError("Numerical problem: dangling correction denominator is zero.")

    dangling_mass = np.sum(x[dangling]) / denominator
    rank = x + dangling_mass * y
    rank /= rank.sum()

    return rank
