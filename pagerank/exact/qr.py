import numpy as np
import sparseqr

from pagerank.graph import build_system

QR_ORDERING = 4


def pagerank_qr(P, out_degree, damping=0.85, ordering=QR_ORDERING):
    """Solve (I - dP) r = b via SuiteSparseQR, using the same dangling-mass correction as LU."""

    n = P.shape[0]
    dangling = (out_degree == 0)

    print(f"Dangling nodes: {np.sum(dangling):,}")

    A = build_system(P, damping)
    b = np.full(n, (1.0 - damping) / n)
    dangling_rhs = np.full(n, damping / n)

    print("Running SuiteSparseQR...")
    x = sparseqr.solve(A, b, ordering=ordering)
    y = sparseqr.solve(A, dangling_rhs, ordering=ordering)

    denominator = 1.0 - np.sum(y[dangling])
    if abs(denominator) < 1e-14:
        raise RuntimeError("Dangling-node correction became numerically unstable.")

    dangling_mass = np.sum(x[dangling]) / denominator
    rank = x + dangling_mass * y
    rank /= rank.sum()

    return rank
