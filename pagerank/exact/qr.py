# import numpy as np
# import sparseqr

# from pagerank.graph import build_system

# QR_ORDERING = 4


# def pagerank_qr(P, out_degree, damping=0.85, ordering=QR_ORDERING):
#     """Solve (I - dP) r = b via SuiteSparseQR, using the same dangling-mass correction as LU."""

#     n = P.shape[0]
#     dangling = (out_degree == 0)

#     print(f"Dangling nodes: {np.sum(dangling):,}")

#     A = build_system(P, damping)
#     b = np.full(n, (1.0 - damping) / n)
#     dangling_rhs = np.full(n, damping / n)

#     print("Running SuiteSparseQR...")
#     x = sparseqr.solve(A, b, ordering=ordering)
#     y = sparseqr.solve(A, dangling_rhs, ordering=ordering)

#     denominator = 1.0 - np.sum(y[dangling])
#     if abs(denominator) < 1e-14:
#         raise RuntimeError("Dangling-node correction became numerically unstable.")

#     dangling_mass = np.sum(x[dangling]) / denominator
#     rank = x + dangling_mass * y
#     rank /= rank.sum()

#     return rank

import numpy as np
import sparseqr

from pagerank.graph import build_system

# COLAMD: a single, cheap fill-reducing heuristic. The previous default
# (SPQR_ORDERING_CHOLMOD = 4) is a "best-effort" mode that tries several
# orderings (COLAMD, METIS, ...) internally and keeps whichever fills in
# the least — on a large graph that multiplies the cost of an already
# expensive factorization several times over. COLAMD trades a bit of
# potential fill-reduction for a much faster factorization; it does not
# change the numerical answer, only how long it takes to get there.
QR_ORDERING = 2  # sparseqr.lib.SPQR_ORDERING_COLAMD


def pagerank_qr(P, out_degree, damping=0.85, ordering=QR_ORDERING):
    """Solve (I - dP) r = b via SuiteSparseQR, using the same dangling-mass correction as LU."""

    n = P.shape[0]
    dangling = (out_degree == 0)

    print(f"Dangling nodes: {np.sum(dangling):,}")

    A = build_system(P, damping)
    b = np.full(n, (1.0 - damping) / n)
    dangling_rhs = np.full(n, damping / n)

    # Factorizing A is the expensive part of a sparse QR solve. b and
    # dangling_rhs share the same A, so solve for both right-hand sides
    # in a single call (a dense n-by-2 matrix) instead of calling
    # sparseqr.solve() twice — that halves the number of factorizations
    # instead of paying for it once per right-hand side.
    print("Running SuiteSparseQR (single factorization, two right-hand sides)...")
    rhs = np.column_stack([b, dangling_rhs])
    solution = sparseqr.solve(A, rhs, ordering=ordering)
    x, y = solution[:, 0], solution[:, 1]

    denominator = 1.0 - np.sum(y[dangling])
    if abs(denominator) < 1e-14:
        raise RuntimeError("Dangling-node correction became numerically unstable.")

    dangling_mass = np.sum(x[dangling]) / denominator
    rank = x + dangling_mass * y
    rank /= rank.sum()

    return rank
