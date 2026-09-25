import gc
import resource
import time

import numpy as np
import scipy.sparse as sp
import sparseqr
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve_triangular

# COLAMD: a single, cheap fill-reducing heuristic. The previous default
# (SPQR_ORDERING_CHOLMOD = 4) is a "best-effort" mode that tries several
# orderings (COLAMD, METIS, ...) internally and keeps whichever fills in
# the least — on a large graph that multiplies the cost of an already
# expensive factorization several times over. COLAMD trades a bit of
# potential fill-reduction for a much faster factorization; it does not
# change the numerical answer, only how long it takes to get there.
QR_ORDERING = 2  # sparseqr.lib.SPQR_ORDERING_COLAMD

# Blocks at or below this size are solved with a dense LAPACK call. Handing a
# 3-by-3 matrix to SuiteSparseQR costs far more in setup and CHOLMOD copies
# than the arithmetic saves, and a web graph's condensation produces thousands
# of such blocks.
DENSE_BLOCK_LIMIT = 64


def _rss_gb():
    """Current resident set size of this process, in GB."""

    # /proc/self/statm reports sizes in pages; field 1 is the resident set.
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])

    return pages * resource.getpagesize() / 1024**3


def _peak_rss_gb():
    """High-water-mark RSS of this process, in GB (Linux reports ru_maxrss in KB)."""

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _step(message):
    """Print a progress line with live and peak memory, flushed immediately.

    Flushing matters here: if a factorization is killed by the OOM killer,
    anything still sitting in Python's stdout buffer is lost, which is exactly
    why earlier QR logs ended with no indication of how far the run had got.
    """

    print(
        f"[{time.strftime('%H:%M:%S')}] {message}"
        f"   (RSS {_rss_gb():.2f} GB, peak {_peak_rss_gb():.2f} GB)",
        flush=True,
    )


def _build_system_coo(P, damping):
    """Build A = I - damping * P directly in COO format.

    sparseqr documents COO as the performance-optimal input format: its
    converter calls A.tocoo() and then copies the triplets into CHOLMOD.
    Handing it a CSC matrix therefore forces an extra full copy of the matrix
    that is pure overhead. Assembling the system as COO in the first place
    skips that copy, and building it from P's own arrays avoids scipy's
    `I - damping * P` expression, which materializes an identity matrix and
    two intermediate results before arriving at A.
    """

    n = P.shape[0]
    P = P.tocoo(copy=False)

    diagonal = np.arange(n, dtype=P.row.dtype)

    rows = np.concatenate([P.row, diagonal])
    cols = np.concatenate([P.col, diagonal])
    data = np.concatenate([-damping * P.data, np.ones(n, dtype=np.float64)])

    del diagonal
    gc.collect()

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    del rows, cols, data
    gc.collect()

    # Self-loops in the graph put an entry of P on the diagonal, which would
    # leave two COO entries for the same cell. CHOLMOD sums duplicates when it
    # converts the triplet form, so the answer would be right either way, but
    # summing here keeps the reported nonzero count honest and guarantees the
    # explicit diagonal that the triangular solves below rely on.
    A.sum_duplicates()

    return A


def _topological_block_order(labels, n_components, A):
    """Order the strongly connected components so that A becomes block lower triangular.

    A[r, c] is nonzero when node c links to node r, so component labels[c] must
    be placed before labels[r]. Kahn's algorithm on the condensation DAG gives
    such an order. Computing it here rather than reusing SciPy's own component
    numbering keeps the result independent of how SciPy happens to label
    components internally.
    """

    edges = A.tocoo(copy=False)
    crosses = labels[edges.row] != labels[edges.col]
    predecessor = labels[edges.col][crosses]
    successor = labels[edges.row][crosses]

    dag = sp.coo_matrix(
        (np.ones(predecessor.size, dtype=np.int8), (predecessor, successor)),
        shape=(n_components, n_components),
    ).tocsr()
    dag.data[:] = 1  # collapse parallel edges so in-degrees are counted once

    in_degree = np.asarray(dag.sum(axis=0)).ravel().astype(np.int64)

    order = []
    ordered = 0
    frontier = np.flatnonzero(in_degree == 0)

    # Peeled a whole layer at a time: the condensation of a web graph is wide
    # and shallow, so this finishes in a few hundred vectorized passes rather
    # than one Python iteration per component.
    while frontier.size:
        order.append(frontier)
        ordered += frontier.size
        in_degree -= np.asarray(dag[frontier].sum(axis=0)).ravel()
        in_degree[frontier] = -1
        frontier = np.flatnonzero(in_degree == 0)

    if ordered != n_components:
        raise RuntimeError("Condensation of the graph is not acyclic; SCC decomposition failed.")

    return np.concatenate(order)


def _segments(block_sizes):
    """Group consecutive blocks into the units the substitution actually solves.

    Every maximal run of size-1 components is merged into one segment: within a
    run the diagonal sub-block is already lower triangular, so a single sparse
    triangular solve handles thousands of components at once instead of looping
    over them in Python. Larger components each become their own segment.
    """

    segments = []
    start = 0
    run_start = None

    for size in block_sizes:
        if size == 1:
            if run_start is None:
                run_start = start
        else:
            if run_start is not None:
                segments.append((run_start, start, "triangular"))
                run_start = None
            segments.append((start, start + size, "block"))

        start += size

    if run_start is not None:
        segments.append((run_start, start, "triangular"))

    return segments


def _solve_block(block, rhs, ordering):
    """Solve one diagonal block, dense for small ones and SuiteSparseQR otherwise."""

    if block.shape[0] <= DENSE_BLOCK_LIMIT:
        return np.linalg.solve(block.toarray(), rhs)

    solution = sparseqr.solve(block.tocoo(), rhs, ordering=ordering)
    if solution is None:
        raise RuntimeError(f"SuiteSparseQR failed on a {block.shape[0]:,}-node block.")

    return solution


def pagerank_qr(P, out_degree, damping=0.85, ordering=QR_ORDERING):
    """Solve (I - dP) r = b via SuiteSparseQR, using the same dangling-mass correction as LU.

    The system is first split along the graph's strongly connected components.
    Ordering the components topologically makes A block lower triangular, so
    the solve reduces to a forward substitution over the blocks and only the
    diagonal blocks are ever factorized. That matters because the cost of a
    sparse QR is driven by fill-in, which grows far faster than linearly in the
    block size: factorizing one 281,903-node matrix is dramatically more
    expensive, in both time and memory, than factorizing the single large
    component and sweeping through the rest with substitution.
    """

    n = P.shape[0]
    dangling = (out_degree == 0)

    _step(f"Matrix P: {n:,} x {n:,}, {P.nnz:,} nonzeros")
    _step(f"Dangling nodes: {np.sum(dangling):,}")

    _step("Assembling A = I - d*P in COO format...")
    A = _build_system_coo(P, damping)
    _step(f"A assembled: {A.nnz:,} nonzeros")

    _step("Finding strongly connected components...")
    # P[r, c] is nonzero when c links to r, so P.T is the adjacency matrix.
    n_components, labels = connected_components(P.T.tocsr(), directed=True, connection="strong")
    block_sizes_by_label = np.bincount(labels, minlength=n_components)
    _step(
        f"{n_components:,} components; largest holds {block_sizes_by_label.max():,} nodes, "
        f"{np.sum(block_sizes_by_label == 1):,} are single nodes"
    )

    _step("Topologically ordering the components...")
    component_order = _topological_block_order(labels, n_components, A)

    # position_of_component[c] is where component c sits in the new ordering;
    # sorting the nodes by it groups each component together and puts the
    # components in topological order.
    position_of_component = np.empty(n_components, dtype=np.int64)
    position_of_component[component_order] = np.arange(n_components)
    permutation = np.argsort(position_of_component[labels], kind="stable")

    inverse_permutation = np.empty(n, dtype=np.int64)
    inverse_permutation[permutation] = np.arange(n)

    _step("Permuting A into block lower triangular form...")
    A_permuted = sp.coo_matrix(
        (A.data, (inverse_permutation[A.row], inverse_permutation[A.col])),
        shape=(n, n),
    ).tocsr()

    del A
    gc.collect()

    block_sizes = block_sizes_by_label[component_order]
    segments = _segments(block_sizes)
    _step(f"Reduced to {len(segments):,} segments to solve in sequence")

    b = np.full(n, (1.0 - damping) / n)
    dangling_rhs = np.full(n, damping / n)

    # Both right-hand sides share the same A, so they ride through the whole
    # substitution together as a single n-by-2 dense array. Every block is then
    # factorized once instead of once per right-hand side.
    rhs = np.column_stack([b, dangling_rhs])[permutation]
    solution = np.zeros((n, 2), dtype=np.float64)

    _step(f"Starting forward block substitution (ordering={ordering})...")
    substitution_started = time.perf_counter()

    for index, (start, stop, kind) in enumerate(segments):
        rows = A_permuted[start:stop].tocoo()

        # solution is still zero from `start` onwards, so this subtracts exactly
        # the already-solved blocks' contribution and nothing else.
        local_rhs = rhs[start:stop] - rows @ solution

        inside = (rows.col >= start) & (rows.col < stop)
        diagonal_block = sp.coo_matrix(
            (rows.data[inside], (rows.row[inside], rows.col[inside] - start)),
            shape=(stop - start, stop - start),
        )

        del rows

        if kind == "triangular":
            solution[start:stop] = spsolve_triangular(
                diagonal_block.tocsr(), local_rhs, lower=True
            )
        else:
            size = stop - start
            if size > DENSE_BLOCK_LIMIT:
                _step(
                    f"  segment {index + 1}/{len(segments)}: "
                    f"SuiteSparseQR on a {size:,}-node block ({diagonal_block.nnz:,} nonzeros)"
                )

            solution[start:stop] = _solve_block(diagonal_block, local_rhs, ordering)

        del diagonal_block
        gc.collect()

    _step(f"Substitution finished in {time.perf_counter() - substitution_started:.2f} s")

    del A_permuted, rhs
    gc.collect()

    solution = solution[inverse_permutation]
    x, y = solution[:, 0], solution[:, 1]

    _step("Applying the dangling-node correction...")
    denominator = 1.0 - np.sum(y[dangling])
    if abs(denominator) < 1e-14:
        raise RuntimeError("Dangling-node correction became numerically unstable.")

    dangling_mass = np.sum(x[dangling]) / denominator
    rank = x + dangling_mass * y
    rank /= rank.sum()

    _step(f"Done. Peak memory for the whole solve: {_peak_rss_gb():.2f} GB")

    return rank
