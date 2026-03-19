from lumeq import np
from lumeq.utils import monitor_performance


def _offdiag_norm_sq(matrices):
    r"""Return the squared Frobenius norm of off-diagonal matrix entries.

    The input may be a stack of square matrices with shape ``(m, n, n)`` or
    the same data in transposed internal layout ``(n, n, m)``. In either case,
    the off-diagonal entries from all matrices are collected and their squared
    Frobenius norm is returned.

    Args:
        matrices (numpy.ndarray): Three-dimensional array containing square
            matrix data in shape ``(m, n, n)`` or ``(n, n, m)``.

    Returns:
        float: Squared Frobenius norm of all off-diagonal entries.

    Raises:
        ValueError: If ``matrices`` is not 3D or does not contain square
            matrix axes.
    """
    if matrices.ndim != 3:
        raise ValueError('matrices must be a 3D array')

    if matrices.shape[1] == matrices.shape[2]:
        n = matrices.shape[1]
        mask = ~np.eye(n, dtype=bool)
        vals = matrices[:, mask]
        return np.vdot(vals, vals).real

    elif matrices.shape[0] == matrices.shape[1]:
        n = matrices.shape[0]
        mask = ~np.eye(n, dtype=bool)
        vals = matrices[mask]
        return np.vdot(vals, vals).real

    raise ValueError('matrices must have square matrix axes')


def _pair_features(block):
    r"""Map Hermitian 2x2 blocks onto Pauli vectors and their Gram matrix.

    Any Hermitian ``2 x 2`` block can be expanded as

    ``block = a0 * I + ax * sigma_x + ay * sigma_y + az * sigma_z``,

    where ``I`` is the identity and ``sigma_x, sigma_y, sigma_z`` are the
    Pauli matrices. This helper returns the real coefficient vector
    ``(ax, ay, az)``. The scalar ``a0`` is omitted because it shifts both
    diagonal entries equally and therefore does not affect the Jacobi rotation
    used to reduce off-diagonal couplings.

    For a Hermitian block

    ``[[h00, h01], [h01.conj(), h11]]``,

    the coefficients are

    ``ax = Re(h01) * 2``,
    ``ay = -Im(h01) * 2``,
    ``az = Re(h00 - h11)``.

    If ``block`` has shape ``(2, 2, ...)``, the same mapping is applied over
    all trailing batch dimensions. In that case, the returned array has shape
    ``(3, ...)``, where the first axis stores ``(ax, ay, az)``.

    Args:
        block (numpy.ndarray): Hermitian block or batch of Hermitian blocks
            with shape ``(2, 2)`` or ``(2, 2, ...)``.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Pauli-vector representation of
        ``block`` and the accumulated Gram matrix. The vector output shape is
        ``(3,)`` for a single block and ``(3, ...)`` for batched input. The
        Gram matrix always has shape ``(3, 3)`` and sums outer products over
        all trailing batch dimensions.
    """
    if block.ndim < 2 or block.shape[:2] != (2, 2):
        raise ValueError('block must have shape (2, 2) or (2, 2, ...)')

    vecs = np.array([
        2.0 * block[0, 1].real,
        -2.0 * block[0, 1].imag,
        (block[0, 0] - block[1, 1]).real,
    ])
    vecs_2d = vecs.reshape(3, -1)
    gram = vecs_2d @ vecs_2d.T
    return vecs, gram


def _unitary_from_axis(axis):
    r"""Build the SU(2) rotation that maps the supplied axis onto ``+z``.

    The associated SO(3) rotation is obtained by diagonalizing ``n . sigma``.

    Args:
        axis (array_like): Real three-component axis vector.

    Returns:
        numpy.ndarray: 2x2 unitary rotation matrix.
    """
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-15:
        return np.eye(2, dtype=complex)

    axis = axis / norm
    if axis[2] < 0.0:
        axis = -axis

    pauli_axis = np.array([
        [axis[2], axis[0] - 1j * axis[1]],
        [axis[0] + 1j * axis[1], -axis[2]],
    ], dtype=complex)

    evals, evecs = np.linalg.eigh(pauli_axis)
    order = np.argsort(evals)[::-1]
    unitary = evecs[:, order]

    # Remove arbitrary column phases for deterministic sweeps.
    for col in range(unitary.shape[1]):
        idx = np.argmax(np.abs(unitary[:, col]))
        if abs(unitary[idx, col]) > 1e-15:
            unitary[:, col] *= np.exp(-1j * np.angle(unitary[idx, col]))

    det = np.linalg.det(unitary)
    if abs(det) > 1e-15:
        unitary /= np.sqrt(det)

    return unitary


@monitor_performance
def joint_jacobi_sweep(matrices, tol=1e-12, max_sweeps=100, return_diag=False):
    r"""Approximate joint diagonalization of Hermitian matrices by Jacobi sweeps.

    This follows the Cardoso-Souloumiac idea of optimizing each two-state
    subspace independently.  For every pair (p, q), the Hermitian 2x2 blocks
    are mapped onto Pauli vectors; the dominant eigenvector of the accumulated
    3x3 moment matrix gives the optimal SO(3) rotation, which is then lifted
    back to an SU(2) unitary acting on that subspace.

    The dominant scaling is approximately:

    - time: ``O(s * m * n^3)``
    - memory: ``O(m * n^2)``

    where ``n`` is the matrix dimension, ``m`` is the number of matrices, and
    ``s`` is the number of Jacobi sweeps needed for convergence.

    Reference:
        J.-F. Cardoso and A. Souloumiac, "Jacobi angles for simultaneous
        diagonalization", SIAM J. Matrix Anal. Appl. 17, 161-164 (1996).

    Args:
        matrices (array_like): Stack of matrices with shape ``(m, n, n)`` or
            a single ``(n, n)`` matrix.
        tol (float, optional): Convergence threshold for both pair rotations
            and relative off-diagonal improvement.
        max_sweeps (int, optional): Maximum number of full Jacobi sweeps.
        return_diag (bool, optional): If ``True``, also return the transformed
            matrices ``U^\dagger A_k U``.

    Returns:
        numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray]: Unitary matrix
        approximately jointly diagonalizing the inputs. If ``return_diag`` is
        ``True``, also returns the transformed matrices.
    """
    matrices = np.asarray(matrices)
    if matrices.ndim == 2:
        matrices = matrices[np.newaxis]
    if matrices.ndim != 3:
        raise ValueError('matrices must have shape (m, n, n) or (n, n)')
    if matrices.shape[1] != matrices.shape[2]:
        raise ValueError('matrices must be square')

    matrices = np.array(matrices, dtype=complex, copy=True)
    _, n, _ = matrices.shape
    unitary = np.eye(n, dtype=complex)

    prev_off = _offdiag_norm_sq(matrices)
    if prev_off < tol:
        return (unitary, matrices) if return_diag else unitary

    # Use (n, n, nmats) layout so each (p, q) block is a batched 2x2 slice.
    matrices = matrices.transpose(1, 2, 0)

    for _ in range(max_sweeps):
        max_update = 0.0

        for p in range(n - 1):
            for q in range(p + 1, n):
                pair_weight = np.vdot(matrices[p, q], matrices[p, q]).real
                if pair_weight < tol:
                    continue

                cols = [p, q]
                block = matrices[np.ix_(cols, cols)]
                _, gram = _pair_features(block)

                evals, evecs = np.linalg.eigh(gram)
                axis = evecs[:, np.argmax(evals)]
                rot2 = _unitary_from_axis(axis)

                delta = np.linalg.norm(rot2 - np.eye(2))
                if delta < tol:
                    continue

                # Apply the 2x2 unitary on the selected column/row subspace
                # across the full batch of matrices.
                matrices[:, cols, :] = np.einsum(
                    'icm,cj->ijm', matrices[:, cols, :], rot2, optimize=True
                )
                matrices[cols, :, :] = np.einsum(
                    'ac,cbm->abm', rot2.conj().T, matrices[cols, :, :], optimize=True
                )
                unitary[:, cols] = unitary[:, cols] @ rot2
                max_update = max(max_update, delta)

        matrices = 0.5 * (matrices + matrices.transpose(1, 0, 2).conj())
        off = _offdiag_norm_sq(matrices)
        rel_improve = (prev_off - off) / max(prev_off, 1e-30)
        if max_update < tol or rel_improve < tol:
            break
        prev_off = off

    if return_diag:
        return unitary, matrices.transpose(2, 0, 1)
    return unitary


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    n = 40
    nmats = 8

    xmat = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    qmat, _ = np.linalg.qr(xmat)
    diag_vals = rng.normal(size=(nmats, n))
    hermitian = np.array([
        qmat @ np.diag(diag_vals[k]) @ qmat.conj().T
        for k in range(nmats)
    ])

    mask = ~np.eye(n, dtype=bool)
    before = np.linalg.norm(hermitian[:, mask])
    umat, diag_mats = joint_jacobi_sweep(hermitian, return_diag=True)
    after = np.linalg.norm(diag_mats[:, mask])
    unitarity = np.linalg.norm(umat.conj().T @ umat - np.eye(n))

    print('Approximate joint diagonalization self-test')
    print(f'number of matrices: {nmats}')
    print(f'matrix dimension:   {n}')
    print(f'offdiag norm before: {before:.6e}')
    print(f'offdiag norm after:  {after:.6e}')
    print(f'unitarity error:     {unitarity:.6e}')
