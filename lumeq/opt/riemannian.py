"""Sphere, orthogonal group, and Stiefel geometry for orbital arrays.

Vector points lie on a B-unit sphere. Matrix points have shape ``(n, k)``
and satisfy ``X.conj().T @ B @ X = I``. The metric is
``Re tr(U.conj().T @ B @ V)`` for matrices; ``B`` defaults to identity.
Tangent vectors use ambient coordinates. Objectives and optimizers are
defined separately in ``function_on_manifold.py``.
"""

from lumeq import np, scipy
from lumeq.utils import get_ortho_basis, random_matrix

def solve_sylvester(A, C, B=None, solver='iter'):
    r"""Solve a Sylvester or Lyapunov-type matrix equation.

    Solves for ``X`` in ``AX + XB = C``. If ``B`` is omitted, ``B = A^H`` is
    used.

    Args:
        A: Square array of shape ``(n, n)``.
        C: Right-hand side of shape ``(n, m)`` or ``(batch, n, m)``.
        B: Square array of shape ``(m, m)``; defaults to ``A.conj().T``.
        solver: Legacy argument retained for compatibility. All values use
            SciPy's Schur-based solver, without a Kronecker system.

    Returns:
        Solution array with the same shape as ``C``. A unique solution
        requires that no eigenvalue of ``A`` equals an eigenvalue of ``-B``.

    Raises:
        ValueError: If matrix dimensions are incompatible.
    """
    A, C = np.asarray(A), np.asarray(C)
    B = A.conj().T if B is None else np.asarray(B)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('A must be square.')
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError('B must be square.')
    if C.ndim not in (2, 3) or C.shape[-2:] != (A.shape[0], B.shape[0]):
        raise ValueError('C must have trailing shape (A.shape[0], B.shape[0]).')
    if C.ndim == 2:
        return scipy.linalg.solve_sylvester(A, B, C)
    if C.shape[0] == 0:
        return np.empty(C.shape, dtype=np.result_type(A, B, C, float))
    return np.stack([scipy.linalg.solve_sylvester(A, B, rhs) for rhs in C])



class Riemannian:
    """Base geometry with a constant positive-definite overlap metric.

    Subclasses provide projections, retractions, and sanity checks. Initial
    arrays are normalized without mutation; ``exp`` uses polar initialization.
    """
    def __init__(self, ndim=None, x0=None, B=None,
                 retraction='qr', transport='differentiated',
                 **kwargs):
        r"""Initialize a Riemannian manifold model.

        Args:
            ndim: Manifold shape when ``x0`` is not provided.
            x0: Initial point on the manifold.
            B: Hermitian positive-definite ``(n, n)`` overlap matrix. Singular
                overlaps must be reduced to an independent basis beforehand.
            retraction: Retraction method name.
            transport: ``'vector'``, ``'differentiated'``, or ``'parallel'``;
                a compatible callable is also accepted.
            **kwargs: Additional options reserved for subclasses.
        """
        if x0 is not None:
            x0 = np.asarray(x0)
        if isinstance(ndim, (int, np.integer)):
            self.ndim = (int(ndim),)
        elif ndim is not None:
            self.ndim = tuple(ndim)
        else:
            self.ndim = None

        if self.ndim is None: # infer from x0 if possible
            if x0 is not None:
                self.ndim = tuple(x0.shape)
            elif kwargs.get('p0', None) is not None:
                p0 = np.asarray(kwargs['p0'])
                if p0.ndim != 2 or p0.shape[0] != p0.shape[1]:
                    raise ValueError('p0 must be a square projector matrix.')
                overlap = np.eye(p0.shape[0]) if B is None else np.asarray(B)
                k = int(round(np.real(np.trace(overlap @ p0))))
                self.ndim = (p0.shape[0], k)

        if self.ndim is None:
            raise ValueError('Riemannian dimension is not given.')
        if len(self.ndim) not in (1, 2) or any(
                not isinstance(d, (int, np.integer)) or d < 1
                for d in self.ndim):
            raise ValueError('Dimensions must contain one or two positive integers.')
        if len(self.ndim) == 2 and self.ndim[1] > self.ndim[0]:
            raise ValueError('Matrix points require n >= k.')

        if len(self.ndim) == 1:
            if retraction == 'qr':
                retraction = 'norm'
                print('Required retraction is not applicable and changed to norm.')
        elif len(self.ndim) == 2:
            if retraction == 'norm':
                retraction = 'qr'
                print('Required retraction is not applicable and changed to qr.')


        shape = (self.ndim[0], self.ndim[0])
        if B is not None:
            self._B = np.asarray(B)
            if self._B.shape != shape or not np.allclose(self._B, self._B.conj().T):
                raise ValueError('B must be a Hermitian (n, n) matrix.')
            if np.min(np.linalg.eigvalsh(self._B)) <= 1e-15:
                raise ValueError('B must be positive definite without discarded modes.')
            # _L = B^1/2 whitens X: (_L X)^H (_L X) = X^H B X = I.
            # _Z = B^-1/2 converts back; _Binv = B^-1.
            self._L, self._Z, self._Binv = get_ortho_basis(self._B, method='lowdin')
        else: # identity matrix
            self._B = np.eye(self.ndim[0])
            self._L, self._Z, self._Binv = np.copy(self._B), np.copy(self._B), np.copy(self._B)


        # get actual retraction and transport methods, with defaults
        self.retraction = getattr(
                self, 'retraction_' + str(retraction), self.exp)
        self.inverse_retraction = getattr(
                self, 'inverse_retraction_' + str(retraction), self.log)
        self.transport = getattr(
                self, str(transport) + '_transport', self.parallel_transport)

        self.initialize_point(x0, **kwargs)
        self.check_sanity()


    def initialize_point(self, x0=None, **kwargs):
        r"""Initialize a vector or frame using the selected retraction."""
        if x0 is None:
            seed = kwargs.get('seed', 12345)
            x0 = random_matrix(self.ndim, seed=seed, sym=False)
        initializer = self.retraction
        if self.retraction.__name__ == 'exp':
            initializer = (self.retraction_norm if len(self.ndim) == 1
                           else self.retraction_polar)
        self.x0 = initializer(x0, np.zeros(x0.shape))


    @staticmethod
    def to_density(x):
        r"""Return the rank-one or frame density represented by ``x``."""
        return np.outer(x, x.conj()) if x.ndim == 1 else x @ x.conj().T


    @staticmethod
    def density_tangent(x, v):
        r"""Return the density change induced by a point tangent ``v``."""
        if x.ndim == 1:
            return np.outer(v, x.conj()) + np.outer(x, v.conj())
        return v @ x.conj().T + x @ v.conj().T


    def dimension(self, dtype=float):
        r"""Return the real dimension of the manifold over the chosen field."""
        raise NotImplementedError('Actual dimension is not implemented.')


    def norm(self, x, v):
        r"""Return the norm of a tangent vector.

        Args:
            x: Base point on the manifold.
            v: Tangent vector at ``x``.

        Returns:
            The tangent-vector norm induced by the overlap metric.
        """
        return np.sqrt(max(0.0, self.inner_product(v, v)))


    def inner_product(self, u, v):
        r"""Return ``Re tr(U^H B V)`` for vectors or coefficient matrices."""
        return float(np.real(np.vdot(u, self._B @ v)))


    def dist(self, v1, v2, x1=None, x2=None):
        r"""Return the distance between two tangent vectors
        at same point on the manifold if ``x1`` and ``x2`` is None;
        at different points on the manifold if ``x1`` and ``x2`` are given.
        Then ``v1`` and ``v2`` are tangent vectors at ``x1`` and ``x2``,
        respectively. The distance is evaluated at point ``x2``.

        Args:
            v1: First tangent vector.
            v2: Second tangent vector.
            x1: Base point of ``v1`` on the manifold.
            x2: Base point of ``v2`` on the manifold.

        Returns:
            The metric norm of the difference between ``v1`` and ``v2``.
        """
        if (x1 is None) != (x2 is None):
            raise ValueError('Provide both base points or neither.')
        if x1 is not None:
            v1 = self.transport(x1, x2, v1)
        dv = v1 - v2
        return self.norm(x2, dv)


    def distance_of_points(self, x1, x2):
        r"""Return the local geodesic distance between two manifold points."""
        v = self.log(x1, x2)
        return self.norm(x1, v)


    def exp(self, x, v, dt=1.):
        r"""Map a tangent vector to the manifold via the exponential map
        to arrive at a new point.

        Args:
            x: Base point on the manifold.
            v: Tangent vector at ``x``.
            dt: Step length along the geodesic.

        Returns:
            A new point on the manifold reached from ``x`` along ``v``.
        """
        raise NotImplementedError('Actual exp is not implemented.')


    def geodesic(self, x, v, dt=1.):
        r"""Alias for the subclass exponential map at step length ``dt``."""
        return self.exp(x, v, dt)


    def log(self, x1, x2, dt=1.):
        r"""Map a point on the manifold back to a tangent vector.

        Args:
            x1: Base point on the manifold.
            x2: Target point on the manifold.
            dt: Step length scaling.

        Returns:
            A tangent vector at ``x1`` pointing toward ``x2``.
        """
        raise NotImplementedError('Actual log is not implemented.')


    def projection(self, x, v):
        r"""Project a vector onto the tangent space at ``x``.

        Args:
            x: Base point on the manifold.
            v: Ambient-space vector or matrix.

        Returns:
            The projection of ``v`` onto the tangent space at ``x``.
        """
        raise NotImplementedError('Actual projection is not implemented.')


    def to_tangent_space(self, x, v):
        r"""Alias for the subclass tangent projection."""
        return self.projection(x, v)


    def to_normal_space(self, x, v):
        r"""Project a vector onto the normal space at ``x``.

        Args:
            x: Base point on the manifold.
            v: Ambient-space vector or matrix.

        Returns:
            The projection of ``v`` onto the normal space at ``x``.
        """
        return v - self.projection(x, v)


    def parallel_transport(self, x0, x1, v0):
        r"""
        Exact parallel transport of a tangent vector from ``x0`` to ``x1``.
        """
        raise NotImplementedError('Actual parallel_transport is not implemented.')


    def differentiated_transport(self, x0, x1, v0):
        r"""Differentiate the selected retraction from ``x0`` to ``x1``."""
        raise NotImplementedError('Actual differentiated_transport is not implemented.')


    def vector_transport(self, x0, x1, v0):
        r"""Project a tangent vector from ``x0`` to ``x1``."""
        raise NotImplementedError('Actual vector_transport is not implemented.')


    def weingarten(self, x, v, n, *args, **kwargs):
        r"""Return the shape-operator correction for a metric-normal vector.

        With the convention used by :meth:`riemannian_hessian`, this is the
        negative tangent projection of the normal field's derivative along
        ``v``.

        Args:
            x: Base point on the manifold.
            v: Tangent direction at ``x``.
            n: Normal vector at the point ``x``.
            *args: Additional implementation-specific arguments.
            **kwargs: Additional implementation-specific keyword arguments.

        Returns:
            A tangent vector giving the curvature correction along ``v``.
        """
        raise NotImplementedError('Actual weingarten is not implemented.')


    def check_tangent(self, x, v):
        r"""Check whether ``v`` is tangent at ``x``.

        Args:
            x: Base point on the manifold.
            v: Candidate tangent vector.

        Returns:
            ``True`` if ``v`` satisfies the tangent-space constraints.
        """
        raise NotImplementedError('Actual check_tangent is not implemented.')


    def riemannian_gradient(self, x, grad):
        r"""Convert a Euclidean gradient to the overlap metric and project.

        Args:
            x: Base point on the manifold.
            grad: Euclidean gradient in the ambient space.

        Returns:
            The Riemannian gradient at ``x``.
        """
        return self.projection(x, self._Binv @ grad)


    def riemannian_hessian(self, x, v, grad, hess, hess_dot_v=None):
        r"""Convert an ambient Hessian action to the manifold metric.

        Args:
            x: Base point on the manifold.
            v: Tangent direction at ``x``.
            grad: Euclidean gradient at ``x``.
            hess: Euclidean Hessian matrix or action callable.
            hess_dot_v: Precomputed Euclidean Hessian action along ``v``.

        Returns:
            The Riemannian Hessian action along ``v``.
        """
        if hess_dot_v is None:
            hess_dot_v = hess(v) if callable(hess) else hess @ v
        metric_grad = self._Binv @ grad
        normal = self.to_normal_space(x, metric_grad)
        return self.projection(
            x, self._Binv @ hess_dot_v + self.weingarten(x, v, normal)
        )



class OrthogonalGroup(Riemannian):
    r"""Square orthogonal/unitary frames, with legacy sphere support.

    A one-dimensional ``x0`` represents a sphere under the real metric.
    Matrix tangent vectors are ambient arrays ``V = X Omega``; callers
    must not pass ``Omega`` directly to an update method.
    """
    def check_sanity(self):
        r"""Require a B-normalized vector or a B-orthonormal square frame."""
        x = self.x0
        if x.ndim == 2 and x.shape[0] != x.shape[1]:
            raise ValueError('Points on the orthogonal group should be square matrices.')
        overlap = (np.vdot(x, self._B @ x) if x.ndim == 1
                   else x.conj().T @ self._B @ x)
        identity = 1. if x.ndim == 1 else np.eye(x.shape[1])
        if not np.allclose(overlap, identity):
            raise ValueError('The initial point is not B-orthonormal.')


    def dimension(self, dtype=None):
        r"""Return the manifold's real dimension, inferring its field by default."""
        n = self.ndim[0]
        if dtype is None:
            dtype = self.x0.dtype
        complex_field = np.issubdtype(np.dtype(dtype), np.complexfloating)
        if len(self.ndim) == 1:
            return 2 * n - 1 if complex_field else n - 1
        return n * n if complex_field else n * (n - 1) // 2


    def projection(self, x, v):
        r"""Project onto the metric sphere or square-frame tangent space."""
        if x.ndim == 1:
            return v - x * np.real(np.vdot(x, self._B @ v))

        overlap = x.conj().T @ self._B @ v
        return v - x @ ((overlap + overlap.conj().T) / 2)


    def check_tangent(self, x, v):
        r"""Check the metric tangent constraint for sphere or frame points."""
        if x.shape != v.shape:
            return False
        if x.ndim == 1:
            return bool(np.isclose(np.real(np.vdot(x, self._B @ v)), 0))
        overlap = x.conj().T @ self._B @ v
        return bool(np.allclose(overlap + overlap.conj().T, 0))


    def weingarten(self, x, v, normal):
        r"""Return the curvature correction for a metric-normal vector."""
        if x.ndim == 1:
            scale = np.real(np.vdot(x, self._B @ normal))
            return -scale * v
        overlap = x.conj().T @ self._B @ normal
        curvature = v @ ((overlap + overlap.conj().T) / 2)
        return -self.projection(x, curvature)


    def retraction_norm(self, x, v, dt=1.):
        r"""Retract by normalizing ``x + dt * v`` onto the sphere."""
        x = x + v * dt
        return x / np.sqrt(np.real(np.vdot(x, self._B @ x)))


    def retraction_qr(self, x, v, dt=1.):
        r"""QR-retract in whitened coordinates with positive diagonal phases."""
        y = self._L @ (x + dt * v) # whitened coordinates
        q, r = np.linalg.qr(y)

        diagonal = np.diag(r)
        phase = np.ones_like(diagonal)
        nonzero = np.abs(diagonal) > 0
        phase[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
        return self._Z @ (q * phase)


    def retraction_polar(self, x, v, dt=1.):
        r"""Polar-retract ``x + dt*v`` in whitened coordinates."""
        y = self._L @ (x + dt * v)
        u, _, vh = np.linalg.svd(y, full_matrices=False)
        return self._Z @ (u @ vh)


    def retraction_cayley(self, x, v, dt=1.):
        r"""Apply the Cayley transform of a square frame's ambient tangent."""
        if x.ndim != 2 or x.shape[0] != x.shape[1]:
            raise ValueError('Cayley retraction requires a square matrix.')

        y = self._L @ x
        identity = np.eye(x.shape[0], dtype=y.dtype)
        if not np.allclose(y.conj().T @ y, identity):
            if np.any(v):
                raise ValueError('Cayley retraction requires an orthonormal base point.')
            # The constructor also uses a zero step to normalize its initial point.
            return self.retraction_polar(x, v, dt)

        tangent = self._L @ v
        generator = tangent @ y.conj().T
        generator = (generator - generator.conj().T) / 2
        update = np.linalg.solve(identity - dt * generator / 2,
                                 (identity + dt * generator / 2) @ y)
        return self._Z @ update


    def inverse_retraction_norm(self, x1, x2, dt=1.):
        r"""Invert sphere normalization locally, where ``Re(x1^H B x2)>0``."""
        overlap = np.real(np.vdot(x1, self._B @ x2))
        if overlap <= 0:
            raise ValueError('The target lies outside the normalization chart.')
        return (x2 - overlap * x1) / (dt * overlap)


    def inverse_retraction_qr(self, x1, x2, dt=1.):
        r"""Invert the frame QR retraction in its positive-diagonal chart.

        Solve ``M R + R^H M^H = 2 I`` for upper-triangular ``R`` with positive
        real diagonal, where ``M = x1^H B x2``. Then ``x2 R = x1 + dt*v``.
        """
        m = x1.conj().T @ self._B @ x2
        n = m.shape[0]
        r = np.zeros_like(m)
        mr = np.zeros_like(m)
        complex_field = np.iscomplexobj(m)
        for j in range(n):
            basis = []
            for i in range(j):
                e = np.zeros(j + 1, dtype=m.dtype)
                e[i] = 1
                basis.append(e)
                if complex_field:
                    basis.append(1j * e)
            e = np.zeros(j + 1, dtype=m.dtype)
            e[j] = 1
            basis.append(e)
            basis = np.column_stack(basis)
            columns = m[:, :j + 1] @ basis
            target = -mr[j, :j].conj()
            if complex_field:
                system = np.vstack((columns[:j].real, columns[:j].imag,
                                    columns[j].real))
                rhs = np.r_[target.real, target.imag, 1.]
            else:
                system = columns[:j + 1]
                rhs = np.r_[target, 1.]
            try:
                r[:j + 1, j] = basis @ np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError as exc:
                raise ValueError('The target lies outside the QR chart.') from exc
            if r[j, j].real <= 0:
                raise ValueError('The target lies outside the QR chart.')
            mr[:, j] = m[:, :j + 1] @ r[:j + 1, j]
        return (x2 @ r - x1) / dt


    def inverse_retraction_polar(self, x1, x2, dt=1.):
        r"""Invert the square-frame polar retraction in its local chart."""
        m = x1.conj().T @ self._B @ x2
        hermitian = m + m.conj().T
        if np.min(np.linalg.eigvalsh(hermitian)) <= 0:
            raise ValueError('The target lies outside the polar chart.')

        omega = np.linalg.solve(hermitian, m - m.conj().T) / dt
        omega = (omega - omega.conj().T) / 2
        return x1 @ omega


    def inverse_retraction_cayley(self, x1, x2, dt=1.):
        r"""Invert the square-frame Cayley retraction when ``M + I`` is invertible.

        Here ``M = x1^H B x2``. The principal local inverse is
        ``v = (2/dt) x1 (M + I)^-1 (M - I)``.
        """
        m = x1.conj().T @ self._B @ x2
        identity = np.eye(m.shape[0], dtype=m.dtype)
        try:
            omega = 2 * np.linalg.solve(m + identity, m - identity) / dt
        except np.linalg.LinAlgError as exc:
            raise ValueError('The target lies outside the Cayley chart.') from exc

        omega = (omega - omega.conj().T) / 2
        return x1 @ omega


    def parallel_transport(self, x1, x2, v1):
        r"""Parallel-transport ``v1`` along the shortest local geodesic.

        Sphere points use the real metric induced by ``B``. For square
        frames, write ``x2 = x1 exp(Omega)`` and ``v1 = x1 Xi``. The
        transported vector is ``x1 exp(Omega/2) Xi exp(Omega/2)``.
        The sphere antipode and a group logarithm at eigenvalue ``-1``
        have no unique chosen geodesic.
        """
        if x1.ndim == 1:
            overlap = np.real(np.vdot(x1, self._B @ x2))
            denominator = 1. + overlap
            if denominator <= 1e-12:
                raise ValueError('Parallel transport is ambiguous at sphere antipodes.')
            component = np.real(np.vdot(x2, self._B @ v1))
            return v1 - (component / denominator) * (x1 + x2)

        if x1.shape[0] != x1.shape[1]:
            raise NotImplementedError(
                'Parallel transport is implemented only for square frames.'
            )

        omega = x1.conj().T @ self._B @ self.log(x1, x2)
        half_step = scipy.linalg.expm(omega / 2)
        xi = x1.conj().T @ self._B @ v1
        return x1 @ half_step @ xi @ half_step


    def differentiated_transport(self, x1, x2, v1):
        r"""Differentiate the selected retraction from ``x1`` toward ``x2``.

        With ``eta = inverse_retraction(x1, x2)``, return
        ``D retraction(x1, eta)[v1]``. Unlike parallel transport, this need
        not preserve tangent-vector norms. Rectangular subclasses use
        projection transport here.
        """
        if x1.ndim == 2 and x1.shape[0] != x1.shape[1]:
            return self.projection(x2, v1)

        method = self.retraction.__name__
        eta = self.inverse_retraction(x1, x2)
        if method == 'retraction_norm':
            point = x1 + eta
            scale = np.sqrt(np.real(np.vdot(point, self._B @ point)))
            return self.projection(x2, v1) / scale

        if method == 'retraction_cayley':
            m = x1.conj().T @ self._B @ x2
            xi = x1.conj().T @ self._B @ v1
            half_sum = (np.eye(m.shape[0], dtype=m.dtype) + m) / 2
            return x1 @ half_sum @ xi @ half_sum

        if method == 'exp':
            if x1.ndim == 1:
                speed = self.norm(x1, eta)
                if speed == 0:
                    return v1.copy()
                sine_over_speed = np.sinc(speed / np.pi)
                if speed < 1e-4:
                    coefficient = -1 / 3 + speed**2 / 30
                else:
                    coefficient = (
                        speed * np.cos(speed) - np.sin(speed)
                    ) / speed**3
                overlap = self.inner_product(eta, v1)
                return (sine_over_speed * v1
                        - overlap * sine_over_speed * x1
                        + overlap * coefficient * eta)
            omega = x1.conj().T @ self._B @ eta
            xi = x1.conj().T @ self._B @ v1
            return x1 @ scipy.linalg.expm_frechet(
                omega, xi, compute_expm=False
            )

        point = x1 + eta
        if method == 'retraction_qr':
            r = x2.conj().T @ self._B @ point
            step = np.linalg.solve(r.T, v1.T).T
            c = x2.conj().T @ self._B @ step
            lower = np.tril(c, -1)
            skew = lower - lower.conj().T
            if np.iscomplexobj(c):
                skew[np.diag_indices_from(skew)] = 1j * np.imag(np.diag(c))
            return step - x2 @ (c - skew)

        if method == 'retraction_polar':
            h = x2.conj().T @ self._B @ point
            h = (h + h.conj().T) / 2
            c = x2.conj().T @ self._B @ v1
            skew = solve_sylvester(h, c - c.conj().T, h)
            step = np.linalg.solve(h.T, v1.T).T
            return step - x2 @ (x2.conj().T @ self._B @ step - skew)

        raise NotImplementedError(
            'Differentiated transport is unavailable for this retraction.'
        )


    def vector_transport(self, x1, x2, v1):
        r"""Project ``v1`` onto the tangent space at ``x2``.

        This vector transport works for sphere and frame points but need not
        preserve tangent-vector norms.
        """
        return self.projection(x2, v1)


    def exp(self, x, v, dt=1.):
        r"""Follow the sphere or square-frame geodesic with initial velocity v."""
        if x.ndim == 1:
            speed = np.sqrt(np.real(np.vdot(v, self._B @ v)))
            if speed == 0:
                return x.copy()
            return x * np.cos(dt * speed) + v * np.sin(dt * speed) / speed

        y = self._L @ x
        tangent = self._L @ v
        generator = y.conj().T @ tangent
        return self._Z @ (y @ scipy.linalg.expm(dt * generator))


    def log(self, x1, x2, dt=1.):
        r"""Return the local sphere or square-frame logarithm at ``x1``.

        The result satisfies ``exp(x1, log(x1, x2, dt), dt) == x2`` near
        ``x1``. Sphere antipodes and square-frame eigenvalues at ``-1`` have
        no unique principal logarithm.
        """
        if x1.ndim == 1:
            overlap = np.clip(np.real(np.vdot(x1, self._B @ x2)), -1., 1.)
            residual = x2 - overlap * x1
            sine = self.norm(x1, residual)
            if sine <= 10 * np.finfo(float).eps:
                if overlap < 0:
                    raise ValueError('The sphere logarithm is not unique at antipodes.')
                return np.zeros_like(x1)
            angle = np.arctan2(sine, overlap)
            return residual * (angle / (dt * sine))

        m = x1.conj().T @ self._B @ x2
        if np.min(np.abs(np.linalg.eigvals(m) + 1)) < 1e-12:
            raise ValueError('The group logarithm is not unique at eigenvalue -1.')

        omega = scipy.linalg.logm(m) / dt
        omega = (omega - omega.conj().T) / 2
        if not np.iscomplexobj(x1) and not np.iscomplexobj(x2):
            omega = np.real_if_close(omega)
            if np.iscomplexobj(omega):
                raise ValueError('The target is outside the real logarithm chart.')
        return x1 @ omega



class Stiefel(OrthogonalGroup):
    r"""Stiefel manifold of orthonormal ``k``-frames in ``n`` dimensions.

    A point ``X`` satisfies ``X^dagger B X = I_k`` and a tangent vector ``V``
    satisfies ``X^dagger B V + V^dagger B X = 0``.
    The QR and polar retractions are inherited from ``OrthogonalGroup``;
    their formulas also apply to rectangular frames.
    """
    def check_sanity(self):
        r"""Require an n-by-k frame with n >= k >= 1."""
        ndim = self.ndim
        if len(ndim) != 2 or ndim[0] < ndim[1]:
            raise ValueError('Points on Stiefel should be tall rectangular matrices.')
        if not np.allclose(self.x0.conj().T @ self._B @ self.x0,
                           np.eye(ndim[1])):
            raise ValueError('The initial frame is not B-orthonormal.')


    def dimension(self, dtype=None):
        r"""Real dimension of the frame manifold over the requested field."""
        n, k = self.ndim
        if dtype is None:
            dtype = self.x0.dtype
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            return int(2*n*k - k*k)
        return int(n*k - k*(k+1)/2)


    def retraction_norm(self, x, v, dt=1.):
        r"""Reject vector normalization, which is not a frame retraction."""
        raise ValueError('Use QR, polar, or Cayley retraction for Stiefel frames.')


    def inverse_retraction_norm(self, x1, x2, dt=1.):
        r"""Reject the sphere-only normalization inverse for frames."""
        raise ValueError('Normalization inverse is only defined for sphere points.')


    def retraction_cayley(self, x, v, dt=1.):
        r"""Cayley-retract a tangent vector on a rectangular Stiefel frame.

        In whitened coordinates, use the skew-Hermitian generator ``K`` with
        ``K @ X = V``. This includes the horizontal part of ``V``.
        """
        point = self._L @ x
        identity = np.eye(point.shape[0], dtype=np.result_type(point, v))
        if not np.allclose(point.conj().T @ point, np.eye(point.shape[1])):
            if np.any(v):
                raise ValueError('Cayley retraction requires an orthonormal base point.')
            return self.retraction_polar(x, v, dt)

        tangent = self._L @ v
        vertical = point.conj().T @ tangent
        generator = (tangent @ point.conj().T
                     - point @ tangent.conj().T
                     - point @ vertical @ point.conj().T)
        update = np.linalg.solve(identity - dt * generator / 2,
                                 (identity + dt * generator / 2) @ point)
        return self._Z @ update


    def inverse_retraction_cayley(self, x1, x2, dt=1.):
        r"""Invert the Stiefel Cayley retraction in its local chart.

        The inverse includes both the vertical and horizontal components of
        the tangent vector. ``I + x1^H B x2`` must be nonsingular.
        """
        overlap = x1.conj().T @ self._B @ x2
        identity = np.eye(overlap.shape[0], dtype=overlap.dtype)
        chart = identity + overlap
        horizontal = x2 - x1 @ overlap
        try:
            horizontal = np.linalg.solve(chart.T, horizontal.T).T
            vertical = np.linalg.solve(chart.conj().T,
                                       overlap - overlap.conj().T)
            vertical = np.linalg.solve(chart.T, vertical.T).T
        except np.linalg.LinAlgError as exc:
            raise ValueError('The target lies outside the Cayley chart.') from exc
        return (2. / dt) * (x1 @ vertical + horizontal)


    def inverse_retraction_polar(self, x1, x2, dt=1.):
        r"""Recover v from ``x2 = polar_retraction(x1, v, dt)`` locally.

        Requires a nonsingular local Sylvester equation and nonzero ``dt``.
        """
        pt = x1.conj().T @ self._B @ x2
        scale = solve_sylvester(pt, 2. * np.eye(x1.shape[1]), pt.conj().T)
        return (x2 @ scale - x1) / dt


    def exp(self, x, v, dt=1.):
        r"""Evaluate the Stiefel exponential map.

        Args:
            x: Point on the manifold.
            v: Tangent vector at ``x``.
            dt: Step length.

        Returns:
            The point reached by the exponential map.
        """
        point = self._L @ x
        tangent = self._L @ v
        k = x.shape[1]
        pt_tv = point.conj().T @ tangent
        identity = np.eye(k, dtype=point.dtype)
        a = np.concatenate((point, tangent), axis=1)
        b = np.block([[pt_tv, -tangent.conj().T @ tangent],
                      [identity, pt_tv]])
        return self._Z @ (a @ (scipy.linalg.expm(dt * b)[:, :k]
                                @ scipy.linalg.expm(-dt * pt_tv)))


    def log(self, x1, x2, dt=1.):
        r"""Find a local tangent whose Euclidean-metric geodesic reaches ``x2``.

        A reduced shooting problem in the span of the two whitened frames
        inverts :meth:`exp`. The result is a local logarithm; for distant
        frames, a different or shorter geodesic may exist. ``dt`` scales the
        returned initial velocity and must be nonzero.
        """
        p, q = self._L @ x1, self._L @ x2
        n, k = p.shape
        residual = q - p @ (p.conj().T @ q)
        u, singular_values, _ = np.linalg.svd(residual, full_matrices=False)
        rank = np.count_nonzero(
            singular_values > 10 * np.finfo(float).eps * max(n, k)
        )
        if rank == 0:
            return super().log(x1, x2, dt)

        complement = u[:, :rank]
        dtype = np.result_type(p, q)
        frame = np.zeros((k + rank, k), dtype=dtype)
        frame[:k] = np.eye(k)
        target = np.vstack((p.conj().T @ q, complement.conj().T @ q))
        upper = np.triu_indices(k, 1)
        off_diagonal = len(upper[0])
        complex_field = np.iscomplexobj(target)


        def unpack(coordinates):
            a = np.zeros((k, k), dtype=dtype)
            if complex_field:
                a[np.diag_indices(k)] = 1j * coordinates[:k]
                real = coordinates[k:k + off_diagonal]
                imag = coordinates[k + off_diagonal:k + 2 * off_diagonal]
                a[upper] = (real + 1j * imag) / np.sqrt(2)
                a[(upper[1], upper[0])] = (-real + 1j * imag) / np.sqrt(2)
                start = k + 2 * off_diagonal
                horizontal = (coordinates[start:start + rank * k]
                              + 1j * coordinates[start + rank * k:])
            else:
                entries = coordinates[:off_diagonal] / np.sqrt(2)
                a[upper], a[(upper[1], upper[0])] = entries, -entries
                horizontal = coordinates[off_diagonal:]
            return a, horizontal.reshape(rank, k)

        a0 = (target[:k] - target[:k].conj().T) / 2
        horizontal0 = target[k:]
        if complex_field:
            initial = np.r_[np.imag(np.diag(a0)),
                            np.sqrt(2) * np.real(a0[upper]),
                            np.sqrt(2) * np.imag(a0[upper]),
                            horizontal0.real.ravel(),
                            horizontal0.imag.ravel()]
        else:
            initial = np.r_[np.sqrt(2) * a0[upper], horizontal0.ravel()]


        def endpoint_error(coordinates):
            a, horizontal = unpack(coordinates)
            direction = np.vstack((a, horizontal))
            generator = np.block([
                [a, -direction.conj().T @ direction],
                [np.eye(k, dtype=dtype), a],
            ])
            endpoint = (np.concatenate((frame, direction), axis=1)
                        @ scipy.linalg.expm(generator)[:, :k]
                        @ scipy.linalg.expm(-a))
            error = (endpoint - target).ravel()
            return np.r_[error.real, error.imag] if complex_field else error

        solution = scipy.optimize.least_squares(
            endpoint_error, initial, ftol=1e-12, xtol=1e-12,
            gtol=1e-12, max_nfev=200
        )
        if np.linalg.norm(endpoint_error(solution.x)) > 1e-9 * np.sqrt(k):
            raise ValueError('The Stiefel logarithm did not converge locally.')
        a, horizontal = unpack(solution.x)
        return self._Z @ (p @ a + complement @ horizontal) / dt


    def parallel_transport(self, x1, x2, v1):
        r"""Transport ``v1`` along the local Stiefel geodesic from ``x1`` to ``x2``.

        Integrate the parallel-field equation in whitened coordinates. This
        evaluates exact parallel transport up to ODE solver tolerance; the
        geodesic is selected by the local logarithm.
        """
        point = self._L @ x1
        direction = self._L @ self.log(x1, x2)
        k = point.shape[1]
        a = point.conj().T @ direction
        generator = np.block([[a, -direction.conj().T @ direction],
                              [np.eye(k, dtype=point.dtype), a]])
        frame = np.concatenate((point, direction), axis=1)


        def derivative(t, vector):
            exponential = scipy.linalg.expm(t * generator)
            rotation = scipy.linalg.expm(-t * a)
            position = frame @ exponential[:, :k] @ rotation
            velocity = frame @ exponential[:, k:] @ rotation
            tangent = vector.reshape(point.shape)
            overlap = velocity.conj().T @ tangent
            normal = (overlap + overlap.conj().T) / 2
            return (-position @ normal).ravel()

        result = scipy.integrate.solve_ivp(
            derivative, (0., 1.), (self._L @ v1).ravel(),
            method='DOP853', rtol=1e-10, atol=1e-12
        )
        if not result.success:
            raise RuntimeError('Stiefel parallel transport integration failed.')
        return self._Z @ result.y[:, -1].reshape(point.shape)


    def differentiated_transport(self, x1, x2, v1):
        r"""Differentiate the selected Stiefel retraction along ``v1``.

        The step to ``x2`` is obtained with the selected inverse retraction.
        The result is tangent at ``x2`` and need not preserve norm.
        """
        eta = self.inverse_retraction(x1, x2)
        method = self.retraction.__name__
        if method == 'retraction_qr':
            point = x1 + eta
            r = x2.conj().T @ self._B @ point
            step = np.linalg.solve(r.T, v1.T).T
            c = x2.conj().T @ self._B @ step
            lower = np.tril(c, -1)
            skew = lower - lower.conj().T
            if np.iscomplexobj(c):
                skew[np.diag_indices_from(skew)] = 1j * np.imag(np.diag(c))
            return step - x2 @ (c - skew)

        if method == 'retraction_polar':
            point = x1 + eta
            h = x2.conj().T @ self._B @ point
            h = (h + h.conj().T) / 2
            c = x2.conj().T @ self._B @ v1
            skew = solve_sylvester(h, c - c.conj().T, h)
            step = np.linalg.solve(h.T, v1.T).T
            return step - x2 @ (x2.conj().T @ self._B @ step - skew)

        point = self._L @ x1
        direction = self._L @ eta
        tangent = self._L @ v1
        a = point.conj().T @ direction
        da = point.conj().T @ tangent
        if method == 'retraction_cayley':
            generator = (direction @ point.conj().T
                         - point @ direction.conj().T
                         - point @ a @ point.conj().T)
            derivative = (tangent @ point.conj().T
                          - point @ tangent.conj().T
                          - point @ da @ point.conj().T)
            identity = np.eye(point.shape[0], dtype=generator.dtype)
            target = self._L @ x2
            return self._Z @ np.linalg.solve(
                identity - generator / 2,
                derivative @ (point + target) / 2
            )

        if method == 'exp':
            k = point.shape[1]
            generator = np.block([[a, -direction.conj().T @ direction],
                                  [np.eye(k, dtype=point.dtype), a]])
            dgenerator = np.block([
                [da, -(tangent.conj().T @ direction
                       + direction.conj().T @ tangent)],
                [np.zeros((k, k), dtype=point.dtype), da]
            ])
            frame = np.concatenate((point, direction), axis=1)
            dframe = np.concatenate((np.zeros_like(point), tangent), axis=1)
            exponential = scipy.linalg.expm(generator)
            dexponential = scipy.linalg.expm_frechet(
                generator, dgenerator, compute_expm=False
            )
            rotation = scipy.linalg.expm(-a)
            drotation = scipy.linalg.expm_frechet(
                -a, -da, compute_expm=False
            )
            return self._Z @ (
                (dframe @ exponential[:, :k]
                 + frame @ dexponential[:, :k]) @ rotation
                + frame @ exponential[:, :k] @ drotation
            )

        raise NotImplementedError(
            'Differentiated transport is unavailable for this retraction.'
        )



if __name__ == '__main__':
    for manifold in (OrthogonalGroup(ndim=4),
                     OrthogonalGroup(ndim=(4, 4)),
                     Stiefel(ndim=(5, 2))):
        x = manifold.x0
        v = manifold.projection(
            x, random_matrix(x.shape, seed=7, sym=False)
        )
        v /= max(1., manifold.norm(x, v))
        y = manifold.retraction(x, 0.05 * v)
        recovered = manifold.inverse_retraction(x, y)
        assert np.allclose(manifold.retraction(x, recovered), y)
        assert manifold.check_tangent(y, manifold.vector_transport(x, y, v))
        print(type(manifold).__name__, manifold.ndim, 'passed')
