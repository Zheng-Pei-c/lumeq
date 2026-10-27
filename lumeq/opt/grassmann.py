"""Grassmann manifold geometries."""

from lumeq import np, scipy
from lumeq.opt.riemannian import (Riemannian, OrthogonalGroup, Stiefel,
                                 solve_sylvester)
from lumeq.utils import random_matrix

class Grassmann(Riemannian):
    r"""Base for Grassmann manifolds with ``k``-planes in ``n`` dimensions.

    Concrete subclasses choose how a subspace and its tangent are stored.
    The quotient class uses frames; projection and involution classes use
    square Hermitian matrices. Every class keeps a B-orthonormal frame ``x``
    and its AO density projector ``p = x x^H`` for the latest point.
    """
    def initialize_point(self, x0=None, **kwargs):
        r"""Initialize the frame and projector from ``x0`` or ``p0``."""
        p0 = kwargs.pop('p0', None)
        if x0 is not None and p0 is not None:
            raise ValueError('Provide either x0 or p0, not both.')
        if p0 is not None:
            self.p0 = np.asarray(p0).copy()
            self.x0 = self.to_frame(self.p0)
        else:
            super().initialize_point(x0, **kwargs)
            self.p0 = self.to_projector(self.x0)
        self.x = self.x0.copy()
        self.p = self.p0.copy()
        self.frame0 = self.x0.copy()


    @staticmethod
    def to_projector(frame):
        r"""Return the AO density projector ``X X^H`` of a frame."""
        return frame @ frame.conj().T


    def to_frame(self, p):
        r"""Choose a B-orthonormal frame for a projector; gauge is arbitrary."""
        whitened = self._L @ p @ self._L
        _, vectors = np.linalg.eigh((whitened + whitened.conj().T) / 2)
        return self._Z @ vectors[:, -self.ndim[1]:]


    def dimension(self, dtype=None):
        r"""Return the real dimension of the represented Grassmann manifold."""
        n, k = self.ndim
        if dtype is None:
            dtype = self.x0.dtype
        complex_field = np.issubdtype(np.dtype(dtype), np.complexfloating)
        return int((2 if complex_field else 1) * (n-k) * k)



class GrassmannQuotient(Grassmann):
    r"""Grassmann manifold represented as a quotient of the Stiefel manifold.

    Points are B-normalized coefficient frames ``X`` of shape ``(n, k)``.
    A tangent ``V`` is a horizontal array of shape ``(n, k)`` satisfying
    ``X^H B V = 0``. Right multiplication by an orthogonal (real) or unitary
    (complex) matrix changes the frame but represents the same subspace.
    The metric is the real Frobenius metric after whitening with ``B^1/2``.

    The corresponding AO projector and projector tangent are ``(n, n)``
    matrices ``P = X X^H`` and ``U = V X^H + X V^H``. Given ``P``, ``U``,
    and a frame ``X`` for ``P``, recover the horizontal tangent as
    ``V = U B X``.

    A point-based objective must be invariant under ``X -> X U`` for unitary
    ``U``; density-based objectives have this invariance automatically.

    Notes:
        The quotient-space relations used here are

        .. code-block:: text

            Gr(k, n) = St(k, n) / O(k)       (real case)
                     = St(k, n) / U(k)       (complex case)

            P = X X^dagger
            T_X Gr(k, n) = {V in F^(n x k) | X^dagger B V = 0}
    """
    def check_sanity(self):
        r"""Require a B-orthonormal frame of shape ``(n, k)``."""
        if len(self.ndim) != 2:
            raise ValueError('Grassmann quotient points must be frame matrices.')
        x = self.x0
        if not np.allclose(x.conj().T @ self._B @ x,
                           np.eye(self.ndim[1])):
            raise ValueError('The initial frame is not B-orthonormal.')


    def retraction_qr(self, x, v, dt=1.):
        r"""QR-retract a horizontal frame tangent."""
        return OrthogonalGroup.retraction_qr(self, x, v, dt)


    def retraction_polar(self, x, v, dt=1.):
        r"""Polar-retract a horizontal frame tangent."""
        return OrthogonalGroup.retraction_polar(self, x, v, dt)


    def retraction_cayley(self, x, v, dt=1.):
        r"""Cayley-retract a horizontal frame tangent."""
        return Stiefel.retraction_cayley(self, x, v, dt)


    def distance_of_points(self, x1, x2):
        r"""Return the norm of principal angles between two subspaces."""
        overlap = x1.conj().T @ self._B @ x2
        residual = x2 - x1 @ overlap
        cosines = np.linalg.svd(overlap, compute_uv=False)
        sines = np.linalg.svd(self._L @ residual, compute_uv=False)
        angles = np.arctan2(sines[::-1], cosines)
        return np.linalg.norm(angles)


    def projection(self, x, v):
        r"""Project onto the Grassmann tangent space.

        Args:
            x: Point on the manifold.
            v: Ambient-space vector or matrix.

        Returns:
            A tangent vector orthogonal to the span of ``x`` under ``B``.
        """
        return v - x @ (x.conj().T @ (self._B @ v))


    def inverse_retraction_polar(self, x1, x2, dt=1.):
        r"""Recover a horizontal polar-retraction step from two nearby frames.

        ``x1^H B x2`` must be invertible. Different column gauges for ``x2``
        yield the same horizontal step. ``dt`` must be nonzero.
        """
        pt = x1.conj().T @ self._B @ x2
        return (np.linalg.solve(pt.T, x2.T).T - x1) / dt


    def inverse_retraction_qr(self, x1, x2, dt=1.):
        r"""Recover a horizontal QR step, independent of the target gauge."""
        return self.inverse_retraction_polar(x1, x2, dt)


    def inverse_retraction_cayley(self, x1, x2, dt=1.):
        r"""Recover a horizontal Cayley step from nearby subspaces.

        Align the target frame to the base frame before using the Stiefel
        inverse. The overlap must be nonsingular in this local chart.
        """
        overlap = x1.conj().T @ self._B @ x2
        u, singular_values, vh = np.linalg.svd(overlap)
        if np.min(singular_values) <= 1e-12:
            raise ValueError('The target lies outside the Cayley chart.')
        aligned = x2 @ (vh.conj().T @ u.conj().T)
        return Stiefel.inverse_retraction_cayley(self, x1, aligned, dt)


    def weingarten(self, x, v, normal):
        r"""Return the Grassmann shape-operator correction ``-V (X^H B N)``."""
        return -v @ (x.conj().T @ self._B @ normal)


    def _exp_rotation(self, x, v, dt=1., full_matrices=False):
        r"""Exponentiate a horizontal tangent or a full B-skew generator.

        ``v`` has shape ``(n, k)`` by default. With ``full_matrices=True``,
        it is an ``(n, n)`` generator in the original B-weighted coordinates.
        The returned rotation acts on whitened frames.
        """
        if full_matrices:
            if v.shape != self._B.shape:
                raise ValueError('A full generator must have shape (n, n).')
            generator = self._L @ v @ self._Z
        else:
            if v.shape != x.shape:
                raise ValueError('A horizontal tangent must match the point shape.')
            point = self._L @ x
            tangent = self._L @ v
            generator = tangent @ point.conj().T - point @ tangent.conj().T
        return scipy.linalg.expm(dt * generator)


    def exp(self, x, v, dt=1., full_matrices=False):
        r"""Evaluate the Grassmann exponential by compact SVD or full rotation.

        Args:
            x: Point on the manifold.
            v: Tangent vector at ``x``.
            dt: Step length.
            full_matrices: Build a full ambient rotation from a full SVD
                instead of using the compact SVD formula.

        Returns:
            The point reached by the exponential map.
        """
        point = self._L @ x
        tangent = self._L @ v
        u, s, vt = np.linalg.svd(tangent, full_matrices=full_matrices)
        angles = s * dt
        if full_matrices:
            noccupied = point.shape[1]
            nvirtual = point.shape[0] - noccupied
            if nvirtual > noccupied:
                active = noccupied  # extra virtual directions stay fixed
            else:
                active = nvirtual  # extra occupied directions stay fixed
            base = (point @ vt.conj().T)[:, :active]
            complement = u[:, :active]
            angles = angles[:active]
            cosine_minus_one = -2 * np.sin(angles / 2)**2
            sine = np.sin(angles)
            identity = np.eye(point.shape[0], dtype=np.result_type(point, tangent))
            rotation = (identity
                        + (base * cosine_minus_one) @ base.conj().T
                        + (complement * cosine_minus_one) @ complement.conj().T
                        + (complement * sine) @ base.conj().T
                        - (base * sine) @ complement.conj().T)
            return self._Z @ (rotation @ point)

        y = np.einsum('ij,kj,k->ik', point, vt.conj(), np.cos(angles))
        y = (y + np.einsum('ij,j->ij', u, np.sin(angles))) @ vt
        return self._Z @ y


    def log(self, x1, x2, dt=1.):
        r"""Evaluate the Grassmann logarithmic map.

        The implementation follows Algorithm 1 of
        ``10.1007/s10444-023-10090-8`` and avoids an explicit matrix inverse
        in the reconstruction step.

        Args:
            x1: Base point on the manifold.
            x2: Target point on the manifold.
            dt: Nonzero step length. The returned velocity satisfies
                ``exp(x1, velocity, dt) == x2`` as a subspace.

        Returns:
            A horizontal tangent at ``x1``. The shortest logarithm is not
            unique when a principal angle equals pi/2 (the cut locus).
        """
        ytx = x2.conj().T @ self._B @ x1
        u, _, vt = np.linalg.svd(ytx, full_matrices=False)

        point = self._L @ x1
        target = (self._L @ x2) @ (u @ vt)
        residual = target - point @ (point.conj().T @ target)
        u, s, vt = np.linalg.svd(residual, full_matrices=False)
        angles = np.arcsin(np.clip(s, 0, 1)) / dt
        return self._Z @ np.einsum('ij,j,jk->ik', u, angles, vt)


    def get_complement(self, x):
        r"""Choose a B-orthonormal virtual complement of an occupied frame.

        ``x`` has shape ``(n, k)``. The returned ``q`` has shape
        ``(n, n-k)`` and satisfies ``x^H B q = 0`` and ``q^H B q = I``.
        Its column basis is not unique. For ``k=n``, it has zero columns.
        """
        x = np.asarray(x)
        if x.shape != self.ndim:
            raise ValueError('The occupied frame must have shape (n, k).')
        if not np.allclose(x.conj().T @ self._B @ x, np.eye(self.ndim[1])):
            raise ValueError('The occupied frame must be B-orthonormal.')
        complement = scipy.linalg.null_space((self._L @ x).conj().T)
        return self._Z @ complement


    def horizontal_lift(self, x, v, lift=True, representation=None):
        r"""Convert between quotient and Stiefel tangent representations.

        ``X`` is an occupied frame, ``Q`` a B-orthonormal virtual complement,
        and ``V`` a horizontal tangent. The conversion depends on ``lift``:

        .. list-table::
           :header-rows: 1

           * - Representation
             - ``x``
             - ``lift=True``: input to output
             - ``lift=False``: input to output
           * - ``'complement'``
             - ``Q``: ``(n, n-k)``
             - ``T``: ``(n-k, k)`` to ``V = Q T``: ``(n, k)``
             - ``V``: ``(n, k)`` to ``T = Q^H B V``: ``(n-k, k)``
           * - ``'generator'``
             - ``X``: ``(n, k)``
             - ``G``: ``(n, n)`` to ``V = G X``: ``(n, k)``
             - ``V``: ``(n, k)`` to
               ``G = (V X^H - X V^H) B``: ``(n, n)``

        Args:
            x: Occupied frame for ``'generator'``, or a B-orthonormal virtual
                complement of shape ``(n, n-k)`` for ``'complement'``.
            v: Full generator ``(n, n)``, complement coordinates
                ``(n-k, k)``, or horizontal vector ``(n, k)``. Leading batch
                dimensions are supported.
            lift: If ``True``, convert a full generator ``(n, n)`` or
                complement coordinates ``(n-k, k)`` to a horizontal vector
                ``(n, k)``. If ``False``, convert a horizontal vector
                ``(n, k)`` to the selected generator or coordinates.
            representation: ``'generator'`` or ``'complement'``. If omitted,
                inferred from shapes; specify it when ``n == 2*k`` makes
                occupied and complement frames indistinguishable.

        Returns:
            A horizontal ``(n, k)`` vector when ``lift=True``; otherwise an
            ``(n, n)`` generator or ``(n-k, k)`` coordinate matrix.
        """
        n, k = self.ndim
        if representation is None:
            if lift and v.shape[-2:] == (n, n):
                representation = 'generator'
            elif x.shape == (n, n-k) and (lift or n-k != k):
                representation = 'complement'
            else:
                representation = 'generator'
        if representation not in ('generator', 'complement'):
            raise ValueError('Unknown tangent representation.')
        expected_x = (n, k) if representation == 'generator' else (n, n-k)
        if lift:
            expected_v = (n, n) if representation == 'generator' else (n-k, k)
        else:
            expected_v = (n, k)
        if x.shape != expected_x or v.shape[-2:] != expected_v:
            raise ValueError('Point and tangent do not match the representation.')
        if representation == 'complement':
            return x @ v if lift else x.conj().T @ self._B @ v
        if lift:
            return v @ x
        # The full generator must be B-skew and satisfy generator @ x = v.
        return (v @ x.conj().T - x @ np.swapaxes(v.conj(), -1, -2)) @ self._B


    def get_tangent(self, x, v):
        r"""Extract occupied/virtual tangent coordinates using complement x."""
        return self.horizontal_lift(x, v, False, representation='complement')


    def check_tangent(self, x, v):
        r"""Return whether ``v`` is horizontal at ``x``."""
        return (x.shape == v.shape and
                bool(np.allclose(x.conj().T @ self._B @ v, 0)))


    def parallel_transport(self, x1, x2, v1):
        r"""Parallel-transport a horizontal vector along a shortest geodesic.

        The geodesic and its transport are local away from the cut locus.
        """
        direction = self.log(x1, x2)
        point = self._L @ x1
        u, angles, vt = np.linalg.svd(self._L @ direction, full_matrices=False)
        base = point @ vt.conj().T
        sine, cosine = np.sin(angles), np.cos(angles)
        endpoint = (base * cosine + u * sine) @ vt
        rotated = -base * sine + u * cosine
        vector = self._L @ v1
        transported = vector + (rotated - u) @ (u.conj().T @ vector)
        target_gauge = endpoint.conj().T @ (self._L @ x2)
        return self._Z @ transported @ target_gauge


    def vector_transport(self, x1, x2, v1):
        r"""Project ``v1`` horizontally and express it in the target gauge."""
        overlap = x1.conj().T @ self._B @ x2
        u, _, vh = np.linalg.svd(overlap)
        gauge = u @ vh
        return self.projection(x2, v1) @ gauge


    def differentiated_transport(self, x1, x2, v1):
        r"""Differentiate the retraction in the target frame's gauge."""
        eta = self.inverse_retraction(x1, x2)
        endpoint = self.retraction(x1, eta)
        tangent = Stiefel.differentiated_transport(self, x1, endpoint, v1)
        gauge = endpoint.conj().T @ self._B @ x2
        return self.projection(endpoint, tangent) @ gauge


    def tangent_solver(self, x, grad, hess, method='direct'):
        r"""Solve a Sylvester model in occupied/virtual coordinates.

        Requires ``B=I``. ``hess`` is a Hermitian left-action matrix whose
        occupied/virtual blocks define the local model; it is not an
        arbitrary four-index orbital Hessian. The reduced equation requires
        nonzero occupied/virtual curvature gaps.
        """
        if method != 'direct':
            raise ValueError('Unknown tangent solver method.')
        if not np.allclose(self._B, np.eye(x.shape[0])):
            raise NotImplementedError('The Newton tangent solver requires B = I.')
        complement = scipy.linalg.null_space(x.conj().T)
        if complement.shape[1] == 0:
            return np.zeros_like(x)
        projected_hessian = complement.conj().T @ hess @ complement
        overlap = x.conj().T @ hess @ x
        overlap = (overlap + overlap.conj().T) / 2
        rhs = -complement.conj().T @ grad
        step = solve_sylvester(projected_hessian, rhs, -overlap)
        return complement @ step



class GrassmannProjection(Grassmann):
    r"""Grassmann manifold with AO density projectors as optimization points.

    ``x0`` is the initial B-orthonormal frame of shape ``(n, k)``; ``p0`` is
    its AO projector. Optimizer points ``P`` are Hermitian ``(n, n)`` matrices
    satisfying ``P B P = P``. A tangent ``U`` is a Hermitian ``(n, n)`` matrix
    satisfying ``U B P + P B U = U``.
    For a B-orthonormal frame ``X`` of shape ``(n, k)``, the corresponding
    horizontal frame tangent ``V`` has shape ``(n, k)`` and satisfies
    ``V = U B X``; conversely ``U = V X^H + X V^H``.

    ``p`` holds the latest optimized projector and ``x`` a corresponding
    B-orthonormal ``(n, k)`` frame. ``p0`` and ``frame0`` stay initial.
    The frame's gauge is arbitrary. ``to_projector`` and ``to_frame`` convert
    between representations when needed.
    """
    def initialize_point(self, x0=None, **kwargs):
        r"""Initialize from a projector ``p0`` or a B-normalized frame ``x0``."""
        if kwargs.get('p0') is not None:
            super().initialize_point(x0, **kwargs)
            return
        frame = (random_matrix(self.ndim, seed=kwargs.get('seed', 12345),
                               sym=False)
                 if x0 is None else np.asarray(x0))
        if frame.shape != self.ndim:
            raise ValueError('The initial frame must have shape (n, k).')
        self.x0 = OrthogonalGroup.retraction_polar(
            self, frame, np.zeros_like(frame)
        )
        self.p0 = self.to_projector(self.x0)
        self.x = self.x0.copy()
        self.p = self.p0.copy()
        self.frame0 = self.x0.copy()


    def check_sanity(self):
        r"""Require a Hermitian rank-k matrix satisfying ``P B P = P``."""
        p = self.p0
        n, k = self.ndim
        if (p.shape != (n, n)
                or not np.allclose(p, p.conj().T)
                or not np.allclose(p @ self._B @ p, p)
                or not np.isclose(np.trace(self._B @ p), k)):
            raise ValueError('The initial point is not a rank-k B-projector.')


    def inner_product(self, u, v):
        r"""Return half the whitened Frobenius product of projector tangents."""
        whiten_u = self._L @ u @ self._L
        whiten_v = self._L @ v @ self._L
        return float(np.real(np.vdot(whiten_u, whiten_v)) / 2)


    def projection(self, p, v):
        r"""Project an ambient matrix onto the tangent space at projector p."""
        p_white = self._L @ p @ self._L
        ambient = self._L @ v @ self._L
        ambient = (ambient + ambient.conj().T) / 2
        complement = np.eye(p_white.shape[0], dtype=p_white.dtype) - p_white
        tangent = (p_white @ ambient @ complement
                   + complement @ ambient @ p_white)
        return self._Z @ tangent @ self._Z


    def check_tangent(self, p, v):
        r"""Check Hermiticity and the derivative of ``P B P = P``."""
        return (p.shape == v.shape
                and np.allclose(v, v.conj().T)
                and np.allclose(v @ self._B @ p + p @ self._B @ v, v))


    def riemannian_gradient(self, p, grad):
        r"""Convert an ambient Frobenius gradient to the projector metric."""
        return self.projection(p, 2 * self._Binv @ grad @ self._Binv)


    @staticmethod
    def to_density(p):
        r"""The optimization point is already the AO density projector."""
        return p


    @staticmethod
    def density_tangent(p, v):
        return v


    def riemannian_hessian(self, p, v, grad=None, hess=None,
                          hess_dot_v=None):
        r"""Apply an ambient Hessian and the projector curvature term."""
        if grad is None:
            raise ValueError('An ambient gradient is required.')
        if hess_dot_v is None:
            hess_dot_v = (np.zeros_like(grad) if hess is None else
                            hess(v) if callable(hess) else hess @ v)
        grad = (grad + grad.conj().T) / 2
        hess_dot_v = (hess_dot_v + hess_dot_v.conj().T) / 2
        p_white = self._L @ p @ self._L
        tangent = self._L @ v @ self._L
        kernel = self._Z @ grad @ self._Z
        response = self._Z @ hess_dot_v @ self._Z
        derivative = 2 * (tangent @ kernel + kernel @ tangent
                          - 2 * tangent @ kernel @ p_white
                          - 2 * p_white @ kernel @ tangent + response)
        complement = np.eye(p_white.shape[0], dtype=p_white.dtype) - p_white
        projected = (p_white @ derivative @ complement
                     + complement @ derivative @ p_white)
        return self._Z @ projected @ self._Z


    def tangent_solver(self, p, grad, hess, method='direct'):
        r"""Solve a Sylvester model through occupied-frame coordinates."""
        if method != 'direct':
            raise ValueError('Unknown tangent solver method.')
        if not np.allclose(self._B, np.eye(self.ndim[0])):
            raise NotImplementedError('The Newton tangent solver requires B = I.')
        frame = self.to_frame(p)
        complement = scipy.linalg.null_space(frame.conj().T)
        if complement.shape[1] == 0:
            return np.zeros_like(p)
        projected_hessian = complement.conj().T @ hess @ complement
        overlap = frame.conj().T @ hess @ frame
        overlap = (overlap + overlap.conj().T) / 2
        rhs = -complement.conj().T @ grad @ frame
        step = complement @ solve_sylvester(
            projected_hessian, rhs, -overlap
        )
        return step @ frame.conj().T + frame @ step.conj().T


    def distance_of_points(self, p1, p2):
        r"""Return principal-angle distance between two projectors."""
        x1, x2 = self.to_frame(p1), self.to_frame(p2)
        overlap = x1.conj().T @ self._B @ x2
        residual = x2 - x1 @ overlap
        cosines = np.linalg.svd(overlap, compute_uv=False)
        sines = np.linalg.svd(self._L @ residual, compute_uv=False)
        return np.linalg.norm(np.arctan2(sines[::-1], cosines))


    def _graph_retraction(self, p, v, dt=1.):
        r"""Project the graph of a tangent onto a rank-k orthogonal projector."""
        p_white = self._L @ p @ self._L
        tangent = self._L @ v @ self._L
        identity = np.eye(p_white.shape[0], dtype=p_white.dtype)
        graph = p_white + dt * tangent @ p_white
        gram = graph.conj().T @ graph + identity - p_white
        updated = graph @ np.linalg.solve(gram, graph.conj().T)
        updated = (updated + updated.conj().T) / 2
        return self._Z @ updated @ self._Z


    def retraction_qr(self, p, v, dt=1.):
        r"""Update the projector through a local graph retraction."""
        return self._graph_retraction(p, v, dt)


    def retraction_polar(self, p, v, dt=1.):
        r"""Update the projector through a local graph retraction."""
        return self._graph_retraction(p, v, dt)


    def retraction_cayley(self, p, v, dt=1.):
        r"""Cayley-rotate the whitened projector without choosing a frame."""
        p_white = self._L @ p @ self._L
        tangent = self._L @ v @ self._L
        generator = tangent @ p_white - p_white @ tangent
        identity = np.eye(p_white.shape[0], dtype=p_white.dtype)
        rotation = np.linalg.solve(identity - dt * generator / 2,
                                   identity + dt * generator / 2)
        return self._Z @ rotation @ p_white @ rotation.conj().T @ self._Z


    def inverse_retraction_polar(self, p1, p2, dt=1.):
        r"""Recover the tangent of a local projector graph update."""
        p1_white = self._L @ p1 @ self._L
        p2_white = self._L @ p2 @ self._L
        identity = np.eye(p1_white.shape[0], dtype=p1_white.dtype)
        overlap = p1_white @ p2_white @ p1_white + identity - p1_white
        off_diagonal = (identity - p1_white) @ p2_white @ p1_white
        graph = np.linalg.solve(overlap.T, off_diagonal.T).T
        return self._Z @ ((graph + graph.conj().T) / dt) @ self._Z


    def inverse_retraction_qr(self, p1, p2, dt=1.):
        r"""Recover the tangent of a local projector graph update."""
        return self.inverse_retraction_polar(p1, p2, dt)


    def _rotation_between(self, p1, p2):
        r"""Return the minimal unitary rotation between nearby projectors."""
        p1_white = self._L @ p1 @ self._L
        p2_white = self._L @ p2 @ self._L
        identity = np.eye(p1_white.shape[0], dtype=p1_white.dtype)
        intertwiner = (p2_white @ p1_white
                       + (identity - p2_white) @ (identity - p1_white))
        u, singular_values, vh = np.linalg.svd(intertwiner)
        if np.min(singular_values) <= 1e-12:
            raise ValueError('The target lies outside the local projector chart.')
        return u @ vh


    def inverse_retraction_cayley(self, p1, p2, dt=1.):
        r"""Recover the tangent of a local Cayley projector update."""
        rotation = self._rotation_between(p1, p2)
        identity = np.eye(rotation.shape[0], dtype=rotation.dtype)
        generator = (2 / dt) * np.linalg.solve(
            rotation + identity, rotation - identity
        )
        p_white = self._L @ p1 @ self._L
        tangent = generator @ p_white - p_white @ generator
        tangent = (tangent + tangent.conj().T) / 2
        return self._Z @ tangent @ self._Z


    def exp(self, p, v, dt=1.):
        r"""Follow the exact projector geodesic by conjugation."""
        p_white = self._L @ p @ self._L
        tangent = self._L @ v @ self._L
        generator = tangent @ p_white - p_white @ tangent
        rotation = scipy.linalg.expm(dt * generator)
        return self._Z @ rotation @ p_white @ rotation.conj().T @ self._Z


    def log(self, p1, p2, dt=1.):
        r"""Return a projector tangent for the shortest local geodesic."""
        rotation = self._rotation_between(p1, p2)
        generator = scipy.linalg.logm(rotation) / dt
        p_white = self._L @ p1 @ self._L
        tangent = generator @ p_white - p_white @ generator
        tangent = (tangent + tangent.conj().T) / 2
        return self._Z @ tangent @ self._Z


    def parallel_transport(self, p1, p2, v1):
        r"""Parallel-transport a projector tangent along the short geodesic."""
        rotation = self._rotation_between(p1, p2)
        tangent = self._L @ v1 @ self._L
        return self._Z @ rotation @ tangent @ rotation.conj().T @ self._Z


    def differentiated_transport(self, p1, p2, v1):
        r"""Differentiate the selected projector retraction at its local step."""
        p_white = self._L @ p1 @ self._L
        direction = self._L @ self.inverse_retraction(p1, p2) @ self._L
        tangent = self._L @ v1 @ self._L
        identity = np.eye(p_white.shape[0], dtype=p_white.dtype)
        method = self.retraction.__name__
        if method in ('retraction_qr', 'retraction_polar'):
            graph = p_white + direction @ p_white
            dgraph = tangent @ p_white
            gram = graph.conj().T @ graph + identity - p_white
            dgram = dgraph.conj().T @ graph + graph.conj().T @ dgraph
            inverse_graph = np.linalg.solve(gram, graph.conj().T)
            derivative = (dgraph @ inverse_graph
                          + graph @ np.linalg.solve(
                              gram, dgraph.conj().T - dgram @ inverse_graph
                          ))
        else:
            generator = direction @ p_white - p_white @ direction
            dgenerator = tangent @ p_white - p_white @ tangent
            if method == 'retraction_cayley':
                denominator = identity - generator / 2
                rotation = np.linalg.solve(denominator,
                                           identity + generator / 2)
                drotation = np.linalg.solve(
                    denominator, dgenerator @ (rotation + identity) / 2
                )
            else:
                rotation, drotation = scipy.linalg.expm_frechet(
                    generator, dgenerator, compute_expm=True
                )
            derivative = (drotation @ p_white @ rotation.conj().T
                          + rotation @ p_white @ drotation.conj().T)
        derivative = (derivative + derivative.conj().T) / 2
        return self.projection(p2, self._Z @ derivative @ self._Z)


    def vector_transport(self, p1, p2, v1):
        r"""Project a tangent directly onto the target projector tangent."""
        return self.projection(p2, v1)



class GrassmannInvolution(Grassmann):
    r"""Grassmann points stored as whitened Hermitian involutions.

    A point ``J = 2 B^1/2 P B^1/2 - I`` satisfies ``J^H = J`` and ``J^2 = I``.
    It represents the same rank-k subspace as the AO density projector ``P``.
    Geometry methods and ``x0`` use involutions. ``j``, ``p``, and ``x`` save
    the latest optimized involution, AO projector, and occupied frame;
    ``p0`` and ``frame0`` store the initial projector and frame.

    References: Z. Lai, L.-H. Lim, and K. Ye, "Simpler Grassmannian
    Optimization," arXiv:2009.13502 (2020),
    https://arxiv.org/abs/2009.13502; "Simple Matrix Expressions for the
    Curvatures of Grassmannian," Foundations of Computational Mathematics
    26, 2053–2089 (2026), https://doi.org/10.1007/s10208-025-09723-9.
    These papers treat real matrices; Hermitian matrices here extend the
    representation to complex subspaces, and ``B^1/2`` whitens the metric.
    """
    def initialize_point(self, x0=None, **kwargs):
        r"""Initialize the involution, projector, and occupied frame."""
        supplied_p0 = kwargs.get('p0')
        if x0 is not None and supplied_p0 is not None:
            raise ValueError('Provide either x0 or p0, not both.')
        if x0 is not None:
            j0 = np.asarray(x0)
            if j0.ndim != 2 or j0.shape[0] != j0.shape[1]:
                raise ValueError('The initial involution must be square.')
            n = j0.shape[0]
            if self.ndim == (n, n):
                k = int(round((n + np.real(np.trace(j0))) / 2))
                self.ndim = (n, k)
            self.p0 = self._to_projector(j0)
        elif supplied_p0 is not None:
            self.p0 = np.asarray(supplied_p0).copy()
            j0 = self._to_involution(self.p0)
        else:
            frame = random_matrix(self.ndim, seed=kwargs.get('seed', 12345),
                                  sym=False)
            frame = OrthogonalGroup.retraction_polar(
                self, frame, np.zeros_like(frame)
            )
            self.p0 = self.to_projector(frame)
            j0 = self._to_involution(self.p0)
        self.x0 = j0.copy()
        self.j = self.x0.copy()
        self.p = self.p0.copy()
        self.x = self.to_frame(self.p0)
        self.frame0 = self.x.copy()


    def _to_projector(self, j):
        identity = np.eye(j.shape[0], dtype=j.dtype)
        return self._Z @ ((j + identity) / 2) @ self._Z


    def _to_involution(self, p):
        identity = np.eye(p.shape[0], dtype=p.dtype)
        return 2 * self._L @ p @ self._L - identity


    def _to_projector_tangent(self, tangent):
        return self._Z @ (tangent / 2) @ self._Z


    def _to_involution_tangent(self, tangent):
        return 2 * self._L @ tangent @ self._L


    def to_density(self, j):
        r"""Return the AO density projector represented by ``j``."""
        return self._to_projector(j)


    def density_tangent(self, j, v):
        return self._to_projector_tangent(v)


    def check_sanity(self):
        r"""Require a Hermitian involution with exactly ``k`` plus modes."""
        j = self.x0
        n, k = self.ndim
        if (j.shape != (n, n)
                or not np.allclose(j, j.conj().T)
                or not np.allclose(j @ j, np.eye(n))
                or not np.isclose(np.trace(j), 2*k-n)):
            raise ValueError('The initial point is not a rank-k involution.')


    def inner_product(self, u, v):
        r"""Return the quotient-induced metric on involution tangents."""
        return float(np.real(np.vdot(u, v)) / 8)


    def projection(self, j, v):
        r"""Project onto Hermitian matrices anticommuting with ``j``."""
        hermitian = (v + v.conj().T) / 2
        return (hermitian - j @ hermitian @ j) / 2


    def check_tangent(self, j, v):
        r"""Check Hermiticity and the derivative of ``J^2 = I``."""
        return (j.shape == v.shape
                and np.allclose(v, v.conj().T)
                and np.allclose(j @ v + v @ j, 0))


    def riemannian_gradient(self, j, grad):
        r"""Convert an ambient Frobenius gradient to the involution metric."""
        return 8 * self.projection(j, grad)


    def riemannian_hessian(self, j, v, grad=None, hess=None,
                          hess_dot_v=None):
        r"""Apply an ambient Hessian and the involution curvature term."""
        if grad is None:
            raise ValueError('An ambient gradient is required.')
        if hess_dot_v is None:
            hess_dot_v = (np.zeros_like(grad) if hess is None else
                            hess(v) if callable(hess) else hess @ v)
        grad = (grad + grad.conj().T) / 2
        hess_dot_v = (hess_dot_v + hess_dot_v.conj().T) / 2
        derivative = 8 * hess_dot_v - 4 * (v @ grad @ j + j @ grad @ v)
        return self.projection(j, derivative)


    def tangent_solver(self, j, grad, hess, method='direct'):
        r"""Solve a Sylvester model through projector coordinates."""
        step = GrassmannProjection.tangent_solver(
            self, self._to_projector(j),
            self._to_projector_tangent(grad), hess, method=method
        )
        return self._to_involution_tangent(step)


    def distance_of_points(self, j1, j2):
        return GrassmannProjection.distance_of_points(
            self, self._to_projector(j1), self._to_projector(j2)
        )


    def _graph_retraction(self, j, v, dt):
        r"""Retract through the graph of the plus-one eigenspace of ``j``."""
        identity = np.eye(j.shape[0], dtype=j.dtype)
        p = (j + identity) / 2
        graph = p + dt * v @ p / 2
        gram = graph.conj().T @ graph + identity - p
        updated = graph @ np.linalg.solve(gram, graph.conj().T)
        updated = (updated + updated.conj().T) / 2
        return 2 * updated - identity


    def _inverse_graph(self, j1, j2, dt):
        r"""Recover an involution tangent from a local graph retraction."""
        identity = np.eye(j1.shape[0], dtype=j1.dtype)
        p1, p2 = (j1 + identity) / 2, (j2 + identity) / 2
        overlap = p1 @ p2 @ p1 + identity - p1
        off_diagonal = (identity - p1) @ p2 @ p1
        graph = np.linalg.solve(overlap.T, off_diagonal.T).T
        return 2 * (graph + graph.conj().T) / dt


    def retraction_qr(self, j, v, dt=1.):
        return self._graph_retraction(j, v, dt)


    def retraction_polar(self, j, v, dt=1.):
        return self._graph_retraction(j, v, dt)


    def retraction_cayley(self, j, v, dt=1.):
        r"""Cayley-rotate the involution using its skew-Hermitian generator."""
        generator = v @ j / 2
        identity = np.eye(j.shape[0], dtype=j.dtype)
        rotation = np.linalg.solve(identity - dt * generator / 2,
                                   identity + dt * generator / 2)
        return rotation @ j @ rotation.conj().T


    def inverse_retraction_qr(self, j1, j2, dt=1.):
        return self._inverse_graph(j1, j2, dt)


    def inverse_retraction_polar(self, j1, j2, dt=1.):
        return self._inverse_graph(j1, j2, dt)


    def inverse_retraction_cayley(self, j1, j2, dt=1.):
        r"""Recover the tangent of a local Cayley involution update."""
        rotation = self._rotation_between(j1, j2)
        identity = np.eye(j1.shape[0], dtype=rotation.dtype)
        generator = (2 / dt) * np.linalg.solve(
            rotation + identity, rotation - identity
        )
        return self.projection(j1, generator @ j1 - j1 @ generator)


    def exp(self, j, v, dt=1.):
        r"""Follow the involution geodesic by unitary conjugation."""
        rotation = scipy.linalg.expm(dt * v @ j / 2)
        return rotation @ j @ rotation.conj().T


    def log(self, j1, j2, dt=1.):
        r"""Return the tangent of the shortest local involution geodesic."""
        generator = scipy.linalg.logm(j2 @ j1) / (2 * dt)
        return self.projection(j1, generator @ j1 - j1 @ generator)


    def _rotation_between(self, j1, j2):
        r"""Return the minimal unitary rotation between nearby involutions."""
        product = j2 @ j1
        identity = np.eye(j1.shape[0], dtype=product.dtype)
        if np.min(np.linalg.svd(identity + product, compute_uv=False)) <= 1e-12:
            raise ValueError('The target lies outside the local involution chart.')
        return scipy.linalg.expm(scipy.linalg.logm(product) / 2)


    def parallel_transport(self, j1, j2, v1):
        r"""Parallel-transport a tangent along the shortest local geodesic."""
        rotation = self._rotation_between(j1, j2)
        return rotation @ v1 @ rotation.conj().T


    def differentiated_transport(self, j1, j2, v1):
        r"""Differentiate the selected involution retraction at its local step."""
        identity = np.eye(j1.shape[0], dtype=j1.dtype)
        method = self.retraction.__name__
        if method in ('retraction_qr', 'retraction_polar'):
            p = (j1 + identity) / 2
            direction = self._inverse_graph(j1, j2, 1.)
            graph = p + direction @ p / 2
            dgraph = v1 @ p / 2
            gram = graph.conj().T @ graph + identity - p
            dgram = dgraph.conj().T @ graph + graph.conj().T @ dgraph
            inverse_graph = np.linalg.solve(gram, graph.conj().T)
            derivative = 2 * (dgraph @ inverse_graph
                              + graph @ np.linalg.solve(
                                  gram, dgraph.conj().T
                                  - dgram @ inverse_graph
                              ))
        else:
            direction = (self.inverse_retraction_cayley(j1, j2)
                         if method == 'retraction_cayley'
                         else self.log(j1, j2))
            generator = direction @ j1 / 2
            dgenerator = v1 @ j1 / 2
            if method == 'retraction_cayley':
                denominator = identity - generator / 2
                rotation = np.linalg.solve(denominator,
                                           identity + generator / 2)
                drotation = np.linalg.solve(
                    denominator, dgenerator @ (rotation + identity) / 2
                )
            else:
                rotation, drotation = scipy.linalg.expm_frechet(
                    generator, dgenerator, compute_expm=True
                )
            derivative = (drotation @ j1 @ rotation.conj().T
                          + rotation @ j1 @ drotation.conj().T)
        return self.projection(j2, derivative)


    def vector_transport(self, j1, j2, v1):
        r"""Project an involution tangent onto the target tangent space."""
        return self.projection(j2, v1)



if __name__ == '__main__':
    for cls, initial in ((GrassmannQuotient, 'x0'),
                         (GrassmannProjection, 'p0'),
                         (GrassmannInvolution, 'x0')):
        manifold = cls(ndim=(5, 2))
        x = getattr(manifold, initial)
        v = manifold.projection(
            x, random_matrix(x.shape, seed=9, sym=False)
        )
        v /= max(1., manifold.norm(x, v))
        y = manifold.retraction(x, 0.05 * v)
        recovered = manifold.inverse_retraction(x, y)
        round_trip = manifold.retraction(x, recovered)
        assert manifold.distance_of_points(round_trip, y) < 1e-8
        assert manifold.check_tangent(y, manifold.vector_transport(x, y, v))
        print(cls.__name__, 'passed')
