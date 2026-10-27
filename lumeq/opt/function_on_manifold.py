"""Ambient objectives bound to Riemannian manifold geometries.

Create ``FunctionObjective(energy, gradient, hessian)`` and pass it as
``obj`` to ``make_objective_manifold`` or an objective-bound geometry class.
The plain classes in ``riemannian.py`` and ``grassmann.py`` provide geometry
without storing an objective.
"""

from lumeq import np
from lumeq.opt.optimization import optimizer
from lumeq.opt.riemannian import Riemannian, OrthogonalGroup, Stiefel
from lumeq.opt.grassmann import (Grassmann, GrassmannQuotient,
                                 GrassmannProjection, GrassmannInvolution)

class FunctionObjective:
    r"""Provide an ambient energy and its first and optional second derivative.

    The default domain is a Hermitian density ``P``. ``gradient(P)`` returns
    the ambient derivative and ``hessian(P, dP)`` its directional response.
    Use ``domain='point'`` when the functions act on manifold points instead.
    """
    def __init__(self, energy, gradient, hessian=None, domain='density',
                 norm_estimate=None):
        if not callable(energy) or not callable(gradient):
            raise TypeError('energy and gradient must be callable.')
        if hessian is not None and not callable(hessian):
            raise TypeError('hessian must be callable when provided.')
        if domain not in ('density', 'point'):
            raise ValueError("domain must be 'density' or 'point'.")
        self.energy = energy
        self.gradient = gradient
        self.hessian = hessian
        self.domain = domain
        self.norm_estimate = norm_estimate


    def func(self, argument):
        return self.energy(argument)


    def func_grad(self, argument):
        return self.gradient(argument)


    def func_hess(self, argument, direction):
        if self.hessian is None:
            raise NotImplementedError('The objective has no Hessian action.')
        return self.hessian(argument, direction)



class QMFunction(FunctionObjective):
    r"""Bind DFT derivatives to orbital and nuclear optimization problems.

    Callbacks use a density matrix ``P`` and nuclear coordinates ``R``:

    ``dft_energy(P, R)`` returns the total energy, and
    ``electronic_gradient(P, R)`` returns its derivative with respect to
    ``P`` (the Fock matrix in the chosen density convention).
    ``electronic_hessian(P, dP, R)`` returns the directional Fock response.
    ``nuclear_gradient(P, R)`` returns the gradient with the shape of ``R``;
    ``nuclear_hessian(P, dR, R)`` returns its directional response.

    For nuclear optimization, ``density_at_geometry(R)`` must return a
    self-consistent density. The nuclear derivative callbacks must then give
    derivatives of that relaxed energy, including basis and orbital response
    terms where applicable. All Hessians here are actions, not matrices.
    The class supplies interfaces; the callbacks perform the DFT calculation.

    ``nuclei`` fixes the geometry when this instance is passed directly as
    ``obj`` to ``make_objective_manifold``. Alternatively, use
    ``orbital_problem(nuclei=..., ...)`` to create a fixed-geometry problem.
    """
    def __init__(self, dft_energy, electronic_gradient,
                 electronic_hessian=None, nuclear_gradient=None,
                 nuclear_hessian=None, density_at_geometry=None, nuclei=None):
        if not callable(dft_energy) or not callable(electronic_gradient):
            raise TypeError(
                'dft_energy and electronic_gradient must be callable.'
            )
        callbacks = {
            'electronic_hessian': electronic_hessian,
            'nuclear_gradient': nuclear_gradient,
            'nuclear_hessian': nuclear_hessian,
            'density_at_geometry': density_at_geometry,
        }
        for name, callback in callbacks.items():
            if callback is not None and not callable(callback):
                raise TypeError(f'{name} must be callable when provided.')

        self._dft_energy = dft_energy
        self._electronic_gradient = electronic_gradient
        self._electronic_hessian = electronic_hessian
        self._nuclear_gradient = nuclear_gradient
        self._nuclear_hessian = nuclear_hessian
        self._density_at_geometry = density_at_geometry
        self.nuclei = None if nuclei is None else np.array(nuclei, copy=True)
        super().__init__(
            lambda density: self.dft_energy(density),
            lambda density: self.electronic_gradient(density),
            (None if electronic_hessian is None else
             lambda density, direction:
             self.electronic_hessian(density, direction)),
        )


    def _geometry(self, nuclei):
        if nuclei is None:
            nuclei = self.nuclei
        if nuclei is None:
            raise ValueError('Nuclear coordinates are required.')
        return nuclei


    def dft_energy(self, density, nuclei=None):
        """Return the DFT energy for a density and nuclear geometry."""
        return self._dft_energy(density, self._geometry(nuclei))


    def electronic_gradient(self, density, nuclei=None):
        """Return the density derivative, usually a Fock matrix."""
        return self._electronic_gradient(density, self._geometry(nuclei))


    def electronic_hessian(self, density, direction, nuclei=None):
        """Return the Fock response to a density direction ``direction``."""
        if self._electronic_hessian is None:
            raise NotImplementedError(
                'No electronic Hessian action was supplied.'
            )
        return self._electronic_hessian(
            density, direction, self._geometry(nuclei)
        )


    def density_at_geometry(self, nuclei):
        """Return the self-consistent density at ``nuclei``."""
        if self._density_at_geometry is None:
            raise NotImplementedError(
                'No density_at_geometry callback was supplied.'
            )
        return self._density_at_geometry(nuclei)


    def nuclear_energy(self, nuclei):
        """Return the energy after obtaining the density at ``nuclei``."""
        density = self.density_at_geometry(nuclei)
        return self.dft_energy(density, nuclei)


    def nuclear_gradient(self, nuclei, density=None):
        """Return the relaxed nuclear gradient with the shape of ``nuclei``."""
        if self._nuclear_gradient is None:
            raise NotImplementedError('No nuclear gradient was supplied.')
        if density is None:
            density = self.density_at_geometry(nuclei)
        return self._nuclear_gradient(density, nuclei)


    def nuclear_hessian(self, nuclei, direction, density=None):
        """Return the relaxed nuclear Hessian action along ``direction``."""
        if self._nuclear_hessian is None:
            raise NotImplementedError(
                'No nuclear Hessian action was supplied.'
            )
        if density is None:
            density = self.density_at_geometry(nuclei)
        return self._nuclear_hessian(density, direction, nuclei)


    def orbital_objective(self, nuclei=None):
        """Return a density objective with the nuclear geometry fixed."""
        nuclei = np.array(self._geometry(nuclei), copy=True)
        hessian = None
        if self._electronic_hessian is not None:
            hessian = lambda density, direction: self.electronic_hessian(
                density, direction, nuclei
            )
        return FunctionObjective(
            lambda density: self.dft_energy(density, nuclei),
            lambda density: self.electronic_gradient(density, nuclei),
            hessian,
        )


    def nuclear_objective(self):
        """Return a Euclidean objective on SCF-relaxed nuclear coordinates."""
        if self._density_at_geometry is None or self._nuclear_gradient is None:
            raise ValueError(
                'Nuclear optimization needs density_at_geometry and '
                'nuclear_gradient callbacks.'
            )
        hessian = None
        if self._nuclear_hessian is not None:
            hessian = self.nuclear_hessian
        return FunctionObjective(
            self.nuclear_energy, self.nuclear_gradient, hessian,
            domain='point',
        )


    def orbital_problem(self, manifold='grassmann', *, nuclei=None,
                        representation=None, **kwargs):
        """Create a fixed-geometry manifold objective for orbitals."""
        return make_objective_manifold(
            manifold, representation=representation,
            obj=self.orbital_objective(nuclei), **kwargs
        )


    def optimize_nuclei(self, x0, method='bfgs', **kwargs):
        r"""Optimize nuclear coordinates and return ``(xs, ys, gradient)``.

        The default uses Euclidean BFGS. Newton and trust-region methods need
        a caller-supplied ``hessf`` and ``tangent_solver`` in ``kwargs``.
        """
        objective = self.nuclear_objective()
        options = dict(
            func=objective.func, gradf=objective.func_grad,
            retraction=lambda point, step: point + step,
            transport=lambda old, new, vector: vector,
            preconditioner=None, hess_inv=None,
            x0=np.asarray(x0),
        )
        options.update(kwargs)
        return optimizer(method, **options)



class ObjectiveOnManifold(Riemannian, FunctionObjective):
    r"""Convert an ambient objective to a concrete manifold objective.

    Instantiate a concrete subclass below with ``obj=``. Geometry comes from
    the other base class; this class supplies energy, Riemannian gradient,
    Hessian action, and the optimizer interface.
    """
    def __init__(self, *args, obj, **kwargs):
        if (not callable(getattr(obj, 'func', None))
                or not callable(getattr(obj, 'func_grad', None))):
            raise TypeError('obj must provide func and func_grad methods.')
        hessian = getattr(obj, 'hessian', getattr(obj, 'func_hess', None))
        estimate = getattr(obj, 'norm_estimate',
                           getattr(obj, 'func_norm', None))
        FunctionObjective.__init__(
            self, obj.func, obj.func_grad, hessian,
            domain=getattr(obj, 'domain', 'density'), norm_estimate=estimate
        )
        self.obj = obj
        super().__init__(*args, **kwargs)


    def func(self, point):
        r"""Evaluate the ambient objective at a point or its density."""
        argument = (point if self.domain == 'point'
                    else self.to_density(point))
        return FunctionObjective.func(self, argument)


    def _density_derivatives(self, point, direction=None):
        density = self.to_density(point)
        derivative = FunctionObjective.func_grad(self, density)
        derivative = (derivative + derivative.conj().T) / 2
        response = None
        if direction is not None:
            response = FunctionObjective.func_hess(
                self,
                density, self.density_tangent(point, direction)
            )
            response = (response + response.conj().T) / 2

        return self._pullback_density_derivatives(
            point, direction, derivative, response
        )


    def _pullback_density_derivatives(self, point, direction,
                                      derivative, response):
        r"""Pull density derivatives back to vector or frame coordinates."""
        ambient = 2 * derivative @ point
        action = (None if response is None else
                  2 * (response @ point + derivative @ direction))
        return ambient, action


    def _initial_point(self):
        return self.x0


    def _save_result(self, final):
        r"""Save representation-specific state after optimization, if any."""
        pass


    def func_grad(self, point):
        r"""Return the Riemannian gradient of the attached objective."""
        if self.domain == 'point':
            ambient = FunctionObjective.func_grad(self, point)
        else:
            ambient, _ = self._density_derivatives(point)
        return self.riemannian_gradient(point, ambient)


    def func_hess(self, point, direction):
        r"""Return the Riemannian Hessian action along ``direction``."""
        if self.hessian is None:
            raise NotImplementedError('The objective has no Hessian action.')
        if self.domain == 'point':
            ambient = FunctionObjective.func_grad(self, point)
            action = FunctionObjective.func_hess(self, point, direction)
        else:
            ambient, action = self._density_derivatives(point, direction)
        return self.riemannian_hessian(point, direction, ambient, None, action)


    def optimizer(self, opt_method='bfgs', *, nmax=50,
                  ls_method='armijo', cg_method='fletcher_reeves',
                  thresh=1e-8, grad_thresh=1e-6, disp_thresh=1e-6,
                  restart=True, **kwargs):
        r"""Minimize the objective and return ``(xs, ys, gradient)``.

        Newton and trust-region methods require ``hessf`` and a compatible
        ``tangent_solver``. ``func_hess(point, direction)`` provides a Hessian
        action to solvers that accept one. GDIIS requires affine points.
        """
        if ls_method is None:
            scale = (2 * self.norm_estimate(self._Z)
                     if callable(self.norm_estimate) else 1.)
            ls_method = 1 / scale if scale > 0 else 1.

        kwargs.setdefault('preconditioner', getattr(self, 'preconditioner', None))
        kwargs.setdefault('hess_inv', None)
        kwargs.setdefault('hessf', None)
        kwargs.setdefault('tangent_solver', None)
        kwargs.setdefault('dt', 1.)
        if opt_method in ('newton_2nd', 'newton', 'trust_region') and (
                kwargs['hessf'] is None or kwargs['tangent_solver'] is None):
            raise ValueError('This optimizer requires hessf and tangent_solver.')
        if opt_method == 'gdiis':
            raise ValueError('GDIIS requires affine Euclidean coordinates.')

        x0 = kwargs.pop('x0', self._initial_point())
        result = optimizer(
            opt_method, func=self.func, gradf=self.func_grad,
            retraction=self.retraction, transport=self.transport,
            inner_product=self.inner_product,
            norm=lambda v: self.norm(None, v),
            distance=self.distance_of_points, x0=x0,
            ls_method=ls_method, cg_method=cg_method,
            nmax=nmax, thresh=thresh, grad_thresh=grad_thresh,
            disp_thresh=disp_thresh, restart=restart, **kwargs
        )
        self._save_result(result[0][-1].copy())
        return result



class OrthogonalGroupObjective(OrthogonalGroup, ObjectiveOnManifold):
    """Objective bound to a sphere or square orthogonal-group geometry."""



class StiefelObjective(Stiefel, ObjectiveOnManifold):
    """Objective bound to a Stiefel geometry."""



class GrassmannQuotientObjective(GrassmannQuotient, ObjectiveOnManifold):
    """Objective bound to a quotient Grassmann geometry."""

    def _save_result(self, final):
        self.x = final
        self.p = self.to_projector(final)



class GrassmannProjectionObjective(GrassmannProjection, ObjectiveOnManifold):
    """Objective bound to a projector Grassmann geometry."""

    def _pullback_density_derivatives(self, point, direction,
                                      derivative, response):
        return derivative, response


    def _initial_point(self):
        return self.p0


    def _save_result(self, final):
        self.p = final
        self.x = self.to_frame(final)



class GrassmannInvolutionObjective(GrassmannInvolution, ObjectiveOnManifold):
    """Objective bound to an involution Grassmann geometry."""

    def _pullback_density_derivatives(self, point, direction,
                                      derivative, response):
        ambient = self._Z @ derivative @ self._Z / 2
        action = (None if response is None else
                  self._Z @ response @ self._Z / 2)
        return ambient, action


    def _save_result(self, final):
        self.j = final
        self.p = self.to_density(final)
        self.x = self.to_frame(self.p)



def make_objective_manifold(manifold='grassmann', *, representation=None,
                            obj, **kwargs):
    r"""Construct an objective-bound manifold from a name or geometry class.

    ``manifold`` accepts ``'sphere'``, ``'orthogonal'``, ``'stiefel'``, or
    ``'grassmann'``, as well as their geometry classes. For Grassmann, choose
    ``representation='quotient'`` (default), ``'projection'``, or
    ``'involution'``. Specific Grassmann names and classes also work without
    ``representation``. Remaining arguments initialize the selected manifold.

    Example:
        ``make_objective_manifold('grassmann', representation='projection',
        obj=objective, ndim=(n, k), B=overlap)``
    """
    grassmann_types = {
        'quotient': GrassmannQuotientObjective,
        'projection': GrassmannProjectionObjective,
        'involution': GrassmannInvolutionObjective,
    }
    if manifold in ('grassmann', Grassmann):
        selected = grassmann_types.get(representation or 'quotient')
        if selected is None:
            raise ValueError('Unknown Grassmann representation.')
    else:
        if representation is not None:
            raise ValueError('representation is only used with Grassmann.')
        manifold_types = {
            'sphere': OrthogonalGroupObjective,
            'orthogonal': OrthogonalGroupObjective,
            'orthogonal_group': OrthogonalGroupObjective,
            OrthogonalGroup: OrthogonalGroupObjective,
            'stiefel': StiefelObjective,
            Stiefel: StiefelObjective,
            'grassmann_quotient': GrassmannQuotientObjective,
            GrassmannQuotient: GrassmannQuotientObjective,
            'grassmann_projection': GrassmannProjectionObjective,
            GrassmannProjection: GrassmannProjectionObjective,
            'grassmann_involution': GrassmannInvolutionObjective,
            GrassmannInvolution: GrassmannInvolutionObjective,
        }
        selected = manifold_types.get(manifold)
        if selected is None:
            raise ValueError('Unknown manifold.')
    return selected(obj=obj, **kwargs)



if __name__ == '__main__':
    kernel = np.diag([1., 2., 3., 4., 5.])
    objective = FunctionObjective(
        lambda p: float(np.real(np.trace(kernel @ p))),
        lambda p: kernel,
        lambda p, dp: np.zeros_like(kernel),
    )
    for representation in ('quotient', 'projection', 'involution'):
        problem = make_objective_manifold(
            'grassmann', representation=representation,
            obj=objective, ndim=(5, 2)
        )
        point = problem._initial_point()
        direction = problem.func_grad(point)
        assert problem.check_tangent(
            point, problem.func_hess(point, direction)
        )
        xs, ys, gradient = problem.optimizer('bfgs', nmax=8)
        assert ys[-1] < ys[0]
        assert problem.check_tangent(xs[-1], gradient)
        print(representation, ys[0], '->', ys[-1])
