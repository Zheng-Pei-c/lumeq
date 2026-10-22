r"""Optimization over Euclidean arrays and manifold tangent coordinates.

Updates have the form ``x_next = retraction(x, alpha * direction)``.
Objectives return real scalars, gradients have the shape of the point, and
manifold gradients and search directions must be tangent at that point.
The default inner product is ``Re vdot(u, v)``. Convergence requires a small
gradient, a small energy error, and a small displacement. An
initially stationary point returns without a step.
Displacement uses Euclidean/Frobenius distance by default; optimizers accept a
``distance(old, new)`` callback for other point geometries.
"""

from numbers import Real

from lumeq import np


class LineSearchError(RuntimeError):
    """A search direction or trial sequence failed the line-search conditions."""


def _identity_preconditioner(vector):
    """Return the input unchanged when no preconditioner is supplied."""
    return vector


def _inner_product(vector_a, vector_b):
    """Real Frobenius inner product for real or complex arrays."""
    return float(np.real(np.vdot(vector_a, vector_b)))


def _point_distance(old_point, new_point):
    """Return Euclidean/Frobenius distance between two points."""
    return np.linalg.norm(new_point - old_point)


def _hessian_action(hessian, vector):
    """Apply a Hessian action callable or left-multiplication matrix."""
    return hessian(vector) if callable(hessian) else hessian @ vector


def _quadratic_reduction(gradient, hessian, step,
                         inner_product=_inner_product):
    """Return the reduction predicted by a quadratic model at ``step``."""
    return (-inner_product(gradient, step)
            - 0.5 * inner_product(step, _hessian_action(hessian, step)))


def converged(iteration, yi, error, displacement, gradient_norm,
              thresh=1e-8, disp_thresh=1e-6, grad_thresh=1e-6):
    """Report progress and decide whether the optimizer should stop.

    ``displacement`` is the distance between accepted points, as measured by
    the optimizer's ``distance(old, new)`` callback.

    Return ``'converged'`` only when the energy error, displacement, and
    gradient norm all meet their thresholds. Return ``'stationary'`` when the
    gradient falls below ``grad_thresh * 1e-3`` before the other criteria, or
    when the initial point (indicated by ``iteration=None``) already meets
    ``grad_thresh``. Otherwise return ``None``. Call this only for accepted
    steps.
    """
    if iteration is None:
        if gradient_norm <= grad_thresh:
            print('Initial point is stationary.\n')
            return 'stationary'
        return None

    print(
        ('Iteration:%3d  Value:%12.8f  Error:%8.4e  '
         'Displacement:%8.4e  Grad:%8.4e')
        % (iteration, yi, error, displacement, gradient_norm),
    )
    if (error <= thresh
            and displacement <= disp_thresh
            and gradient_norm <= grad_thresh):
        print('Optimization converged.\n')
        return 'converged'
    if gradient_norm <= grad_thresh * 1e-3:
        print('Gradient threshold reached before all convergence criteria.\n')
        return 'stationary'
    return None


def optimizer(method, *args, **kwargs):
    """Run the selected optimization algorithm.

    Args:
        method: ``'gradient_descent'`` (or ``'gd'``),
            ``'conjugate_gradient'`` (or ``'cg'``), ``'newton_2nd'`` (or
            ``'newton'``), ``'trust_region'``, ``'bfgs'``, ``'gdiis'``, or an
            optimizer
            callable.
        *args: Positional arguments for the selected optimizer.
        **kwargs: Keyword arguments for the selected optimizer.

    Returns:
        The selected optimizer's ``(xs, ys, gradient)`` result.

    Raises:
        NotImplementedError: If ``method`` is unknown.

    The optimizers have different required arguments. Supply the arguments
    accepted by the chosen optimizer after ``method``.
    """
    methods = {
        'gradient_descent': gradient_descent, 'gd': gradient_descent,
        'conjugate_gradient': conjugate_gradient, 'cg': conjugate_gradient,
        'newton_2nd': newton_2nd, 'newton': newton_2nd,
        'trust_region': trust_region, 'bfgs': bfgs, 'gdiis': gdiis,
    }
    if callable(method):
        algorithm = method
    else:
        try:
            algorithm = methods[method]
        except (KeyError, TypeError):
            raise NotImplementedError(
                'Optimization method %s is not implemented.' % method
            ) from None
    return algorithm(*args, **kwargs)


def gradient_descent(func, gradf, retraction, x0, ls_method='armijo',
                     nmax=50, thresh=1e-8, grad_thresh=1e-6,
                     disp_thresh=1e-6, *args, **kwargs):
    r"""Run gradient descent with a configurable line-search strategy.

    Args:
        func: Objective function.
        gradf: Gradient of ``func``.
        retraction: Retraction or update map.
        x0: Initial point.
        ls_method: Line-search method name or fixed step size.
        nmax: Maximum number of iterations.
        thresh: Maximum energy error for the joint stopping test. The error is
            measured against ``ymin`` when supplied, otherwise the last value.
        grad_thresh: Convergence threshold on the gradient norm.
        disp_thresh: Maximum point displacement for the joint stopping test.
        *args: Additional positional arguments forwarded to the line search.
        **kwargs: Line-search options. ``ymin`` optionally sets the reference
            energy. ``inner_product(u, v)`` and ``norm(v)`` optionally replace
            the Euclidean metric. ``distance(old, new)`` optionally replaces
            Euclidean/Frobenius point distance for convergence. Cached
            ``gradient`` and ``value`` are supplied to the line search.

    Returns:
        ``(xs, ys, gradient)`` containing independent point snapshots, real
        objective values, and the final gradient. Histories include the
        initial point; reaching ``nmax`` does not imply convergence.
    """
    print('%s line-search method is used in gradient descent.' % ls_method)
    ls_method = line_search(ls_method)
    vector_norm = kwargs.get('norm', np.linalg.norm)
    distance = kwargs.get('distance', _point_distance)
    ymin = kwargs.get('ymin', None)

    xi = np.array(x0, copy=True)
    y0 = func(xi)
    gradient = gradf(xi)
    xs, ys = [xi.copy()], [y0]

    if converged(None, y0, 0, 0, vector_norm(gradient),
                 thresh, disp_thresh, grad_thresh):
        return np.asarray(xs), np.asarray(ys), gradient

    for i in range(nmax):
        direction = -gradient
        search_kwargs = dict(kwargs, gradient=gradient, value=y0)
        step_size = ls_method(
            func, gradf, retraction, xi, direction, *args, **search_kwargs
        )

        x_prev = xi.copy()
        dx = step_size * direction
        xi = retraction(xi, dx)
        yi = func(xi)
        gradient = gradf(xi)

        xs.append(xi.copy())
        ys.append(yi)

        error = abs(yi - ymin) if ymin is not None else abs(yi - y0)
        displacement = distance(x_prev, xi)
        grad_norm = vector_norm(gradient)
        if converged(i, yi, error, displacement, grad_norm,
                     thresh, disp_thresh, grad_thresh):
            break

        y0 = yi

    return np.asarray(xs), np.asarray(ys), gradient


def line_search(method='armijo'):
    r"""Return a line-search routine.

    Args:
        method: ``'armijo'``, ``'backtracking'``, ``'newton_raphson'``, a
            nonnegative real step size, or a compatible callable.

    Returns:
        A callable line-search routine.

    Raises:
        NotImplementedError: If the requested line-search method is unknown.
    """
    if callable(method):
        return method
    if isinstance(method, Real) and not isinstance(method, bool):
        if not np.isfinite(method) or method < 0:
            raise ValueError('A fixed step must be finite and nonnegative.')
        return ls_steepest(method)
    methods = {'armijo': ls_backtracking, 'backtracking': ls_backtracking,
               'newton_raphson': ls_newton_raphson}
    if method not in methods:
        raise NotImplementedError('Line-search method %s is not implemented.' % method)
    return methods[method]


def ls_steepest(step=1.):
    r"""Return a constant-step line-search callable."""

    def kernel(*args, **kwargs):
        """Return the fixed step, ignoring the optimizer callback arguments."""
        return step

    return kernel


def ls_backtracking(func, gradf, retraction, x0, direction=None,
                    tau=.5, r=1e-4, alpha=1., alpha_min=1e-12,
                    max_backtracks=50, **kwargs):
    r"""Run backtracking line search.

    Args:
        func: Objective function.
        gradf: Gradient of ``func``.
        retraction: Retraction or update map.
        x0: Current point.
        direction: Tangent search direction; defaults to the negative gradient.
        tau: Contraction factor for the step size.
        r: Armijo sufficient-decrease parameter.
        alpha: Initial trial displacement norm. The step coefficient starts
            at ``alpha / norm(direction)``.
        alpha_min: Minimum admissible step coefficient, after normalizing the
            initial trial size by ``norm(direction)``.
        max_backtracks: Maximum number of trial evaluations.
        **kwargs: Optional ``gradient`` and ``value`` caches, ``norm(v)``, and
            ``inner_product(u, v)``. Other optimizer options are ignored.

    Returns:
        The accepted step size.

    Raises:
        LineSearchError: If the search direction is not descending or no
            acceptable step size is found.
    """
    if not (0 < tau < 1 and 0 < r < 1 and alpha > 0 and alpha_min > 0
            and max_backtracks > 0):
        raise ValueError('Invalid Armijo parameters.')
    y0 = kwargs['value'] if 'value' in kwargs else func(x0)
    gradient = kwargs['gradient'] if 'gradient' in kwargs else gradf(x0)
    if direction is None:
        direction = -gradient

    inner_product = kwargs.get('inner_product', _inner_product)
    vector_norm = kwargs.get('norm', np.linalg.norm)

    norm = vector_norm(direction)
    if norm == 0:
        return 0.0

    descent = - inner_product(gradient, direction)
    if descent <= 0:
        raise LineSearchError('Line search received a non-descent direction.')

    alpha /= norm

    for _ in range(max_backtracks):
        if alpha < alpha_min:
            break
        dx = alpha * direction
        xi = retraction(np.array(x0, copy=True), dx)
        dy = y0 - func(xi)

        if dy >= r * alpha * descent:
            return alpha

        alpha *= tau

    raise LineSearchError(
        'Backtracking line search failed to find an acceptable step size.'
    )


ls_armijo = ls_backtracking


def ls_newton_raphson(func, gradf, retraction, x0, direction=None, **kwargs):
    r"""Run a Newton-Raphson line search along a search direction.

    This method minimizes a one-dimensional quadratic model along the supplied
    search direction. If the directional curvature is nonpositive or the trial
    step fails the Armijo decrease condition, it falls back to backtracking.

    Args:
        func: Objective function.
        gradf: Gradient of ``func``.
        retraction: Retraction or update map.
        x0: Current point.
        direction: Tangent search direction; defaults to the negative gradient.
        **kwargs: ``hessf`` is a matrix or callback ``hessf(x)`` returning a
            matrix or action callable. On a manifold it must model curvature
            along the chosen update. ``alpha_max`` caps the displacement norm.
            ``r`` sets the Armijo sufficient-decrease fraction (default
            ``1e-4``). Other options are forwarded to backtracking.

    Returns:
        The accepted step size.

    Raises:
        ValueError: If no Hessian information is supplied.
        LineSearchError: If the supplied direction is not a descent direction.
    """
    hessf = kwargs.get('hessf')
    alpha_max = kwargs.get('alpha_max', 1.0)
    if not np.isfinite(alpha_max) or alpha_max <= 0:
        raise ValueError('alpha_max must be finite and positive.')
    r = kwargs.get('r', 1e-4)
    if not 0 < r < 1:
        raise ValueError('The Armijo parameter r must be between 0 and 1.')
    inner_product = kwargs.get('inner_product', _inner_product)
    vector_norm = kwargs.get('norm', np.linalg.norm)

    gradient = kwargs['gradient'] if 'gradient' in kwargs else gradf(x0)
    value = kwargs['value'] if 'value' in kwargs else func(x0)
    backtracking_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {'direction', 'hessf', 'alpha_max'}
    }

    if direction is None:
        direction = -gradient
    if hessf is None:
        raise ValueError('Line-search method newton_raphson requires hessf.')

    norm = vector_norm(direction)
    if norm == 0:
        return 0.0

    df0 = inner_product(gradient, direction)
    if df0 >= 0:
        raise LineSearchError('Line search received a non-descent direction.')

    hessian = hessf(x0) if callable(hessf) else hessf
    directional_curvature = inner_product(direction, _hessian_action(hessian, direction))

    if directional_curvature <= 0 or np.isclose(directional_curvature, 0):
        return ls_backtracking(
            func, gradf, retraction, x0, direction, **backtracking_kwargs
        )

    alpha = min(-df0 / directional_curvature, alpha_max / norm)
    if alpha <= 0:
        return ls_backtracking(
            func, gradf, retraction, x0, direction, **backtracking_kwargs
        )

    xi = retraction(np.array(x0, copy=True), alpha * direction)
    if value - func(xi) >= r * alpha * (-df0):
        return alpha

    return ls_backtracking(
        func, gradf, retraction, x0, direction, **backtracking_kwargs
    )


def conjugate_gradient(func, gradf, retraction, transport, preconditioner, x0,
                       ls_method='armijo', cg_method='fletcher_reeves',
                       nmax=50, thresh=1e-8, grad_thresh=1e-6,
                       disp_thresh=1e-6, restart=True, *args, **kwargs):
    r"""Run nonlinear conjugate gradient optimization.

    Note:
        Unlike ``gradient_descent()``, this method does not reset the search
        direction to the negative gradient at every iteration. It combines the
        current negative gradient with a transported copy of the previous
        search direction, scaled by the conjugate-gradient coefficient
        ``beta``.

    Args:
        func: Objective function.
        gradf: Gradient of ``func``.
        retraction: Retraction or update map.
        transport: Callback ``transport(old_point, new_point, vector)`` that
            returns a vector tangent at the new point.
        preconditioner: Preconditioner callable. If ``None``, the identity is
            used.
        x0: Initial point.
        ls_method: Line-search method name or fixed step size.
        cg_method: Conjugate-gradient update rule.
        nmax: Maximum number of iterations.
        thresh: Maximum absolute energy change for the joint stopping test.
        grad_thresh: Convergence threshold on the gradient norm.
        disp_thresh: Maximum point displacement for the joint stopping test.
        restart: Restart from the preconditioned negative gradient when the
            CG direction loses sufficient descent or its line search fails.
        *args: Additional positional arguments forwarded to helpers.
        **kwargs: Line-search options and optional ``inner_product(u, v)`` and
            ``norm(v)`` callbacks. ``distance(old, new)`` optionally replaces
            Euclidean/Frobenius point distance for convergence. Line-search
            options are not passed to beta update rules. The preconditioner
            must preserve tangency and give a positive gradient/preconditioned-
            gradient inner product.

    Returns:
        ``(xs, ys, gradient)`` containing point snapshots, objective values,
        and the final gradient. Histories include the initial point.
    """
    print('%s line-search and %s update are used in conjugate gradient.' % (ls_method, cg_method))
    ls_method = line_search(ls_method)
    cg_method = choose_direction(cg_method)

    inner_product = kwargs.get('inner_product', _inner_product)
    vector_norm = kwargs.get('norm', np.linalg.norm)
    distance = kwargs.get('distance', _point_distance)

    if preconditioner is None:
        preconditioner = _identity_preconditioner

    xi = np.array(x0, copy=True)
    y0 = func(xi)
    g0 = gradf(xi)
    pg0 = preconditioner(g0)
    xs, ys = [xi.copy()], [y0]

    if converged(None, y0, 0, 0, vector_norm(g0),
                 thresh, disp_thresh, grad_thresh):
        return np.asarray(xs), np.asarray(ys), g0

    direction = -pg0
    final_gradient = g0

    for i in range(nmax):
        if inner_product(g0, direction) >= 0:
            raise RuntimeError('The preconditioner must produce a descent direction.')
        search_kwargs = dict(kwargs, gradient=g0, value=y0)
        try:
            step_size = ls_method(
                func, gradf, retraction, xi, direction, *args, **search_kwargs
            )
        except LineSearchError:
            if not restart or np.array_equal(direction, -pg0):
                raise
            direction = -pg0
            step_size = ls_method(
                func, gradf, retraction, xi, direction, *args, **search_kwargs
            )

        x_prev = xi.copy()
        dx = step_size * direction
        xi = retraction(xi, dx)
        yi = func(xi)
        gi = gradf(xi)
        final_gradient = gi

        xs.append(xi.copy())
        ys.append(yi)

        error = abs(yi - y0)
        displacement = distance(x_prev, xi)
        grad_norm = vector_norm(gi)
        if converged(i, yi, error, displacement, grad_norm,
                     thresh, disp_thresh, grad_thresh):
            break

        pgi = preconditioner(gi)
        # transport vectors from the previous point to the current point
        transported_g0 = transport(x_prev, xi, g0)
        transported_direction = transport(x_prev, xi, direction)
        beta = cg_method(
            g0=g0,
            g1=gi,
            pg0=pg0,
            pg1=pgi,
            transported_g0=transported_g0,
            inner_product=inner_product,
        )
        # Combine directions in the tangent space of the new point.
        direction = beta * transported_direction - pgi

        if inner_product(gi, direction) >= -0.1 * inner_product(gi, pgi):
            if restart:
                direction = -pgi
            else:
                raise RuntimeError(
                    'Conjugate-gradient update produced a non-descent direction.'
                )

        y0 = yi
        g0 = gi
        pg0 = pgi

    return np.asarray(xs), np.asarray(ys), final_gradient


def choose_direction(method='fletcher_reeves'):
    r"""Return a conjugate-gradient update rule.

    Args:
        method: ``'fletcher_reeves'``, ``'polak_ribiere'``, or a callable
            accepting the keyword arguments supplied by conjugate_gradient.

    Returns:
        A callable conjugate-gradient update routine.

    Raises:
        NotImplementedError: If the requested update rule is unknown.
    """
    if callable(method):
        return method
    methods = {'fletcher_reeves': cg_fletcher_reeves,
               'polak_ribiere': cg_polak_ribiere}
    if method not in methods:
        raise NotImplementedError('Conjugate-gradient beta scheme %s is not implemented.' % method)
    return methods[method]


def cg_fletcher_reeves(*, g0, g1, pg0, pg1, transported_g0=None,
                       inner_product=_inner_product):
    r"""Return ``<g1, pg1> / <g0, pg0>`` for preconditioned CG.

    Returns zero if the previous denominator is nonpositive. Gradients and
    preconditioned gradients must be expressed at their respective points.
    """
    denominator = inner_product(g0, pg0)
    if denominator <= 0:
        return 0.0

    return inner_product(g1, pg1) / denominator


def cg_polak_ribiere(*, g0, g1, pg0, pg1, transported_g0,
                     inner_product=_inner_product):
    r"""Return the nonnegative preconditioned Polak-Ribiere coefficient.

    The gradient difference is ``g1 - transported_g0``, evaluated in the
    current tangent space. A nonpositive denominator restarts CG.
    """
    denominator = inner_product(g0, pg0)
    if denominator <= 0:
        return 0.0

    dg = g1 - transported_g0

    return max(0.0, inner_product(pg1, dg) / denominator)


def newton_2nd(func, gradf, hessf, retraction, tangent_solver, x0, dt=1.,
               nmax=50, thresh=1e-8, grad_thresh=1e-6,
               disp_thresh=1e-6, *args, **kwargs):
    r"""Run fixed-step Newton updates from a caller-supplied tangent solver.

    This local method does not enforce descent or regularize an indefinite
    Hessian. The tangent solver is responsible for the Newton equation.

    Args:
        func: Objective function.
        gradf: Gradient of ``func``.
        hessf: Hessian data or callback ``hessf(x)`` returning such data.
        retraction: Update map ``retraction(point, tangent_step)``.
        tangent_solver: Callback ``solver(x, gradient, hessian)`` returning
            the tangent step, including its descent sign.
        x0: Initial point.
        dt: Scale factor applied to the solver's tangent step.
        nmax: Maximum number of iterations.
        thresh: Maximum energy error for the joint stopping test. The error is
            measured against ``ymin`` when supplied, otherwise the last value.
        grad_thresh: Convergence threshold on the gradient norm.
        disp_thresh: Maximum point displacement for the joint stopping test.
        *args: Additional positional arguments reserved for extensions.
        **kwargs: ``ymin`` optionally sets the reference energy,
            ``norm(v)`` the tangent-vector norm, and ``distance(old, new)``
            the point distance for convergence. Other options are reserved.

    Returns:
        ``(xs, ys, gradient)`` containing point snapshots, objective values,
        and the final gradient. Histories include the initial point.
    """
    ymin = kwargs.get('ymin', None)
    vector_norm = kwargs.get('norm', np.linalg.norm)
    distance = kwargs.get('distance', _point_distance)

    xi = np.array(x0, copy=True)
    y0 = func(xi)
    gradient = gradf(xi)
    xs, ys = [xi.copy()], [y0]

    if converged(None, y0, 0, 0, vector_norm(gradient),
                 thresh, disp_thresh, grad_thresh):
        return np.asarray(xs), np.asarray(ys), gradient

    for i in range(nmax):
        hessian = hessf(xi) if callable(hessf) else hessf
        v = tangent_solver(xi, gradient, hessian)

        x_prev = xi.copy()
        xi = retraction(xi, dt * v)
        yi = func(xi)
        gradient = gradf(xi)

        xs.append(xi.copy())
        ys.append(yi)

        error = abs(yi - ymin) if ymin is not None else abs(yi - y0)
        displacement = distance(x_prev, xi)
        grad_norm = vector_norm(gradient)
        if converged(i, yi, error, displacement, grad_norm,
                     thresh, disp_thresh, grad_thresh):
            break

        y0 = yi

    return np.asarray(xs), np.asarray(ys), gradient


def trust_region(func, gradf, hessf, retraction, tangent_solver, x0, dt=1.,
                 nmax=50, thresh=1e-8, grad_thresh=1e-6,
                 disp_thresh=1e-6, *args, **kwargs):
    r"""Accept or reject norm-clipped trial steps using a quadratic model.

    The supplied solver proposes a tangent step. The scaled step is clipped
    to the radius and compared with a Cauchy (negative-gradient)
    step using the quadratic model. This is not a full trust-region subproblem
    solver. Hessian data must describe the Riemannian Hessian action when used
    on a manifold.

    Args:
        func: Objective function.
        gradf: Gradient of ``func``.
        hessf: Matrix or callback ``hessf(x)`` returning a matrix or Hessian
            action callable.
        retraction: Update map ``retraction(point, tangent_step)``.
        tangent_solver: Solver for the tangent-space trial step.
        x0: Initial point.
        dt: Positive scale factor for the trial tangent step. The radius
            constrains ``dt * trial`` after clipping.
        nmax: Maximum number of iterations.
        thresh: Maximum absolute energy change for the joint stopping test.
        grad_thresh: Convergence threshold on the gradient norm.
        disp_thresh: Maximum point displacement for the joint stopping test.
        *args: Additional positional arguments reserved for extensions.
        **kwargs: ``radius`` (1), ``radius_max`` (10), ``eta1`` (0.25),
            ``eta2`` (0.75), ``shrink`` (0.25), and ``expand`` (2) control
            acceptance and radius changes. ``norm(v)`` and
            ``inner_product(u, v)`` set the tangent metric;
            ``distance(old, new)`` sets point distance for convergence.

    Returns:
        ``(xs, ys, gradient)`` containing the initial point, accepted steps,
        their objective values, and the final gradient.
    """
    inner_product = kwargs.get('inner_product', _inner_product)
    vector_norm = kwargs.get('norm', np.linalg.norm)
    distance = kwargs.get('distance', _point_distance)

    radius = kwargs.get('radius', 1.0)
    radius_max = kwargs.get('radius_max', 10.0)
    eta1 = kwargs.get('eta1', 0.25)
    eta2 = kwargs.get('eta2', 0.75)
    shrink = kwargs.get('shrink', 0.25)
    expand = kwargs.get('expand', 2.0)
    if not (all(np.isfinite(value) for value in
                (radius, radius_max, eta1, eta2, shrink, expand, dt))
            and 0 < radius <= radius_max and 0 < eta1 < eta2 < 1
            and 0 < shrink < 1 and expand > 1 and dt > 0):
        raise ValueError('Invalid trust-region parameters.')

    xi = np.array(x0, copy=True)
    y0 = func(xi)
    gradient = gradf(xi)
    grad_norm = vector_norm(gradient)
    xs, ys = [xi.copy()], [y0]
    if converged(None, y0, 0, 0, grad_norm,
                 thresh, disp_thresh, grad_thresh):
        return np.asarray(xs), np.asarray(ys), gradient

    for i in range(nmax):
        hessian = hessf(xi) if callable(hessf) else hessf
        step = dt * np.asarray(tangent_solver(xi, gradient, hessian))
        step_norm = vector_norm(step)
        if step_norm > radius:
            step = step * (radius / step_norm)

        # The Cauchy step guarantees positive predicted reduction for a
        # nonzero gradient, including when the Hessian is indefinite.
        curvature = inner_product(gradient, _hessian_action(hessian, gradient))
        cauchy_scale = radius / grad_norm
        if curvature > 0:
            cauchy_scale = min(cauchy_scale, grad_norm**2 / curvature)
        cauchy_step = -cauchy_scale * gradient

        predicted = _quadratic_reduction(
            gradient, hessian, step, inner_product
        )
        cauchy_predicted = _quadratic_reduction(
            gradient, hessian, cauchy_step, inner_product
        )
        if not np.isfinite(predicted) or cauchy_predicted > predicted:
            step = cauchy_step
            predicted = cauchy_predicted

        x_trial = retraction(xi.copy(), step)
        y_trial = func(x_trial)
        actual_reduction = y0 - y_trial
        if not (np.isfinite(predicted) and predicted > 0
                and np.isfinite(actual_reduction)):
            rho = -np.inf
        else:
            rho = actual_reduction / predicted

        if rho < eta1:
            radius *= shrink
        else:
            xi = x_trial
            yi = y_trial
            xs.append(xi.copy())
            ys.append(yi)

            error = abs(yi - y0)
            displacement = distance(xs[-2], xi)
            y0 = yi
            gradient = gradf(xi)
            grad_norm = vector_norm(gradient)
            if rho > eta2 and np.isclose(vector_norm(step), radius):
                radius = min(radius * expand, radius_max)

        if rho >= eta1 and converged(i, yi, error, displacement, grad_norm,
                                     thresh, disp_thresh, grad_thresh):
            break
        if rho < eta1:
            print('Iteration:%3d  Trial rejected.' % i)

    return np.asarray(xs), np.asarray(ys), gradient


def _bfgs_inverse_action(vector, pairs, initial_inverse, inner_product):
    r"""Apply the retained inverse-BFGS updates to ``vector``.

    For one secant pair :math:`(s,y)`, write :math:`H` for the current
    inverse approximation and :math:`\rho=1/\langle s,y\rangle`. The update is

    .. math::

        H_+ = (I - \rho s \otimes y^\flat) H
              (I - \rho y \otimes s^\flat) + \rho s \otimes s^\flat,

    where :math:`(u\otimes v^\flat)w=u\langle v,w\rangle`. Its action on
    :math:`g` is

    .. math::

        \alpha = \rho\langle s,g\rangle,\quad
        r = H(g - \alpha y),\quad
        \beta = \rho\langle y,r\rangle,\quad
        H_+g = r + (\alpha-\beta)s.

    The backward and forward loops apply these factors for every retained
    pair. ``initial_inverse=None`` uses a scaled identity as the base
    operator, with :math:`\langle s,y\rangle/\langle y,y\rangle` from the
    newest pair.
    """
    residual = vector.copy()
    coefficients = []
    for step, change in reversed(pairs):
        rho = 1. / inner_product(step, change)
        alpha = rho * inner_product(step, residual)
        coefficients.append(alpha)
        residual -= alpha * change

    if initial_inverse is None:
        if pairs: # scale the identity by the last secant pair
            step, change = pairs[-1]
            scale = (inner_product(step, change) /
                     inner_product(change, change))
        else:
            scale = 1.
        result = scale * residual
    else:
        result = initial_inverse @ residual

    for (step, change), alpha in zip(pairs, reversed(coefficients)):
        rho = 1. / inner_product(step, change)
        beta = rho * inner_product(change, result)
        result += (alpha - beta) * step
    return result


def bfgs(func, gradf, retraction, transport, x0, hess_inv,
         ls_method='armijo', nmax=50, thresh=1e-8, grad_thresh=1e-6,
         disp_thresh=1e-6, *args, **kwargs):
    r"""Minimize over real vectors using inverse BFGS.

    Recent secant pairs are kept in the current tangent space and used in a
    two-loop recursion. With ``hess_inv=None``, the base operator is the
    identity, scaled after the first pair by ``<s, y> / <y, y>``. The selected
    inner product is used directly; no metric matrix is formed.

    Args:
        func: Objective returning a real scalar.
        gradf: Gradient returning a real vector with the shape of ``x0``.
        retraction: Update map ``retraction(point, tangent_step)``.
        transport: Callback ``transport(old, new, vector)`` returning a
            tangent vector at ``new``. It is applied to stored secant pairs.
            Isometric transport preserves the secant-pair inner products.
        x0: Initial real one-dimensional point.
        hess_inv: Base inverse-Hessian matrix of shape ``(x0.size,
            x0.size)``, or ``None`` for a scaled identity. It must act on
            tangent vectors and be positive definite for the selected metric.
        ls_method: Line-search method name, fixed step, or callable.
        nmax: Maximum number of iterations.
        thresh: Maximum energy change for the joint stopping test.
        grad_thresh: Maximum gradient norm for the joint stopping test.
        disp_thresh: Maximum point displacement for the joint stopping test.
        *args: Additional positional arguments passed to the line search.
        **kwargs: ``max_history`` sets the number of retained secant pairs;
            it defaults to ``max(1, nmax)``. Other options are line-search
            options, ``inner_product(u, v)`` (metric), ``norm(v)`` (tangent
            norm), and ``distance(old, new)`` (point distance).

    Returns:
        ``(xs, ys, gradient)`` with independent point snapshots, objective
        values, and the final gradient. Histories include the initial point.

    Raises:
        ValueError: If ``hess_inv`` has the wrong shape or ``max_history`` is
            not a positive integer.

    On a manifold, a supplied base matrix must define a suitable tangent
    operator at every iterate.
    """
    print('%s line-search is used in bfgs.' % ls_method)
    ls_method = line_search(ls_method)

    inner_product = kwargs.get('inner_product', _inner_product)
    vector_norm = kwargs.get('norm', np.linalg.norm)
    distance = kwargs.get('distance', _point_distance)
    max_history = kwargs.get('max_history', nmax)

    xi = np.array(x0, copy=True)
    y0 = func(xi)
    g0 = gradf(xi)
    xs, ys = [xi.copy()], [y0]
    size = xi.size
    initial_inverse = None if hess_inv is None else np.asarray(hess_inv)
    if initial_inverse is not None and initial_inverse.shape != (size, size):
        raise ValueError('hess_inv must have shape (x0.size, x0.size).')
    if converged(None, y0, 0, 0, vector_norm(g0),
                 thresh, disp_thresh, grad_thresh):
        return np.asarray(xs), np.asarray(ys), g0

    # list of (step, change) secant pairs in the current tangent space
    pairs = []
    final_gradient = g0

    for i in range(nmax):
        direction = -_bfgs_inverse_action(
            g0, pairs, initial_inverse, inner_product
        )
        if inner_product(g0, direction) >= 0:
            pairs.clear()
            initial_inverse = None
            direction = -g0

        search_kwargs = dict(kwargs, gradient=g0, value=y0)
        step_size = ls_method(
            func, gradf, retraction, xi, direction, *args, **search_kwargs
        )

        x_prev = xi.copy()
        dx = step_size * direction
        xi = retraction(xi, dx)
        yi = func(xi)
        gi = gradf(xi)
        final_gradient = gi

        xs.append(xi.copy())
        ys.append(yi)

        error = abs(yi - y0)
        displacement = distance(x_prev, xi)
        grad_norm = vector_norm(gi)
        if converged(i, yi, error, displacement, grad_norm,
                     thresh, disp_thresh, grad_thresh):
            break

        # transport all stored secant pairs to the new tangent space
        pairs = [(transport(x_prev, xi, s), transport(x_prev, xi, y))
                 for s, y in pairs]

        # add the new secant pair
        s = transport(x_prev, xi, dx)
        y = gi - transport(x_prev, xi, g0)
        pairs.append((s, y))

        # save pairs having sufficient curvature
        # curvature = <s, y>
        # scale = sqrt(<s, s> * <y, y>)
        pairs = [(s, y) for s, y in pairs
                 if inner_product(s, y) >
                 np.finfo(float).eps * np.sqrt(
                     inner_product(s, s) * inner_product(y, y))]
        pairs = pairs[-max_history:]

        y0 = yi
        g0 = gi

    return np.asarray(xs), np.asarray(ys), final_gradient


def _gdiis_coefficients(errors, inner_product):
    """Return residual-minimizing affine weights, dropping singular history."""
    for count in range(len(errors), 1, -1):
        recent = errors[-count:]
        gram = np.array([[inner_product(u, v) for v in recent]
                         for u in recent], dtype=float)
        scale = np.max(np.abs(gram))
        if scale == 0:
            continue
        system = np.zeros((count + 1, count + 1))
        system[:count, :count] = gram / scale
        system[:count, -1] = -1
        system[-1, :count] = -1
        rhs = np.zeros(count + 1)
        rhs[-1] = -1
        try:
            if np.linalg.cond(system) > 1e12:
                continue
            weights = np.linalg.solve(system, rhs)[:count]
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(weights)):
            return count, weights
    return 1, np.array([1.])


def gdiis(func, gradf, retraction, x0, hess_inv=None,
          ls_method='armijo', max_history=6, nmax=50, thresh=1e-8,
          grad_thresh=1e-6, disp_thresh=1e-6, *args, **kwargs):
    r"""Minimize over real Euclidean coordinates using geometry DIIS.

    Stored points and gradients give error vectors ``-hess_inv @ gradient``.
    Affine weights minimize the norm of their weighted error while summing to
    one. The extrapolated point and gradient receive another inverse-Hessian
    correction. A line search from the current point safeguards the update;
    a non-descent or failed extrapolation falls back to a gradient step.
    The inverse Hessian is held fixed, as in the original GDIIS scheme.

    Args:
        func: Objective returning a real scalar.
        gradf: Gradient with the shape of ``x0``.
        retraction: Euclidean update ``retraction(point, step)``. GDIIS also
            forms affine combinations of points, so this callback must act
            as vector addition in the same coordinate space.
        x0: Initial real coordinate array.
        hess_inv: Fixed inverse-Hessian matrix of shape ``(x0.size, x0.size)``;
            ``None`` uses the identity. It should be positive definite.
        ls_method: Line-search method name, fixed step, or callable.
        max_history: Maximum number of stored point/gradient pairs.
        nmax: Maximum number of accepted steps.
        thresh: Maximum energy change for the joint stopping test.
        grad_thresh: Maximum gradient norm for the joint stopping test.
        disp_thresh: Maximum point displacement for the joint stopping test.
        *args: Additional positional arguments passed to the line search.
        **kwargs: Line-search options; ``inner_product(u, v)`` sets the
            residual metric, ``norm(v)`` the gradient norm, and
            ``distance(old, new)`` the point distance for convergence.

    Returns:
        ``(xs, ys, gradient)`` with point snapshots, objective values, and
        the final gradient. Histories include the initial point.

    Raises:
        ValueError: If ``hess_inv`` has the wrong shape or ``max_history``
            is not a positive integer.

    This affine-coordinate method does not implement manifold GDIIS. A
    nonlinear retraction requires a separate way to combine manifold points
    and compare gradients in one tangent space.
    """
    if isinstance(max_history, bool) or not isinstance(
            max_history, (int, np.integer)) or max_history < 1:
        raise ValueError('max_history must be a positive integer.')
    xi = np.array(x0, copy=True)
    size = xi.size
    inverse_hessian = (np.eye(size) if hess_inv is None
                       else np.asarray(hess_inv))
    if inverse_hessian.shape != (size, size):
        raise ValueError('hess_inv must have shape (x0.size, x0.size).')

    print('%s line-search is used in gdiis.' % ls_method)
    search = line_search(ls_method)
    inner_product = kwargs.get('inner_product', _inner_product)
    vector_norm = kwargs.get('norm', np.linalg.norm)
    distance = kwargs.get('distance', _point_distance)

    yi = func(xi)
    gi = gradf(xi)
    xs, ys = [xi.copy()], [yi]
    if converged(None, yi, 0, 0, vector_norm(gi),
                 thresh, disp_thresh, grad_thresh):
        return np.asarray(xs), np.asarray(ys), gi

    points, gradients = [xi.copy()], [gi.copy()]
    for iteration in range(nmax):
        errors = [-(inverse_hessian @ g.ravel()).reshape(xi.shape)
                  for g in gradients]
        count, weights = _gdiis_coefficients(errors, inner_product)
        base = np.tensordot(weights, np.stack(points[-count:]), axes=(0, 0))
        base_gradient = np.tensordot(
            weights, np.stack(gradients[-count:]), axes=(0, 0)
        )
        direction = base - xi - (
            inverse_hessian @ base_gradient.ravel()
        ).reshape(xi.shape)
        preconditioned = -(inverse_hessian @ gi.ravel()).reshape(xi.shape)
        search_kwargs = dict(kwargs, gradient=gi, value=yi)

        for candidate in (direction, preconditioned, -gi):
            if inner_product(gi, candidate) >= 0:
                continue
            try:
                step_size = search(
                    func, gradf, retraction, xi, candidate,
                    *args, **search_kwargs
                )
                break
            except LineSearchError:
                continue
        else:
            raise LineSearchError('GDIIS found no acceptable descent step.')

        x_prev, y_prev = xi.copy(), yi
        xi = retraction(xi, step_size * candidate)
        yi = func(xi)
        gi = gradf(xi)
        xs.append(xi.copy())
        ys.append(yi)

        error = abs(yi - y_prev)
        displacement = distance(x_prev, xi)
        if converged(iteration, yi, error, displacement, vector_norm(gi),
                     thresh, disp_thresh, grad_thresh):
            break

        if candidate is not direction:
            points, gradients = [], []
        points.append(xi.copy())
        gradients.append(gi.copy())
        points = points[-max_history:]
        gradients = gradients[-max_history:]

    return np.asarray(xs), np.asarray(ys), gi
