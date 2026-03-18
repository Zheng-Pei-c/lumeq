from lumeq import np

r"""
keep the walking direction along the gradient vector
while using line search scheme to determine the step size
x_{k+1} = retr_x_k (s_k grad f(x_k))
"""
def gradient_descent(func, gradf, retraction, x0,
                     ls_method='armijo', nmax=50, thresh=1e-8,
                     *args, **kwargs):
    print('%s line search method is used in gradient descent' % ls_method)
    ls_method = line_search(ls_method)

    xi = x0
    y0 = func(x0)

    xs, ys = [x0], [y0]

    ymin = kwargs.get('ymin', None)

    for i in range(nmax):
        step_size = ls_method(func, gradf, retraction, xi, *args, **kwargs)

        # -gradf is the search direction
        dx = -step_size * gradf(xi)
        xi = retraction(xi, dx)
        yi = func(xi)

        xs.append(xi)
        ys.append(yi)

        if ymin:
            error = abs(yi - ymin)
        else:
            error = abs(yi - y0)

        print('i:%3d  value:%12.8f  error: %8.4e' % (i, yi, error), end=' ')
        if error > thresh:
            print('')
        else:
            print('optimization converged!')
            break

        y0 = yi

    return np.array(xs), np.array(ys)


def line_search(method='armijo'):
    r"""
    method options: float, newton, backtracking, armijo, steepest
    """
    if isinstance(method, float): # step size is a given constant
        return ls_steepest(method).kernel

    ls_method = eval('ls_'+method) # get the working function
    if callable(ls_method):
        #print('ls_method:', ls_method, callable(ls_method))
        return ls_method
    else:
        raise NotImplementedError('line search method %s' % method)


def ls_steepest(step=1.):
    # we need a function
    def kernel(*args, **kwargs):
        return step
    return kernel


def ls_backtracking(func, gradf, retraction, x0,
                    tau=.5, r=1e-4, alpha=1.):
    r"""
    alpha is the step size we need to determine
    reduce it from a large value
    """
    y0 = func(x0)
    df = gradf(x0) # grad is also the negative descent direction
    #dot = np.dot(df, df) # -<grad, descent_direction>
    norm = np.linalg.norm(df)
    dot = norm**2

    alpha /= norm

    def kernel(alpha): # recursion
        dx = -alpha * df
        xi = retraction(x0, dx)
        dy = y0 - func(xi)

        if dy >= r * alpha * dot:
            return alpha

        alpha *= tau # reducing the step size
        return kernel(alpha)

    alpha = kernel(alpha)
    return alpha


ls_armijo = ls_backtracking


def ls_newton_raphson(func, gradf, retraction, x0, **kwargs):
    hessf = kwargs.get('hessf')
    inv = np.linalg.inv(hessf)
    return


"""
conjugate gradient algorithm would change the search direction as well
"""
def conjugate_gradient(func, gradf, retraction, transport, preconditioner, x0,
                       ls_method='armijo', cg_method='fletcher_reeves',
                       nmax=50, thresh=1e-8,
                       *args, **kwargs):
    print('%s line search and %s update are used in conjugate gradient' % (ls_method, cg_method))
    ls_method = line_search(ls_method)
    cg_method = choose_direction(cg_method)

    def _preconditioner(v): # do nothing
        return v
    if preconditioner is None:
        preconditioner = _preconditioner

    xi = x0
    y0 = func(x0)
    g0 = gradf(x0)

    xs, ys = [x0], [y0]

    direction = -g0 # initial search direction

    for i in range(nmax):
        step_size = ls_method(func, gradf, retraction, xi, *args, **kwargs)

        dx = step_size * direction
        xi = retraction(xi, dx)
        yi = func(xi)

        xs.append(xi)
        ys.append(yi)

        error = abs(yi - y0)

        print('i:%3d  value:%12.8f  error: %8.4e' % (i, yi, error), end=' ')
        if error > thresh:
            print('')
        else:
            print('optimization converged!')
            break

        gi = gradf(xi)
        beta = cg_method(transport, preconditioner, dx, g0, gi,
                         *args, **kwargs)
        # < T_dx (direction), G> = 0
        direction = beta * transport(None, xi, direction) - gi

        y0 = yi
        g0 = gi

    return np.array(xs), np.array(ys)


def choose_direction(method='fletcher_reeves'):
    cg_method = eval('cg_'+method)

    if callable(cg_method):
        return cg_method
    else:
        raise NotImplementedError('conjugate gradient %s beta scheme' % method)


def cg_fletcher_reeves(transport, preconditioner, x1, g0, g1):
    pg0, pg1 = preconditioner(g0), preconditioner(g1)

    return np.dot(g1, pg1) / np.dot(g0, pg0)


def cg_polak_ribiere(transport, preconditioner, x1, g0, g1):
    dg = g1 - transport(None, x1, g0)
    g1 = preconditioner(g1)

    return abs(np.dot(g1, dg)) / np.dot(g0, g0)


def newton_2nd(func, gradf, hessf, tangent_solver, geodesic, x0, dt=1.,
               nmax=50, thresh=1e-8, *args, **kwargs):
    xi = x0
    y0 = func(x0)

    xs, ys = [x0], [y0]

    for i in range(nmax):
        v = tangent_solver(xi, gradf, hessf)

        xi = geodesic(xi, v, dt)
        yi = func(xi)

        xs.append(xi)
        ys.append(yi)

        error = abs(yi - y0)

        print('i:%3d  value:%12.8f  error: %8.4e' % (i, yi, error), end=' ')
        if error > thresh:
            print('')
        else:
            print('optimization converged!')
            break

        y0 = yi

    return np.array(xs), np.array(ys)


#TODO: how to do trust_region?
def trust_region(func, gradf, hessf, tangent_solver, geodesic, x0, dt=1.,
                 nmax=50, thresh=1e-8, *args, **kwargs):
    xi = x0
    y0 = func(x0)

    xs, ys = [x0], [y0]

    for i in range(nmax):
        v = tangent_solver(xi, gradf, hessf)

        xi = geodesic(xi, v, dt)
        yi = func(xi)

        xs.append(xi)
        ys.append(yi)

        error = abs(yi - y0)

        print('i:%3d  value:%12.8f  error: %8.4e' % (i, yi, error), end=' ')
        if error > thresh:
            print('')
        else:
            print('optimization converged!')
            break

        y0 = yi

    return np.array(xs), np.array(ys)
