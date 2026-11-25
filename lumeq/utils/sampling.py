from lumeq import np


class Sampler:
    r"""Sample Gaussian disorder, optionally correlated in space and time."""

    def __init__(self, size, variance, sigma=None, mean=0, seed=None,
                 covariance=False, coordinate=None, ell_c=None, kind='exp',
                 nugget=1e-12, **kwargs):
        r"""Initialize the distribution and random-number generator.

        Args:
            size (int or tuple): Default output shape when no values are supplied.
            variance (float): Stationary variance; overridden by sigma.
            sigma (float or None): Stationary standard deviation.
            mean (float): Distribution mean.
            seed (int or numpy.random.SeedSequence or None): Random seed.
            covariance (bool): Enable spatial correlations between flattened entries.
            coordinate (numpy.ndarray or None): Coordinates, shape ``(N, 3)``,
                where N is the product of size; required for spatial correlations.
            ell_c (float or None): Positive spatial correlation length.
            kind (str): Spatial kernel, ``'exp'`` or ``'gauss'``.
            nugget (float): Nonnegative regularization added to the covariance diagonal.
            **kwargs: Additional options ignored for compatibility.
        """
        if sigma is None:
            sigma = np.sqrt(variance)
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError('sigma must be finite and nonnegative.')

        self.size = size
        self.sigma = sigma
        self.mean = mean
        self.rng = np.random.default_rng(seed)
        self.covariance = None
        if covariance:
            self.get_covariance(coordinate, ell_c, kind, nugget)

    def get_covariance(self, coordinate, ell_c, kind='exp', nugget=1e-12):
        r"""Factor the spatial covariance matrix.

        .. math::

            C_{ij}=\sigma^2 K(r_{ij}/\ell_c)+\eta\delta_{ij},\qquad
            LL^T=C,

        where K(x)=exp(-x) or exp(-x**2/2).

        Args:
            coordinate (numpy.ndarray): Coordinates of flattened entries,
                shape ``(N, 3)``.
            ell_c (float): Positive spatial correlation length.
            kind (str): ``'exp'`` or ``'gauss'``.
            nugget (float): Nonnegative diagonal regularization, eta.

        Returns:
            numpy.ndarray: Stored covariance factor L, shape ``(N, N)``.
        """
        coordinate = np.asarray(coordinate, dtype=float)
        n = int(np.prod(self.size))
        if coordinate.shape != (n, 3):
            raise ValueError('coordinate must have shape (product(size), 3).')
        if ell_c is None or not np.isfinite(ell_c) or ell_c <= 0:
            raise ValueError('ell_c must be finite and positive.')
        if not np.isfinite(nugget) or nugget < 0:
            raise ValueError('nugget must be finite and nonnegative.')

        dist = np.linalg.norm(coordinate[:, None] - coordinate[None, :], axis=-1)
        if kind == 'exp':
            cov = self.sigma**2 * np.exp(-dist / ell_c)
        elif kind == 'gauss':
            cov = self.sigma**2 * np.exp(-0.5 * (dist / ell_c)**2)
        else:
            raise ValueError(f'Unknown covariance kind: {kind}')
        cov += nugget * np.eye(n)

        try:
            factor = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(cov)
            factor = v * np.sqrt(np.maximum(w, 0.))

        self.covariance = factor
        return factor

    def sample(self, values=None, *args, **kwargs):
        r"""Draw an independent Gaussian sample with the requested shape.

        Args:
            values (numpy.ndarray or None): Shape template; None uses size.
                Previous values do not affect the draw.
            *args: Ignored temporal-sampling arguments, including tau_c and dt.
            **kwargs: Ignored additional sampling options.

        Returns:
            numpy.ndarray: Fresh sample matching values.shape or size. Spatial
                covariance, when enabled, acts on flattened entries.
        """
        shape = self.size if values is None else np.shape(values)
        if self.covariance is None:
            return self.rng.normal(loc=self.mean, scale=self.sigma, size=shape)

        n = int(np.prod(shape))
        if self.covariance.shape != (n, n):
            raise ValueError('Sample shape does not match the spatial covariance.')
        sample = self.covariance @ self.rng.normal(size=n) + self.mean
        return sample.reshape(shape)

    def correlated_sample(self, values, tau_c, dt=1., **kwargs):
        r"""Advance temporally correlated Gaussian disorder by the elapsed time.

        .. math::

            x_{n+1}=\mu+a(x_n-\mu)+\sqrt{1-a^2}\,L\xi_n,
            \qquad a=e^{-\Delta t/\tau_c},\quad \xi_n\sim\mathcal{N}(0,I).

        Without spatial covariance, L=sigma*I. The update preserves the target
        mean and covariance when initialized in the stationary distribution;
        a zero initial deviation instead gives a transient from zero disorder.

        References:
            Hao Li, Ajay Ram Srimath Kandada, Carlos Silva, and Eric R. Bittner,
            "Stochastic scattering theory for excitation-induced dephasing:
            Comparison to the Anderson-Kubo lineshape,"
            J. Chem. Phys. 153, 154115 (2020).
            https://doi.org/10.1063/5.0026467

            Daniel T. Gillespie,
            "Exact numerical simulation of the Ornstein-Uhlenbeck process
            and its integral,"
            Phys. Rev. E 54, 2084-2091 (1996).
            https://doi.org/10.1103/PhysRevE.54.2084

        Args:
            values (numpy.ndarray): Previous disorder. Floating-point arrays
                are updated in place; output preserves their shape.
            tau_c (float): Positive temporal correlation time.
            dt (float): Nonnegative elapsed time, in the same units as tau_c.
            **kwargs: Ignored additional sampling options.

        Returns:
            numpy.ndarray: Updated disorder with the same shape as values.
        """
        if not np.isfinite(tau_c) or tau_c <= 0:
            raise ValueError('tau_c must be finite and positive.')
        if not np.isfinite(dt) or dt < 0:
            raise ValueError('dt must be finite and nonnegative.')

        values = np.asarray(values, dtype=float)
        if dt == 0:
            return values
        a = np.exp(-dt / tau_c)
        b = np.sqrt(-np.expm1(-2. * dt / tau_c))
        noise = self.sample(values) - self.mean
        values *= a
        values += (1. - a) * self.mean + b * noise
        return values
