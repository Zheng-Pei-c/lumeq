from lumeq import np

class Sampler:
    r"""
    Random numbers sampler.
    """
    def __init__(self, size, variance, sigma=None, mean=0, seed=None, **kwargs):
        r"""
        Parameters
            size : int
                Size of the random numbers array.
            variance : float
                Variance of the random numbers.
            sigma : float, optional
                Standard deviation of the random numbers. If provided, it overrides variance.
            mean : float, optional
                Mean of the random numbers. Default is 0.
            seed : int, optional
                Seed for the random number generator. Default is None.
            covariance : bool, optional
                Whether to generate correlated random numbers. Default is False.
        """
        if sigma is None: sigma = np.sqrt(variance)

        self.size = size
        self.sigma = sigma
        self.mean = mean
        self.rng = np.random.default_rng(seed)

        # correlations between random numbers
        if kwargs.get('covariance', False):
            self.get_covariance(**kwargs)
        else:
            self.covariance = None


    def get_covariance(self, coordinate, ell_c, kind='exp', nugget=1e-12):
        r"""
        Generate covariance matrix for the (to be generated) random numbers.
        Note it includes variance on the diagonal.

        Parameters
            coordinate : (N, 3) 2D Array
                Coordinate array.
            ell_c : float
                Correlation length.
            kind : str, optional
                Type of covariance function. Default is 'exp'.
            nugget : float, optional
                Small diagonal added for numerical stability. Default is 0.

        Returns
            covariance : 2D Array
                lower/left-side of the covariance matrix.
        """
        assert coordinate.shape[0] == self.size # coordinate size must match

        diff = coordinate[:, np.newaxis, :] - coordinate[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)

        variance = self.sigma ** 2
        if kind == 'exp':
            cov = (variance) * np.exp(-dist / ell_c)
        elif kind == 'gauss':
            cov = (variance) * np.exp(-(dist ** 2) / (2. * ell_c ** 2))
        else:
            raise ValueError(f"Unknown covariance kind: {kind}")

        if nugget > 0.:
            cov += nugget * np.eye(self.size)

        try:
            L = np.linalg.cholesky(self.covariance)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(self.covariance)
            w = np.clip(w, a_min=0., a_max=None) # remove negative eigenvalues
            L = v @ np.diag(np.sqrt(w))

        self.covariance = L
        return self.covariance


    def sample(self, *args, **kwargs):
        r"""Generate random numbers from a normal distribution."""
        if self.covariance is None: # no correlations
            return self.rng.normal(loc=self.mean, scale=self.sigma,
                                   size=self.size)
        else:
            # size is determined by covariance matrix
            #mean = self.mean * np.ones(self.size) # length matches covariance
            #return self.rng.multivariate_normal(mean=mean,
            #                                    cov=self.covariance)
            return self.covariance @ self.rng.normal(size=self.size) + self.mean


    def correlated_sample(self, values, tau_c, dt=1., **kwargs):
        r"""
        Generate correlated random numbers from Ornstein-Uhlenbeck process.

        Math : `<delta(t) delta(0)> = sigma^2 exp(-|t|/tau_c)`
            delta^{n+1} = a delta^{n} + b N(0, sigma)
            a = exp(-dt/tau_c)
            b = sigma sqrt(1 - a^2)

        Parameters
            values : Array of previous sampled values
            tau_c : float
                Correlation time.
            dt : float, optional
                Time step. Default is 1.

        Returns
            values : Array of correlated random numbers
        """
        a = np.exp(-dt / tau_c)
        b = np.sqrt(1. - a**2)
        z = self.rng.normal(size=values.shape)

        values *= a

        if self.covariance is None:
            values += (self.sigma * b) * z
        else: # sigma is encoded in covariance
            values += b * (self.covariance @ z)
        return values
