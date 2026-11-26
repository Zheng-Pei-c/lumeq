from lumeq import np
from lumeq.utils import print_matrix, convert_units
from lumeq.utils import put_keys_kwargs_to_object
from lumeq.dynamics import harmonic_oscillator

def get_trans_amplitude(ntot, coupling=1., energy=None, vector=False):
    trans = np.sqrt(np.arange(1, ntot))*coupling # does not include the end value

    if vector:
        return trans

    trans = np.diagflat(trans, 1)
    trans += trans.T

    if energy:
        np.fill_diagonal(trans, np.arange(0, ntot)*energy)

    return trans


class PhotonStep():
    """Propagate a photon density matrix in the Fock basis.

    ``fock``/``number`` use number states and ``thermo`` uses a diagonal Gibbs
    state. ``minimum`` displaces a number state using a supplied molecular
    dipole. ``thermo_minimum`` also samples imaginary displacement amplitudes
    with variance kBT/(2 omega), matching the nuclear velocity-only ensemble.
    It is an ensemble for mean-field IR trajectories, not a quantum Gibbs state.
    Each mode is one scalar oscillator; c_lambda carries its polarization.
    density has shape (nmode, nbasis, nbasis), with a common basis size.
    Quadrature expectations are evaluated only for diagnostics and printing.
    """
    def __init__(self, key, **kwargs):
        key.setdefault('frequency', 0.05)
        key.setdefault('freq_unit', 'hartree')
        key.setdefault('c_lambda', np.array([0.,0.,0.1]))
        key.setdefault('init_method', 'fock')
        key.setdefault('init_number', 0)
        key.setdefault('init_temp', 300.)
        key.setdefault('basis_size', 10)

        put_keys_kwargs_to_object(self, key, **kwargs)

        self.frequency = np.atleast_1d(convert_units(
            np.asarray(self.frequency), self.freq_unit, 'hartree'))
        self.freq_unit = 'hartree'
        self.nmode = len(self.frequency)
        self.c_lambda = np.broadcast_to(self.c_lambda, (self.nmode, 3))
        self.init_number = np.broadcast_to(self.init_number, (self.nmode,))
        self.basis_size = np.broadcast_to(self.basis_size, (self.nmode,)).astype(int)

        self.scaled_freq = np.sqrt(self.frequency/2.)
        self._trans = [get_trans_amplitude(n, vector=True) for n in self.basis_size]
        self.init_density()


    def init_density(self, molecular_dipole=None):
        """Build rho, then optionally displace its real coherence once.

        Construction samples phases only for ``thermo_minimum``. A subsequent
        call with the molecular dipole changes Re<a> without changing Im<a> or
        resampling. All operations act on rho in the Fock basis.
        """
        from scipy.linalg import expm
        if molecular_dipole is None:
            if 'thermo' in self.init_method:
                from .oscillator_dynamics import get_boltzmann_beta
                beta_b = get_boltzmann_beta(self.init_temp)

            self.density = np.array([
                self.get_thermal_density(n, self.frequency[i], beta_b)
                if self.init_method == 'thermo' else self.get_initial_density(n, self.init_number[i])
                for i, n in enumerate(self.basis_size)], dtype=complex)
            alpha = np.zeros(self.nmode, dtype=complex)
            if self.init_method == 'thermo_minimum':
                rng = np.random.default_rng(getattr(self, 'random_seed', None))
                alpha = (1j * rng.normal(size=alpha.shape) /
                         np.sqrt(2. * beta_b * self.frequency))
        elif 'minimum' in self.init_method:
            alpha = -(self.c_lambda @ molecular_dipole) / (2. * self.scaled_freq)
        else:
            return

        for i, ntot in enumerate(self.basis_size):
            annihilation = np.diag(self._trans[i], 1)
            amplitude = np.dot(self._trans[i], np.diag(self.density[i], -1))
            # A real displacement preserves the previously sampled phase.
            shift = alpha[i] - amplitude.real if molecular_dipole is not None else alpha[i]
            if shift == 0.: continue
            displacement = expm(shift * annihilation.T - shift.conjugate() * annihilation)
            self.density[i] = displacement @ self.density[i] @ displacement.conj().T
            expected = amplitude + shift
            actual = np.dot(self._trans[i], np.diag(self.density[i], -1))
            boundary = ntot * self.density[i,-1,-1].real
            if boundary > 1e-8 or not np.isclose(actual, expected, rtol=1e-7, atol=1e-11):
                raise RuntimeError('initial photon density is not converged; increase basis_size')

        if molecular_dipole is None: molecular_dipole = np.zeros(3)
        return self.update_density(molecular_dipole, 0)


    def get_initial_density(self, ntot, n):
        n = int(n)
        if not 0 <= n < ntot:
            raise ValueError('init_number must be within the photon basis')
        init_density = np.zeros((ntot, ntot))
        init_density[n,n] = 1.
        return init_density


    def get_thermal_density(self, ntot, frequency, beta_b):
        # The zero-point energy cancels from the Boltzmann probabilities.
        population = np.exp(-beta_b * frequency * np.arange(ntot))
        population /= population.sum()
        return np.diag(population)


    def get_photon_number_probabilities(self, max_n=None):
        """Return Fock probabilities with shape (nmode, nlevels).

        The optional max_n selects n = 0, ..., max_n. Values beyond the
        finite Fock basis are zero; a shorter selection omits its tail.
        """
        probabilities = np.diagonal(self.density, axis1=-2, axis2=-1).real.copy()
        if max_n is None:
            return probabilities
        if not isinstance(max_n, (int, np.integer)) or max_n < 0:
            raise ValueError('max_n must be a nonnegative integer')

        result = np.zeros((*probabilities.shape[:-1], max_n + 1))
        ncopy = min(max_n + 1, probabilities.shape[-1])
        result[..., :ncopy] = probabilities[..., :ncopy]
        return result


    def get_photon_number(self):
        """Return mean Fock occupation, shaped (nmode,)."""
        probabilities = self.get_photon_number_probabilities()
        return probabilities @ np.arange(probabilities.shape[-1])


    def get_coordinate_velocity(self):
        """Return scalar quadrature expectations, each shaped (nmode,).

        q = (a + a†)/sqrt(2 omega) and v = -i sqrt(omega/2) (a - a†).
        These are diagnostics derived from rho, not propagated variables.
        """
        amplitude = np.array([
            np.dot(self._trans[i], np.diag(self.density[i], -1))
            for i in range(self.nmode)
        ])
        return amplitude.real / self.scaled_freq, 2. * self.scaled_freq * amplitude.imag


    def update_density(self, molecular_dipole, dt, half=1, **kwargs):
        """Advance rho with half=1 followed by half=2; retain the latest coupling."""
        from scipy.linalg import expm
        coupling = self.scaled_freq * (self.c_lambda @ molecular_dipole)

        if half == 1:
            # Predict the photon state at the new time with the old dipole.
            for i, ntot in enumerate(self.basis_size):
                if dt == 0.: continue
                trans = get_trans_amplitude(ntot, coupling[i], self.frequency[i])
                propagator = expm(-1j*trans*dt)
                self.density[i] = propagator @ self.density[i] @ propagator.conj().T
        elif half == 2:
            # Correct using the new dipole, without another electronic step.
            delta = coupling - self.coupling
            for i, ntot in enumerate(self.basis_size):
                if dt == 0. or delta[i] == 0.: continue
                trans = get_trans_amplitude(ntot, .5*delta[i])
                propagator = expm(-1j*trans*dt)
                self.density[i] = propagator @ self.density[i] @ propagator.conj().T
        else:
            raise ValueError('half must be 1 or 2')

        self.coupling = coupling
        self.energy = np.dot(self.frequency, self.get_photon_number())
        return {'trans_coeff': self.get_trans_coeff()}


    def get_trans_coeff(self):
        """Return the Cartesian electronic coupling directly from rho."""
        trans_coeff = np.array([
            np.dot(self._trans[i], np.diag(self.density[i], 1) +
                   np.diag(self.density[i], -1)).real
            for i in range(self.nmode)
        ])
        return (self.scaled_freq * trans_coeff) @ self.c_lambda


    def print_variables(self):
        coordinate, velocity = self.get_coordinate_velocity()
        print_matrix('photon coordinate expectation (au):', coordinate)
        print_matrix('photon velocity expectation (au):', velocity)
        print_matrix('photon number expectation:', self.get_photon_number())



class PhotonStep2(harmonic_oscillator):
    def __init__(self, key, **kwargs):
        # The cavity coordinates in the MD model begin at zero.
        key.setdefault('freq_unit', 'hartree')
        key.setdefault('init_method', 'none')
        key.setdefault('update_method', 'velocity_verlet')
        super().__init__(key, **kwargs)

    def convert_parameter_units(self, unit_dict):
        self.n_site = 1 # one scalar oscillator per mode
        self.frequency = np.atleast_1d(convert_units(
            np.asarray(self.frequency), self.freq_unit, 'hartree'))
        self.c_lambda = np.atleast_2d(self.c_lambda)
        self.mass = np.ones(len(self.frequency))


    def init_coordinate_velocity(self, init_method=None, molecular_dipole=None):
        """Initialize once; set a one-pass minimum when a dipole is supplied.

        Construction sets zero coordinates and samples velocities for methods
        containing 'thermo'. A later call with molecular_dipole updates only
        the coordinates for methods containing 'minimum', preserving velocities.
        """
        if init_method is None: init_method = self.init_method

        if molecular_dipole is None:
            self.omega2 = self.frequency**2
            self.coordinate = np.zeros(len(self.frequency))
            self.velocity = np.zeros_like(self.coordinate)
            if 'thermo' in init_method:
                self.velocity = self.init_velocity_thermo(
                    seed=getattr(self, 'random_seed', None))

        elif 'minimum' in init_method:
            self.coordinate = self.get_minimum_displacement(molecular_dipole)

        self.get_energy(self.velocity)


    def init_density(self, molecular_dipole=None):
        """Initialize the classical photon state through the common MD interface."""
        self.init_coordinate_velocity(molecular_dipole=molecular_dipole)
        return {'trans_coeff': self.get_trans_coeff()}


    def get_minimum_displacement(self, molecular_dipole):
        return -(self.c_lambda @ molecular_dipole) / self.frequency


    def init_velocity_thermo(self, temp=None, seed=None):
        """Sample Maxwell velocities using photon masses and temperature."""
        from .oscillator_dynamics import get_boltzmann_beta
        if temp is None: temp = self.init_temp

        beta_b = get_boltzmann_beta(temp)
        sigma = np.sqrt(1./beta_b/self.mass)
        rng = np.random.default_rng(seed)
        return rng.normal(size=len(self.mass)) * sigma


    def print_variables(self):
        print_matrix('photon coordinate (au):', self.coordinate)
        print_matrix('photon velocity (au):', self.velocity)
        print_matrix('photon mean number (coherent state):',
                     self.get_photon_number())


    def get_photon_number(self):
        """Return coherent-state mean occupation for each classical q, v."""
        if np.any(self.frequency <= 0.):
            raise ValueError('photon frequency must be positive')
        omega = self.frequency
        mass = self.mass
        return mass * (self.velocity**2 + omega**2 * self.coordinate**2) / (2. * omega)


    def get_photon_number_probabilities(self, max_n=9):
        """Return coherent-state Poisson probabilities for each classical q, v.

        These are conditional on this oscillator trajectory. A thermal photon
        distribution requires averaging probabilities over an ensemble. The
        returned n = 0, ..., max_n range omits its Poisson tail.
        """
        from scipy.stats import poisson
        if not isinstance(max_n, (int, np.integer)) or max_n < 0:
            raise ValueError('max_n must be a nonnegative integer')
        mean_number = self.get_photon_number()
        return poisson.pmf(np.arange(max_n + 1), mean_number[..., None])


    def update_density(self, molecular_dipole, dt, half=1, **kwargs):
        force = -self.frequency * (self.c_lambda @ molecular_dipole)
        self.update_coordinate_velocity(force, half, **kwargs)
        return {'trans_coeff': self.get_trans_coeff()}


    def get_trans_coeff(self):
        """Return the Cartesian electronic coupling without advancing time."""
        return (self.frequency * self.coordinate) @ self.c_lambda


if __name__ == '__main__':
    dt = 25 #au
    nsteps = 10
    c_lambda = np.zeros(3)
    key = {}
    key['c_lambda'] = c_lambda

    photon = PhotonStep(key)

    energy = []
    for i in range(nsteps):
        kwargs = photon.update_density(np.zeros(3), dt)
        energy.append(photon.energy)
    energy = np.array(energy)

    print_matrix('energy:', energy, 5)
