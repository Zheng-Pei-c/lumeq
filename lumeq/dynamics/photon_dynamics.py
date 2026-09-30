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
    def __init__(self, key, **kwargs):
        key.setdefault('frequency', 0.05)
        key.setdefault('freq_unit', 'hartree')
        key.setdefault('c_lambda', np.array([0.,0.,0.1]))
        key.setdefault('init_method', 'fock')
        key.setdefault('init_number', [0,0,0])
        key.setdefault('init_temp', 300.)
        key.setdefault('basis_size', 10)

        put_keys_kwargs_to_object(self, key, **kwargs)

        if self.freq_unit not in ('hartree', 'eh'):
            self.frequency = convert_units(self.frequency, self.freq_unit, 'hartree')
            self.freq_unit = 'hartree'

        if isinstance(self.frequency, float):
            self.frequency = np.array([self.frequency])
            self.c_lambda = np.array([self.c_lambda])
            self.init_number = np.array([self.init_number])
            self.basis_size = np.array([self.basis_size])

        self.scaled_freq = np.sqrt(self.frequency/2.)

        self.nmode = len(self.frequency)

        self._trans, self.density = [None]*self.nmode, [None]*self.nmode
        if self.init_method == 'thermo':
            if self.init_temp <= 0. or np.any(self.frequency <= 0.):
                raise ValueError('thermal photons require positive temperature and frequency')
            from .oscillator_dynamics import get_boltzmann_beta
            beta_b = get_boltzmann_beta(self.init_temp)
        elif self.init_method not in ('fock', 'number'):
            raise ValueError("init_method must be 'fock', 'number', or 'thermo'")

        for i in range(self.nmode):
            ntot = self.basis_size[i]
            self._trans[i] = get_trans_amplitude(ntot, vector=True)
            if self.init_method == 'thermo':
                self.density[i] = self.get_thermal_density(ntot, self.frequency[i], beta_b)
            else:
                self.density[i] = self.get_initial_density(ntot, self.init_number[i])

        self.density = np.array(self.density, dtype=complex)
        self.update_density(np.zeros(3), 0) # get initial energy without advancing time
        self._previous_coupling = None


    def get_initial_density(self, ntot, ns):
        if np.isscalar(ns): ns = [ns]

        init_density = np.zeros((len(ns),ntot,ntot))
        #init_density[n,n] = n
        for x in range(len(ns)):
            n = int(ns[x])
            if not 0 <= n < ntot:
                raise ValueError('init_number must be within the photon basis')
            init_density[x,n,n] = 1.
        #init_density = np.ones((ntot,ntot))/ntot
        return init_density


    def get_thermal_density(self, ntot, frequency, beta_b):
        # The zero-point energy cancels from the Boltzmann probabilities.
        population = np.exp(-beta_b * frequency * np.arange(ntot))
        population /= population.sum()
        return np.repeat(np.diag(population)[None, :, :], 3, axis=0)


    def get_photon_number_probabilities(self, max_n=None):
        """Return Fock probabilities with shape (nmode, 3, nlevels).

        The optional max_n selects n = 0, ..., max_n. Values beyond the
        finite Fock basis are zero; a shorter selection omits its tail.
        """
        probabilities = np.diagonal(self.density, axis1=-2, axis2=-1).real.copy()
        if max_n is None:
            return probabilities
        if not isinstance(max_n, (int, np.integer)) or max_n < 0:
            raise ValueError('max_n must be a nonnegative integer')
        result = np.zeros((*probabilities.shape[:2], max_n + 1))
        ncopy = min(max_n + 1, probabilities.shape[-1])
        result[..., :ncopy] = probabilities[..., :ncopy]
        return result


    def update_density(self, molecular_dipole, dt, half=1, **kwargs):
        from scipy.linalg import expm
        coupling = np.einsum('i,ix,x->ix', self.scaled_freq, self.c_lambda, molecular_dipole)
        if half == 1:
            # Predict the photon state at the new time with the old dipole.
            self._previous_coupling = coupling
            for i in range(self.nmode):
                for x in range(3):
                    ntot = self.basis_size[i]
                    trans = get_trans_amplitude(ntot, coupling[i,x], self.frequency[i])
                    propagator = expm(-1j*trans*dt)
                    self.density[i,x] = propagator @ self.density[i,x] @ propagator.conj().T
        elif half == 2:
            if self._previous_coupling is None:
                raise ValueError('photon half=2 requires a preceding half=1 update')
            # The new dipole is available after the electronic step. Correct
            # the momentum using the change in coupling; this uses no extra
            # electronic force or dipole evaluation.
            delta = coupling - self._previous_coupling
            for i in range(self.nmode):
                for x in range(3):
                    if delta[i,x] == 0.:
                        continue
                    ntot = self.basis_size[i]
                    trans = get_trans_amplitude(ntot, .5*delta[i,x])
                    propagator = expm(-1j*trans*dt)
                    self.density[i,x] = propagator @ self.density[i,x] @ propagator.conj().T
            self._previous_coupling = None
        else:
            raise ValueError('half must be 1 or 2')

        energy = 0.
        trans_coeff = np.zeros((self.nmode, 3))
        for i in range(self.nmode):
            ntot = self.basis_size[i]
            for x in range(3): # spatial directions
                trans_coeff[i,x] = np.dot(self._trans[i], np.diag(self.density[i,x], 1)+np.diag(self.density[i,x], -1)).real
                energy += self.frequency[i] * np.dot(range(ntot), np.diag(self.density[i,x])).real

        self.energy = energy

        kwargs = {}
        kwargs['trans_coeff'] = np.einsum('i,ix,ix->x', self.scaled_freq, trans_coeff, self.c_lambda)

        return kwargs


class PhotonStep2(harmonic_oscillator):
    def __init__(self, key, **kwargs):
        # The cavity coordinates in the MD model begin at zero.
        key.setdefault('freq_unit', 'hartree')
        key.setdefault('init_method', 'none')
        key.setdefault('update_method', 'velocity_verlet')
        super().__init__(key, **kwargs)

    def convert_parameter_units(self, unit_dict):
        self.n_site = 3 #xyz
        self.frequency = convert_units(self.frequency, self.freq_unit, 'hartree')

        if isinstance(self.frequency, float):
            self.frequency = np.array([self.frequency])
            self.c_lambda = np.array([self.c_lambda])

        self.mass = np.ones(len(self.frequency))
        if self.init_method == 'thermo':
            from .oscillator_dynamics import get_boltzmann_beta
            self.beta_b = get_boltzmann_beta(self.init_temp)


    def get_minimim_displacement(self, molecular_dipole):
        self.coordinate = -np.einsum('i,ix,x->ix', 1./self.frequency, self.c_lambda, molecular_dipole)
        self.get_energy(self.velocity)

        return self.coordinate


    def get_photon_number_probabilities(self, max_n=9):
        """Return coherent-state Poisson probabilities for each classical q, v.

        These are conditional on this oscillator trajectory. A thermal photon
        distribution requires averaging probabilities over an ensemble. The
        returned n = 0, ..., max_n range omits its Poisson tail.
        """
        from scipy.stats import poisson
        if not isinstance(max_n, (int, np.integer)) or max_n < 0:
            raise ValueError('max_n must be a nonnegative integer')
        if np.any(self.frequency <= 0.):
            raise ValueError('photon frequency must be positive')
        omega = self.frequency[:, None]
        mass = self.mass[:, None]
        mean_number = mass * (self.velocity**2 + omega**2 * self.coordinate**2) / (2. * omega)
        return poisson.pmf(np.arange(max_n + 1), mean_number[..., None])


    def update_density(self, molecular_dipole, dt, half=1, **kwargs):
        force = -np.einsum('i,ix,x->ix', self.frequency, self.c_lambda, molecular_dipole)
        if self.n_site == 1:
            force = np.sum(force, axis=1).reshape(-1, 1)

        self.update_coordinate_velocity(force, half, **kwargs)

        kwargs = {}
        kwargs['trans_coeff'] = np.einsum('i,ix,ix->x', self.frequency, self.coordinate, self.c_lambda)

        return kwargs


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
