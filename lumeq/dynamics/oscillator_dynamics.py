from lumeq import np
from lumeq.utils import print_matrix, convert_units
from lumeq.utils import put_keys_kwargs_to_object
from lumeq.utils.sec_mole import get_molecular_center, get_moment_of_inertia

ELECTRON_MASS_IN_AMU = 5.4857990945e-04 # from qchem

def get_boltzmann_beta(temperature):
    # temperature in Kelvin
    from pyscf.data.nist import HARTREE2J, BOLTZMANN
    return HARTREE2J / (BOLTZMANN * temperature)


class harmonic_oscillator():
    r"""Harmonic oscillator class."""
    def __init__(self, key={}, **kwargs):
        r"""
        needed parameters:
        commom:
            mass, coordinate, velocity,
            update_method, init_method, init_temp,
        atomic system has size (natoms, 3)
        photon system has size (nmode, 3)
            frequency,
        phonon system has size (nmode, n_site_tot)
            n_site, frequency,
        # frequency and mass has same dimension (nmode)
        """
        # set default
        key.setdefault('dt', 1) # au
        key.setdefault('frequency', None)
        key.setdefault('init_temp', 298) # K
        key.setdefault('init_method', 'thermo')
        key.setdefault('update_method', 'euler')
        key.setdefault('debug', 0)

        put_keys_kwargs_to_object(self, key, **kwargs)

        self.convert_parameter_units(getattr(self, 'unit_dict', None))

        self.init_coordinate_velocity()


    #get_boltzmann_beta = get_boltzmann_beta


    def init_coordinate_velocity(self, n_site=None, init_method=None):
        r"""
        Generate the initial oscillator coordinate and velocity.

        Args:
            n_site (int, optional): Total number of molecular sites.
            init_method (str, optional): Initialization method. ``'thermo'`` uses the
                Boltzmann distribution at ``init_temp``.
        """
        if n_site is None: n_site = self.n_site
        if init_method is None: init_method = self.init_method

        nmode = len(self.frequency)

        self.omega2 = self.frequency**2

        self.coordinate = np.zeros((nmode, n_site))
        self.velocity  = np.zeros((nmode, n_site))

        def get_gaussian_distribution(variance, size, mean=0, seed=None):
            rng = np.random.default_rng(seed)
            return rng.normal(loc=mean, scale=np.sqrt(variance), size=size)

        # Boltzmann thermol distribution follows gaussian function
        if init_method == 'thermo':
            # equal parition of the kT energy to kinetic and potential
            # 0.5 is because in the Gaussian function, q^2 = 2 * sigma^2
            seed = getattr(self, 'random_seed', None)
            beta_b = self.beta_b
            K = np.einsum('i,i->i', self.mass, self.omega2)
            variance = .5 / (beta_b * K)
            if self.debug > 0:
                print_matrix('force constant:', K)
                print_matrix('coordinate variance:', variance)

            for i in range(nmode):
                self.coordinate[i] = get_gaussian_distribution(variance[i], n_site, seed=seed)

            variance = .5 / (beta_b * self.mass)
            if self.debug > 0:
                print_matrix('velocity variance:', variance)

            for i in range(nmode):
                self.velocity[i] = get_gaussian_distribution(variance[i], n_site, seed=seed)

        if self.debug > 1:
            print_matrix('initial coordinates:', self.coordinate, 10)
            print_matrix('initial velocities:', self.velocity, 10)

        self.get_energy(self.velocity)


    def update_coordinate_velocity(self, force, half=1, **kwargs):
        r'''
        Update the oscillator coordinate and velocity, as well as the energy.

        Args:
            force (numpy.ndarray): Force acting on the oscillator.
            half (int): ``1`` or ``2``, indicating the first or second half of
                the update.

        Returns:
            numpy.ndarray: Projected force after the update, only for
            ``half=2``.
        '''
        if isinstance(self.frequency, np.ndarray) or isinstance(self.frequency, float): # add oscillator force first
            force -= np.einsum('i,i,ix->ix', self.mass, self.omega2, self.coordinate)

        if half == 1:
            if self.update_method == 'euler':
                self.euler_step(force)
            elif self.update_method == 'leapfrog':
                self.leapfrog_step(force)
            elif self.update_method == 'velocity_verlet':
                self.velocity_verlet_step(force, 1)
                # we will finish the last falf after electronic step
            elif self.update_method == 'recursive':
                force_func = kwargs.pop('force_func', None)
                self.recursive_exploration_step(force, force_func, **kwargs)

        elif half == 2:
            # velocity_verlet is cumbersome: energy is calculated here
            if self.update_method == 'velocity_verlet':
                self.velocity_verlet_step(force, 2)

            # no need to project at every step
            #self.project_velocity(self.velocity)
            #return self.project_force(force)
            self.force = force # save to class
            return force


    def euler_step(self, force):
        r"""Simple Euler integration step."""
        self.velocity += self.dt * np.einsum('ix,i->ix', force, 1./self.mass)
        self.coordinate += self.dt * self.velocity
        self.get_energy(self.velocity)


    def leapfrog_step(self, force):
        r"""Leapfrog integration step."""
        old_velocity = np.copy(self.velocity)
        self.velocity += self.dt * np.einsum('ix,i->ix', force, 1./self.mass)
        self.coordinate += self.dt * self.velocity

        average_velocity = .5 * (old_velocity + self.velocity)
        self.get_energy(average_velocity)


    def velocity_verlet_step(self, force, half):
        r"""Velocity Verlet integration step."""
        self.velocity += .5 * self.dt * np.einsum('ix,i->ix', force, 1./self.mass)
        if half == 1:
            self.coordinate += self.dt * self.velocity
        if half == 2:
            self.get_energy(self.velocity)


    def recursive_exploration_step(self, force, force_func, **kwargs):
        r"""Recursive exploration integration step."""
        N = self.recursive_numbers # >=1
        dt = self.dt
        mass = self.mass
        force0 = np.copy(force)
        x0 = np.copy(self.coordinate)

        # update coordinate
        for i in range(N, 0, -1):
            dv = np.einsum('ix,i->ix', force, 1./mass) * (dt/(2.*i))
            dx = (self.velocity + dv) * (dt/(2.*i-1.))
            force = force_func(x0+dx, **kwargs)[1]
        # we need the accumulation in place to use its pointer
        self.coordinate += dx

        # update velocity
        # find coordinate at where the effective acceleration is calculated
        force = np.copy(force0)
        for i in range(N, 1, -1):
            dv = np.einsum('ix,i->ix', force, 1./mass) * (dt/(2.*i-1.))
            dx = (self.velocity + dv) * (dt/(2.*i-2.))
            force = force_func(x0+dx, **kwargs)[1]
        self.velocity += np.einsum('ix,i->ix', force, 1./mass) * dt

        self.get_energy(self.velocity)


    def get_kinetic_energy(self, velocity, mass=None):
        r"""Calculate the kinetic energy and temperature."""
        if mass is None: mass = self.mass

        v2 = np.einsum('ix,ix->i', velocity, velocity)
        self.kinetic = .5* np.einsum('i,i', mass, v2)
        self.temperature = 2.* self.kinetic / (velocity.size)


    def get_potential_energy(self, mass=None, coordinate=None, omega2=None):
        r"""Calculate the potential energy."""
        if mass is None: mass = self.mass
        if coordinate is None: coordinate = self.coordinate
        if omega2 is None: omega2 = self.omega2

        v2 = np.einsum('ix,ix->i', coordinate, coordinate)
        self.potential = .5 * np.einsum('i,i,i->', mass, omega2, v2)


    def get_energy(self, velocity, mass=None, coordinate=None, omega2=None):
        r"""Calculate the total energy."""
        if mass is None: mass = self.mass

        self.get_kinetic_energy(velocity, mass)
        self.get_potential_energy(mass, coordinate, omega2)

        self.energy = self.kinetic + self.potential
        return self.energy



def angular_property(mass, coords, props):
    # props is for example force or velocity
    inertia = get_moment_of_inertia(mass, coords)

    # total angular property I^-1 * L
    U, s, Vt = np.linalg.svd(inertia)
    idx = np.where(s[s>1e-10])[0] # singular values are non-negative
    s, U, Vt = 1./s[idx], U[:,idx], Vt[idx]
    #Iinv = np.einsum('ji,j,kj->ik', Vt, s, U) # be careful to the index

    ang_prop = np.einsum('ji,j,kj,k->i', Vt, s, U, props)
    return ang_prop


def remove_trans_rotat_velocity(velocity, mass, coords):
    r"""
    Remove the translational and rotational components from the velocity.

    Args:
        velocity (ndarray): The velocities of the particles (shape: (n_particles, 3)).
        mass (ndarray): The masses of the particles (shape: (n_particles,)).
        coords (ndarray): The coordinates of the particles (shape: (n_particles, 3)).

    Returns:
        velocity (ndarray): The projected velocities with translational and rotational components removed.
"""
    # remove translation (ie. center of mass velocity)
    p_com = np.einsum('i,ix->x', mass, velocity) / len(mass)
    velocity -= np.einsum('i,x->ix', 1./mass, p_com)

    # move coords to com
    com = get_molecular_center(mass, coords)
    coords -= com

    # remove rotation
    ang_mom = np.einsum('i,ix->x', mass, np.cross(coords, velocity)) # r x v
    ang_vec = angular_property(mass, coords, ang_mom)

    velocity -= np.cross(ang_vec, coords)
    return velocity


def remove_trans_rotat_force(force, mass, coords):
    r"""
    Remove the translational and rotational components from the force.

    Args:
        force (ndarray): The forces acting on the particles (shape: (n_particles, 3)).
        mass (ndarray): The masses of the particles (shape: (n_particles,)).
        coords (ndarray): The coordinates of the particles (shape: (n_particles, 3)).

    Returns:
        force (ndarray): The projected forces with translational and rotational components removed.
"""
    # remove translation (ie. center of mass force)
    f_com = np.sum(force, axis=0) / np.sum(mass)
    force -= np.einsum('i,x->ix', mass, f_com)

    ## move coords to com
    #com = get_molecular_center(mass, coords)
    #coords -= com

    # remove rotation
    torque = np.sum(np.cross(coords, force), axis=0) # r x f
    ang_f = angular_property(mass, coords, torque)

    force -= np.einsum('i,ix->ix', mass, np.cross(ang_f[None,:], coords))
    return force



class OscillatorStep(harmonic_oscillator):
    r"""Oscillator dynamics step class."""
    def convert_parameter_units(self, unit_dict):
        r"""
        required input:
            mass in amu
            frequency in meV
        """
        from pyscf.data.nist import AMU2AU, HARTREE2EV
        # Boltzmann coefficient is 1/(k_B * T)
        self.beta_b = get_boltzmann_beta(self.init_temp)
        self.mass *= AMU2AU
        self.frequency /= (HARTREE2EV*1000)
        self.omega2 = self.frequency**2 # for computational efficiency

        if self.debug > 0:
            #print('KT:', 1./self.beta_b, 1./self.beta_b*HARTREE2EV*1000)
            print_matrix('oscillator mass (au):', self.mass)
            print_matrix('oscillator omega (au):', self.frequency)


    def get_phonon_hamiltonian(self, velocity=None, mass=None, coordinate=None,
                               omega2=None):
        print('Are we using get_phonon_hamiltonian??')
        # get phonon energy on each site
        if velocity is None: velocity = self.velocity
        if mass is None: mass = self.mass
        if coordinate is None: coordinate = self.coordinate
        if omega2 is None: omega2 = self.omega2

        # the first index of velocity and coordinate is mass related
        mass2 = np.copy(mass) *.5
        self.hamiltonian = np.einsum('i,in,in->n', mass2, velocity, velocity)
        v2 = np.einsum('i,i->i', mass2, omega2)
        self.hamiltonian += np.einsum('i,in,in->n', v2, coordinate, coordinate)

        return self.hamiltonian



class NuclearStep(harmonic_oscillator):
    r"""Nuclear dynamics step class."""
    def convert_parameter_units(self, unit_dict):
        self.natoms = len(self.atmsym)

        self.mass = np.zeros(self.natoms)
        # use pyscf's
        from pyscf.data import elements
        for i in range(self.natoms):
            self.mass[i] = elements.MASSES[elements.charge(self.atmsym[i])] / ELECTRON_MASS_IN_AMU
        #print_matrix('mass:\n', self.mass)

        # assume init_coords in AA and change it to A.U.
        self.coordinate = convert_units(np.reshape(self.coordinate, (-1, 3)), 'angstrom', 'bohr')


    # nuclear atoms only have kinetic energy
    def get_potential_energy(self, mass=None, coordinate=None, omega2=None):
        self.potential = 0.


    def init_coordinate_velocity(self, init_method=None):
        r"""
        Generate the initial nuclear coordinate and velocity.

        Args:
            init_method (str, optional): Method used to initialize the
                coordinates and velocities. ``'restart'`` reads stored values,
                ``'kick'`` sets zero initial velocity, ``'thermo'`` samples a
                Boltzmann distribution at ``init_temp``, and ``'random'`` uses
                random velocities with the given kinetic energy.
        """
        # coordinate has been given
        if init_method is None: init_method = self.init_method

        if 'restart' in init_method: # read in variables
            # has been stored in the class
            return

        elif 'kick' in init_method: # no initial velocity
            self.velocity = np.zeros((self.natoms, 3))
            self.kinetic = self.energy = 0.
            self.temperature = 0.

            self.project_force(self.force)
            return

        elif 'thermo' in init_method:
            self.velocity = self.init_velocity_thermo()
        elif 'random' in init_method:
            self.velocity = self.init_velocity_random()

        self.project_velocity(self.velocity)
        self.get_energy(self.velocity)
        self.force = np.zeros((self.natoms, 3))


    def init_velocity_thermo(self, temp=None, seed=1385448536):
        r"""Generate random velocity following Boltzmann distribution."""
        if temp is None: temp = self.init_temp

        beta_b = get_boltzmann_beta(temp)
        sigma = np.sqrt(1./beta_b/self.mass) # standard deviation

        velocity = np.zeros((self.natoms, 3))

        rng = np.random.default_rng(seed)
        for i in range(self.natoms):
            velocity[i] = rng.normal(loc=0., scale=sigma[i], size=3)

        print_matrix('init velocity:', velocity, digits=[10,5,'e'])
        return velocity


    def init_velocity_random(self, etrans=None, sigma=1e-4, scale=.1, seed=12345):
        r"""Generate random kinetic energy for atoms at three directions."""
        if etrans is None: etrans = self.etrans
        #etrans = convert_units(etrans*scale, 'eh', 'kcal')

        size = 3* self.natoms
        mean = etrans / float(size)

        rng = np.random.default_rng(seed)
        # mean is the center
        # sigma is the standard deviation whose square is variance
        ek = rng.normal(loc=mean, scale=sigma, size=size)
        ek = np.abs(ek) * etrans / np.sum(ek) # scale by the generated kinetic energy

        sign = rng.random((self.natoms, 3))
        sign = np.where(sign>.5, 1, -1)
        velocity = 2.* np.einsum('ix,i->ix', ek.reshape(self.natoms, 3), 1./self.mass)
        velocity = np.einsum('ix,ix->ix', sign, np.sqrt(velocity))
        return velocity


    def project_velocity(self, velocity, mass=None, coords=None):
        r"""
        Project out the translational and rotational components from the velocity."""
        if mass is None: mass = self.mass
        if coords is None: coords = self.coordinate

        self.velocity = remove_trans_rotat_velocity(velocity, mass, coords)


    def project_force(self, force, mass=None, coords=None):
        r"""Project out the translational and rotational components from the force."""
        if mass is None: mass = self.mass
        if coords is None: coords = self.coordinate

        self.force = remove_trans_rotat_force(force, mass, coords)
        return self.force


    def update_coordinate_velocity(self, force, half=1, **kwargs):
        r"""Update the nuclear coordinate and velocity."""
        super().update_coordinate_velocity(force, half, **kwargs)
        if half == 2:
            mass = self.mass
            coords = self.coordinate
            self.project_velocity(self.velocity, mass, coords)
            return self.project_force(force, mass, coords)
