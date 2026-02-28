import warnings

from lumeq import np, itertools
from lumeq.utils import (
        parser, print_matrix, convert_units, Sampler,
        put_keys_kwargs_to_object, put_kwargs_to_keys,
        load_json, save_json)
from lumeq.dynamics.dimers_in_crystal import add_molecule
from lumeq.dynamics.process_exciton_parameters import process_parameters, set_model
from lumeq.utils.sec_mole import write_symbols_coords

class Exciton():
    r"""
    Exciton class at each time for energy transport simulation.
    It builds the exciton Hamiltonian and update the exciton coefficients.
    """
    def __init__(self, key={}, **kwargs):
        r"""
        Parameters
            key : dict of the input parameters
            **kwargs : other input parameters
            The required parameters are listed below with defalut values.
        """
        # default parameters
        self.debug = 1
        self.temperature = 298 # system temperature in K
        self.exciton_dt = 1 # exciton dynamics time step in au
        self.nstate = 0 # exciton number of each site

        self.propogator = 'magnus' # time propagation method

        # unit cell informations
        # including intermolecular distance: abc
        # angle, element, and coordinate: scale
        # number of molecules and unit cell size
        self.unit_cell = {'abc': 0, 'angle': 0, 'element': 0, 'scale': 0,
                          'n_mol' : 0}
        self.n_cell = [5, 5, 5] # number of unit cells [nx, ny, nz]
        self.n_cell_param = [5, 5, 5] # unit cells extract parameters

        self.energy = 0 # exciton energy of each site
        self.coupling_j = 0 # (ndimer, nstate, nstate) exciton-exciton couplings
        self.dipole = 0 # transition dipole of each site

        self.n_mode = 0 # vibrational mode number of each site
        # exciton-phonon couplings with phonon modes
        self.coupling_g = 0 # (nmode, nstate) on-site
        self.coupling_a = 0 # (ndimer, nmode, nstate, nstate) off-diagonal

        self.param_dir = './' # param file directory

        # add user-defined parameters
        put_keys_kwargs_to_object(self, key, **kwargs)

        self.process_parameters()
        # convert the input parameters into atomic unit for the calculations
        self.convert_parameter_units(getattr(self, 'unit_dict', {}))

        self.check_sanity()


    def process_parameters(self):
        r"""Process the input parameters and get real parameters."""
        # number of molecules in an unit cell
        self.n_mol = n_mol = self.unit_cell['n_mol']

        # build the list of unit cells
        nx, ny, nz = self.n_cell
        cells = itertools.product(range(nx), range(ny), range(nz))
        self.cells = getattr(self, 'cells', list(cells))
        print('number of cells:', len(self.cells))
        if self.debug >= 0:
            print('cells:\n', self.cells)

        cells = self.cells
        neighbor_index = self.neighbor_index

        # dictionary for cell index searching. O(1) loopup
        cell_dict = {tuple(row): idx for idx, row in enumerate(cells)}
        cell_dict_r = {idx: tuple(row) for idx, row in enumerate(cells)}

        # premap the hamiltonian coupling index
        sites = [] # left and right sites
        hamil_index = []
        for ic, (i, j, k) in enumerate(cells):
            for x, (l, (a, b, c, d)) in enumerate(neighbor_index):
                # l and d are the molecule index in the unit cell
                a, b, c = i+a, j+b, k+c
                jc = cell_dict.get(tuple([a,b,c]), None)
                if jc:
                    # ic, jc are the indices of hamiltonian elements
                    # l, d are the molecule indices in the unit cell
                    # x is the index for coupling_j
                    sites.append(ic*n_mol+l)
                    sites.append(jc*n_mol+d)
                    hamil_index.append([ic, l, jc, d, x])
                    #print('index: ', ic, '[',i,j,k,l,'];', jc, '[',a,b,c,d,'];',x)

        self.hamil_index = hamil_index
        self.sites = sites = list(set(sites))
        print('%4d out of %4d molecules are included in the Hamiltonian.'
              % (len(self.sites), int((ic+1)*n_mol)))
        hamil_filter = (np.reshape(sites, (-1,1))*2 + np.arange(self.nstate)[None,:]).ravel()
        self.hamil_filter = np.ix_(hamil_filter, hamil_filter) # remove isolated entries in hamiltonian
        if self.debug >= 2:
            print('sites:', self.sites)
            print('hamil_index:\n', self.hamil_index)
            isolated_sites = set(range((ic+1)*n_mol)) - set(sites)
            for s in isolated_sites:
                print('Isolated site:', s, cell_dict_r[s//n_mol], s%n_mol)


        self.n_cell = ic + 1 # total number of unit cells
        self.n_site = len(sites) # total number of sites included

        # add length for c2 calculation
        abc, angles, scales = self.unit_cell['abc'], self.unit_cell['angles'], self.unit_cell['scales']
        center = np.array([.5 * (scales[4] + scales[8])])
        length = np.zeros((self.n_cell, self.n_mol, 3))
        for icount, (i, j, k) in enumerate(cells):
            for s in range(self.n_mol):
                # use s rather s+1 below to be consistent to the definition in
                # dimers_in_crystal.py
                length[icount,s] = add_molecule(i,j,k,s,abc,angles,None,center)

        length = length.reshape(-1, 3)[self.sites]
        #print_matrix('length (AA):', length)
        #print_matrix('distances (AA):', np.linalg.norm((length[:,None,:]-length).reshape(-1,3), axis=1))
        self.length = length - np.mean(length, axis=0)
        #print_matrix('length (AA):', self.length)
        if getattr(self, 'write_coords', False):
            self.print_site_coords()


    def print_site_coords(self, xyzfile=None):
        r"""Print site coordinates to xyz file for visualization."""
        if xyzfile is None:
            xyzfile = 'molecular_sites_'+str(self.model)+'_'+str(self.n_cell)
            xyzfile = getattr(self, 'xyzfile', xyzfile)
        if xyzfile[-4] != '.xyz':
            xyzfile += '.xyz'
        symbols = ['Ag'] * self.n_site
        coords = self.length
        write_symbols_coords(xyzfile, symbols, coords)
        print('Molecular site coordinates written to', xyzfile)


    def convert_parameter_units(self, unit_dict={}):
        r"""Convert the input parameters into atomic unit for the calculations."""
        from lumeq.dynamics import get_boltzmann_beta
        BOHR = convert_units(1., 'aa', 'bohr')
        self.beta_b = get_boltzmann_beta(self.temperature)
        self.energy = convert_units(self.energy, unit_dict.get('energy', 'ev'), 'eh')
        self.coupling_j = convert_units(self.coupling_j, unit_dict.get('coupling_j', 'ev'), 'eh')
        self.coupling_g = convert_units(self.coupling_g, unit_dict.get('coupling_g', 'ev'), 'eh') / BOHR
        self.coupling_a = convert_units(self.coupling_a, unit_dict.get('coupling_a', 'ev'), 'eh') / BOHR


    def check_sanity(self):
        r"""Check the sanity of the input parameters."""
        if self.energy.size != self.n_site*self.nstate:
            # pointer not a copy
            self.energy = np.tile(self.energy, (self.n_site, 1)).ravel()

        if self.dipole.size != self.n_site*self.nstate*3:
            # pointer not a copy
            self.dipole = np.tile(self.dipole, (self.n_site, 1,1))

        if self.coupling_j.ndim == 3:
            self.coupling_j = np.tile(self.coupling_j, (self.n_cell, 1,1,1))

        #print_matrix('Exciton energy (au):', self.energy)
        #print_matrix('Exciton coupling_j (au):', self.coupling_j)
        #print_matrix('Exciton dipole (au):', self.dipole)


    def exciton_diagonal(self, energy=None, **kwargs):
        r"""Return the on-site energy."""
        if energy is None: energy = self.energy
        return energy


    def exciton_couplings(self, coupling_j=None, neighbor_index=None, **kwargs):
        r"""Build the exciton couplings part of exciton Hamiltonian."""
        if coupling_j is None: coupling_j = self.coupling_j
        hamil_index = self.hamil_index

        n_cell, nt, ns = self.n_cell, self.n_mol, self.nstate
        # n_site = n_cell*nt
        hamiltonian = np.zeros((n_cell, nt, ns, n_cell, nt, ns))

        for (left, l, right, d, x) in hamil_index:
            hamiltonian[left,l,:,right,d] = coupling_j[left, x]

        hamiltonian = hamiltonian.reshape(n_cell*nt*ns, -1)
        hamiltonian = hamiltonian[self.hamil_filter]
        hamiltonian += hamiltonian.conj().T
        return hamiltonian


    def get_hamiltonian(self, energy=None, coupling_j=None,
                        neighbor_index=None, **kwargs):
        r"""
        Build the periodic exciton Hamiltonian of exciton part.

        Parameters
            energy : (nstate) exciton energy same for each site
            coupling_j : (ndimer, nstate, nstate) exciton-exciton couplings
            neighbor_index : list of neighbor index for couplings

        Returns
            hamiltonian : (nsite*nstate, nsite*nstate) exciton Hamiltonian
        """
        self.hamiltonian = self.exciton_couplings(coupling_j, neighbor_index, **kwargs)
        diagonal = self.exciton_diagonal(energy, **kwargs)
        np.fill_diagonal(self.hamiltonian, diagonal)

        return self.hamiltonian


    def get_initial_coefficients(self, H=None, method=None, **kwargs):
        r"""
        Get the initial exciton coefficients from the exciton Hamiltonian.

        Parameters
            H : (nsite*nstate, nsite*nstate) exciton Hamiltonian
            method : str, method to get the initial coefficients
                'random' : random coefficients
                'equal' : equal superposition state
                'ground' : ground state
                'thermal' : thermal equilibrium state at given temperature

        Returns
            c2 : (nsite*nstate) square of exciton coefficients
        """
        if H is None: H = self.get_hamiltonian(**kwargs)
        if method is None: method = getattr(self, 'initial_method', 'random')

        if method == 'random':
            idx = getattr(self, 'n_site_init', 0)
            rng = np.random.default_rng(kwargs.get('seed', None))
            if type(idx) is str:
                length = np.linalg.norm(self.length, axis=1)
                order = np.argsort(length)
                if idx == 'center':
                    idx = order[self.n_site//2]
                elif idx == 'edge0':
                    idx = order[0]
                elif idx == 'edge1':
                    idx = order[-1]
                idx = int(idx)
            if type(idx) is int: idx = [idx]
            idx = [x for i in idx for x in (2*i, 2*i+1)]
            print('Wavafunction initializes at sites*states:', idx)
            values = rng.random(len(idx)) + 1j * rng.random(len(idx))
            values /= np.linalg.norm(values)
            coeffs0 = np.zeros(self.n_site*self.nstate, dtype=complex)
            for i, v in zip(idx, values):
                coeffs0[i] += v

        elif method in {'ground', 'thermal'}:
            evals, evecs = np.linalg.eigh(H)

            if method == 'ground':
                coeffs0 = evecs[:,0] # ground state

            elif method == 'thermal':
                weights = np.exp(- self.beta_b * evals)
                weights /= np.sum(weights)
                # TODO what is random choice here?
                index = np.random.choice(len(evals), p=weights)
                coeffs0 = evecs[:,index]

        self.coefficients = coeffs0
        c2 = np.einsum('i,i->i', self.coefficients.conj(), self.coefficients)
        return c2.real


    def update_coefficients(self, H=None, dt=None, exp_h=None, **kwargs):
        r"""
        Update the exciton coefficients with time step dt.

        Parameters
            dt : float, time step in au

        Returns
            c2 : (nsite*nstate) square of exciton coefficients
        """
        if exp_h is None:

            if H is None: H = self.get_hamiltonian(**kwargs)
            if dt is None: dt = self.exciton_dt

            evals, evecs = np.linalg.eigh(self.hamiltonian)
            exp_h = np.exp((-1j * dt) * evals)
            exp_h = np.einsum('ji,i,ki->jk', evecs.conj(), exp_h, evecs)

            self.h_eigens = [evals, evecs] # save for analysis if needed

        self.coefficients = np.einsum('ij,j->i', exp_h, self.coefficients)
        # c2 should be pure real even though the following is complex type
        c2 = np.einsum('i,i->i', self.coefficients.conj(), self.coefficients)
        return c2.real


    def cal_energy(self, hamiltonian=None, coefficients=None):
        r"""Calculate the exciton energy expectation value."""
        if hamiltonian is None: hamiltonian = self.hamiltonian
        if coefficients is None: coefficients = self.coefficients

        self.e_tot = np.einsum('i,ij,j->', coefficients.conj(), hamiltonian, coefficients)
        #print('exciton e_tot (au):', e_tot.real, e_tot.imag)
        return self.e_tot.real


    def cal_spectra(self, dipole=None, coefficients=None):
        r"""
        Calculate the exciton absorption spectra.

        Parameters
            dipole : (nsite*nstate, 3) transition dipole of each site
            coefficients : (nsite*nstate) exciton coefficients

        Returns
            evals : (nsite*nstate) exciton eigenvalues
            f : (nsite*nstate) oscillator strengths
        """
        if hasattr(self, 'h_eigens'):
            evals, evecs = self.h_eigens
        else:
            evals, evecs = np.linalg.eigh(self.hamiltonian)

        dipole = np.reshape(self.dipole, (self.n_site*self.nstate, 3))
        mu = np.einsum('ix,ik->kx', dipole, evecs)
        f = (2./3.) * np.einsum('k,kx,kx->k', evals, mu.conj(), mu)
        return evals, f


    def cal_r_correlation(self, coefficients=None, c2=None):
        r"""Calculate the exciton r^2 - <r>^2 correlation value."""
        if c2 is None:
            if coefficients is None: coefficients = self.coefficients
            c2 = np.einsum('i,i->i', coefficients.conj(), coefficients)

        c2 = np.reshape(c2, (self.n_site, -1))
        r2 = np.einsum('nx,nx,ni->x', self.length, self.length, c2)
        r_2 = np.einsum('nx,ni->x', self.length, c2)**2
        correlation = np.array([r2, r_2])
        #print('c2:', np.sum(c2), c2.shape)
        #print('correlation:', correlation.real, correlation.imag)
        return correlation.real


    def cal_ipr_value(self, coefficients=None, c2=None):
        r"""Calculate the inverse participation ratio (ipr) value."""
        if c2 is None:
            if coefficients is None: coefficients = self.coefficients
            c2 = np.einsum('i,i->i', coefficients.conj(), coefficients)

        # inverse participation ratio (ipr)
        ipr = 1./ np.einsum('i,i->', c2, c2)
        #print('ipr: %8.6f %10.8f' % ipr.real, ipr.imag)
        return ipr.real


    def analyze_wf_property(self, coefficients=None, c2=None):
        r"""Analyze the exciton wavefunction properties including r-correlation and ipr."""
        if c2 is None:
            if coefficients is None: coefficients = self.coefficients
            c2 = np.einsum('i,i->i', coefficients.conj(), coefficients)

        correlation = self.cal_r_correlation(c2=c2)
        ipr = self.cal_ipr_value(c2=c2)

        return correlation, ipr



class ExcitonMC(Exciton):
    r"""
    Thermal fluctuations are accounted for by Monte Carlo sampling Hamiltonian.
    """
    def get_sampler(self, variance=0.1, seed=123, **kwargs):
        r"""
        Get the sampler for Monte Carlo sampling of exciton Hamiltonian.

        Parameters
            variance : float, variance of the random numbers
            seed : int, seed for random number generator
        """
        self.onsite_disorder = kwargs.get('onsite_disorder', True)
        self.coupling_disorder = kwargs.get('coupling_disorder', False)

        sampler = Sampler(size=(self.n_site*self.nstate), variance=variance,
                          seed=seed, **kwargs)
        if hasattr(self, 'tau_c'): # get the function pointer
            self.sample = sampler.correlated_sample
            print('Using correlated sampling with tau_c = %.3f fs.' % convert_units(self.tau_c, 'au', 'fs'))
        else:
            self.sample = sampler.sample

        # save unperturbed parameters
        self.energy0 = np.copy(self.energy)
        self.coupling_j0 = np.copy(self.coupling_j)

        # initial disorders
        self.delta_e = np.zeros_like(self.energy)
        self.delta_j = np.zeros_like(self.coupling_j)


    def update_parameters(self, sample=None, dt=None):
        r"""
        Update the exciton Hamiltonian parameters by Monte Carlo sampling.

        Parameters
            sample : Sampler.sample() function to sample the random numbers
                (pass in callable function for efficiency)
        """
        if sample is None: sample = self.sample
        if dt is None: dt = self.exciton_dt

        if self.onsite_disorder:
            self.delta_e = sample(self.delta_e, self.tau_c, dt)
            self.energy = self.energy0 + self.delta_e

        if self.coupling_disorder:
            self.delta_j = sample(self.delta_j, self.tau_c, dt)
            self.coupling_j = self.coupling_j0 + self.delta_j


    def split_operator(self, dt=None, **kwargs):
        r"""
        Get the split operator for time propagation.

        Math `exp(-i H dt) = exp(-i H_d dt/2) exp(-i H_c dt) exp(-i H_d dt/2)`

        Parameters
            dt : float, time step in au

        Returns
            exp_h : (nsite*nstate, nsite*nstate) time propagation operator
        """
        if dt is None: dt = self.exciton_dt

        J = self.exciton_couplings(**kwargs)
        from scipy.linalg import expm
        exp_j = expm(-1j * J * dt)

        dt_half = dt / 2.
        def kernel():
            exp_d = np.exp((-1j * dt_half) * self.energy)
            exp_h = np.einsum('i,ij,j->ij', exp_d, exp_j, exp_d)
            return exp_h

        return kernel



class ExcitonStep(Exciton):
    r"""
    Thermal fluctuations are accounted for by explicit vibrational oscillators.
    Requires the exciton-phonon couplings, and oscillator coordinates.
    """
    #def get_hamiltonian(self, coordinate=None):
    #    r"""
    #    Build the fluactuating part from oscillators of exciton Hamiltonian
    #    along with the pure excitonic Hamiltonian.
    #    """
    #    self.hamiltonian = super().get_hamiltonian()

    #    # TODO: add the coordinate dependence for 3D
    #    raise NotImplementedError("get_hamiltonian is not implemented for 3D yet.")

    #    (nx, ny, nz), nt, ns = self.n_cell, self.n_mol, self.nstate
    #    # n_site = nx*ny*nz*nt
    #    hamiltonian = np.zeros((nx, ny, nz, nt, ns, nx, ny, nz, nt, ns))

    #    self.hamiltonian += hamiltonian.reshape(self.n_site*nstate, -1)
    #    return self.hamiltonian


    def exciton_diagonal(self, coordinate):
        r"""Return the on-site energy with coordinate dependence."""
        diagonal = super().exciton_diagonal()

        #TODO
        warnings.warn('not checked function for 3D!')
        diagonal += np.einsum('mi,mn->ni', self.coupling_g, coordinate)
        return diagonal.ravel()


    def exciton_couplings(self, coupling_j=None, neighbor_index=None,
                          coupling_a=None, coordinate=None):
        r"""
        Build the exciton couplings part with coordinate fluctuation.

        Parameters
            coupling_j : (ndimer, nstate, nstate) exciton-exciton couplings
            neighbor_index : list of neighbor index for couplings
            coupling_a : (ndimer, nmode, nstate, nstate) off-diagonal exciton-phonon couplings
            coordinate : (nsite, nmode) oscillator coordinates

        Returns
            hamiltonian : (nsite*nstate, nsite*nstate) exciton Hamiltonian
        """
        if coupling_j is None: coupling_j = self.coupling_j
        if neighbor_index is None: neighbor_index = self.neighbor_index
        if coupling_a is None: coupling_a = self.coupling_a
        if coordinate is None: coordinate = np.zeros((self.n_site, self.n_mode))

        (nx, ny, nz), nt, ns = self.n_cell, self.n_mol, self.nstate
        # n_site = nx*ny*nz*nt
        hamiltonian = np.zeros((nx, ny, nz, nt, ns, nx, ny, nz, nt, ns))

        #TODO
        warnings.warn('not checked function for 3D!')

        coordinate1 = np.copy(coordinate)
        coordinate1[:,:-1] -= coordinate[:,1:]

        for icount, (i, j, k) in enumerate(itertools.product(range(1, nx-1), range(1, ny-1), range(1, nz-1))):
            coupling = np.einsum('tmij,m->tij', coupling_a, coordinate1[icount])
            _coupling_j = coupling_j[icount]
            for x, (l, (a, b, c, d)) in enumerate(neighbor_index):
                # l and d are the molecule index in the unit cell
                a, b, c = i+a, j+b, k+c
                if 0<=a<nx and 0<=b<ny and 0<=c<nz:
                    hamiltonian[i,j,k,l,:,i+a,j+b,k+c,d] = _coupling_j[x] + coupling

        hamiltonian = hamiltonian.reshape(self.n_site*ns, -1)
        hamiltonian += hamiltonian.conj().T
        return hamiltonian


    def cal_force(self, coordinate, coefficients=None):
        r"""
        Calculate the forces on each oscillators from exciton-phonon couplings.
        """
        if coefficients is None: coefficients = self.coefficients

        #force = np.zeros((self.n_site_tot, self.n_mode))

        # TODO: change for 3D
        warnings.warn('not checked function for 3D!')

        (nx, ny, nz), nt, ns = self.n_cell, self.n_mol, self.nstate
        coefficients = np.reshape(coefficients, (nx, ny, nz, nt, ns))
        force = np.zeros((nx, ny, nz, nt, self.n_mode))

        for icount, (i, j, k) in enumerate(itertools.product(range(1, nx-1), range(1, ny-1), range(1, nz-1))):
            for x, (l, (a, b, c, d)) in enumerate(neighbor_index):
                # l and d are the molecule index in the unit cell
                a, b, c = i+a, j+b, k+c
                if 0<=a<nx and 0<=b<ny and 0<=c<nz:
                    c2[x] = np.einsum('i,j->ij', coefficients[i,j,k,l].conj(), coefficients[i+a,j+b,k+c,d])
                force[i,j,k,l] = -2.* np.einsum('tmij,tij->m', coupling_a, c2.real)

        return force



class ExcitonDynamics():
    r"""
    Exciton dynamics for energy transport simulation.
    Wrapper class containing exciton and phonon dynamics steps.
    """
    def __init__(self, key, **kwargs):
        r"""
        Parameters
            key : dictionary of input parameters
            kwargs : keyword arguments for input parameters
            The required parameters are listed below with defalut values.
        """
        put_kwargs_to_keys(key, **kwargs)

        # pop out outer md parameters
        self.output_name = key.pop('output_name', 'exciton_dynamics')
        self.dt = key.pop('dt', None) # time step in au
        # print the energy every printing_nsteps steps
        self.printing_nsteps = key.pop('printing_nsteps', 100)
        self.debug = key.pop('debug', 0)
        # only take the nsteps here
        self.nsteps = key.pop('nsteps', 10) # total number steps for dynamics
        self.nsteps += 1 # add one for the initial step

        self.set_dynamics_class(key)

        if self.dt == None:
            self.dt = self.edstep.exciton_dt
        self.total_time = (self.nsteps - 1) * self.dt
        print('Exciton dynamics runs for %4d steps in %6.3f fs.' % (self.nsteps, convert_units(self.total_time, 'au', 'fs')))

        # variables needed
        self.total_energy = np.empty(self.nsteps)
        # <r^2> and <r>^2 correlation for each step
        self.correlation = np.empty((self.nsteps, 2, 3))
        self.ipr = np.empty(self.nsteps)
        self.c2 = np.empty((self.nsteps, self.edstep.n_site))


    def set_dynamics_class(self, key):
        r"""Set the dynamics class for exciton and phonon dynamics."""
        self.edstep = ExcitonStep(key)

        key['n_site'] = self.edstep.n_site
        self.ndstep = OscillatorStep(key)


    def kernel(self):
        # assign local variables for class attributes
        ndstep, edstep = self.ndstep, self.edstep

        coords = ndstep.coordinate # equal sign used here as a pointer
        c2 = edstep.get_initial_coefficients(coordinate=coords)
        correlation, ipr = edstep.analyze_wf_property(c2=c2)

        n_site = edstep.n_site
        self.c2[0] = np.sum(c2.reshape(n_site, -1), axis=1)
        self.correlation[0] = correlation
        self.ipr[0] = ipr

        self.total_energy[0] = edstep.cal_energy() + ndstep.energy
        force = edstep.cal_force(coords)

        for ti in range(1, self.nsteps):
            ndstep.update_coordinate_velocity(force)
            c2 = edstep.update_coefficients(coordinate=coords)
            correlation, ipr = edstep.analyze_wf_property(c2=c2)

            self.c2[ti] = np.sum(c2.reshape(n_site, -1), axis=1)
            self.correlation[ti] = correlation
            self.ipr[ti] = ipr

            # velocity_verlet is cumbersome
            force = ndstep.update_coordinate_velocity(force, 2)

            self.total_energy[ti] = edstep.cal_energy() + ndstep.energy
            force = edstep.cal_force(coords)



class ExcitonDynamicsMC(ExcitonDynamics):
    r"""
    Exciton dynamics with Monte Carlo sampling Hamiltonian.
    Wrapper class containing exciton dynamics with MC disorders.
    """
    def set_dynamics_class(self, key):
        r"""Set the dynamics class for exciton Monte Carlo dynamics."""
        self.edstep = ExcitonMC(key)
        self.edstep.get_sampler(variance=key.get('mc_variance', 0.1),
                                seed=key.get('random_seed', None),
                                covariance=key.get('mc_covariance', False))


    def kernel(self):
        # assign local variables for class attributes
        edstep = self.edstep

        c2 = edstep.get_initial_coefficients()
        correlation, ipr = edstep.analyze_wf_property(c2=c2)

        n_site = edstep.n_site
        self.c2[0] = np.sum(c2.reshape(n_site, -1), axis=1)
        self.correlation[0] = correlation
        self.ipr[0] = ipr

        self.total_energy[0] = edstep.cal_energy()

        if getattr(edstep, 'propogator', 'magnus'):
            # hamiltonian is half step advanced
            edstep.update_parameters(dt=edstep.exciton_dt/2.)
        if getattr(edstep, 'propogator', 'split'):
            exp_h = edstep.split_operator(dt=edstep.exciton_dt)
        else:
            exp_h = lambda: None

        for ti in range(1, self.nsteps):
            edstep.update_parameters()
            c2 = edstep.update_coefficients(exp_h=exp_h())
            correlation, ipr = edstep.analyze_wf_property(c2=c2)

            self.c2[ti] = np.sum(c2.reshape(n_site, -1), axis=1)
            self.correlation[ti] = correlation
            self.ipr[ti] = ipr

            self.total_energy[ti] = edstep.cal_energy()

            if ti % self.printing_nsteps == 0:
                print('Step %4d has energy %12.6f eh' % (ti, self.total_energy[ti]))

        self.total_energy = convert_units(self.total_energy, 'eh', 'ev')

        np.savez_compressed(self.output_name+'.npz',
                            total_energy=self.total_energy,
                            c2=self.c2,
                            correlation=self.correlation,
                            ipr=self.ipr,)
        if self.debug >= 1:
            print_matrix('total_energy (eV):', self.total_energy)
            print_matrix('correlation (bohr^2):', self.correlation)



def setup_exciton_dynamics(infile, key={}, nsteps=300):
    r"""
    Run the exciton dynamics simulation.

    Parameters
        infile : input file for dynamics parameters
        key : dict of the input parameters
        nsteps : int, total time steps for dynamics

    Returns
        obj : ExcitonDynamicsMC object
    """
    parameters = parser(infile)
    key.update(parameters.get('exciton', {}))

    do_mc = parameters.get('monte_carlo', None)
    if do_mc: # use Monte Carlo sampling
        key.update(parameters.get('monte_carlo', {}))
    else: # use explicit vibrational phonon modes
        key.update(parameters.get('phonon', {}))

    key['debug'] = key.get('debug', 0)
    key['random_seed'] = key.get('random_seed', None)
    key['nsteps'] = key.get('nsteps', 300)
    key['exciton_dt'] = key.get('exciton_dt', 10)
    key['n_cell'] = key.get('n_cell', [5, 5, 5])
    key['n_cell_param'] = key.get('n_cell_param', [5, 5, 5])
    key['nstate'] = key.get('nstate', 2)
    key['propagator'] = key.get('propagator', 'split')

    key['cif'] = key.get('cif', 'H2OBPc.cif')
    key['json'] = key.get('json', 'H2OBPc.cif.json')
    key['param_dir'] = key.get('param_dir', './')
    key['write_coords'] = key.get('write_coords', False)

    if key.get('json', None):
        data = load_json(key['param_dir']+key['json'], to_numpy=True)
    if data is None and key.get('cif', None):
        data = process_parameters(key['cif'], key['n_cell_param'],
                                  key['param_dir'], key['nstate'])
        save_json(key['cif']+'.json', data)
    cells, neighbor_index = set_model(data['neighbor_index'],
                                      data['distances'],
                                      key.get('model', 'AB'),
                                      key.get('n_cell', [5,5,5]),
                                      key.get('r_cutoff', 10))
    data['cells'] = cells
    data['neighbor_index'] = neighbor_index
    key.update(data) # real parameters

    if do_mc:
        obj = ExcitonDynamicsMC(key)
    else:
        obj = ExcitonDynamics(key)
    return obj


if __name__ == '__main__':
    from lumeq import sys
    from lumeq.plot import plt, broadening

    infile = sys.argv[1]
    obj = setup_exciton_dynamics(infile)

    if obj.debug >= 1: # print the exciton Hamiltonian and spectra for analysis
        obj.edstep.get_hamiltonian()
        #print_matrix('Exciton Hamiltonian (au):', obj.edstep.hamiltonian)
        e, f = obj.edstep.cal_spectra()
        e_shift = 0.47 # ev
        e = convert_units(e-convert_units(e_shift, 'ev', 'eh'), 'eh', 'nm')

        fig, ax = plt.subplots()
        e, f = broadening(e, f, method='voigt', margin=20, width=25, gamma=20)
        ax.plot(e, f.real)
        ax.set_xlabel('Exciton wavelength (nm)')
        ax.set_ylabel('Oscillator strength (au)')
        plt.show()


    # write to file directly
    #from contextlib import redirect_stdout
    #with open(obj.output_name+'.out', 'w') as f, redirect_stdout(f):
    #    obj.kernel()
    obj.kernel()

    fig, ax = plt.subplots(3,1)
    ax[0].plot(convert_units(np.arange(obj.nsteps)*obj.dt, 'au', 'fs'), obj.total_energy)
    r2 = obj.correlation[:,0] - obj.correlation[0,1]
    for x in range(3):
        ax[1].plot(convert_units(np.arange(obj.nsteps)*obj.dt, 'au', 'fs'), r2[:,x], label='R$^2$-'+str(chr(ord('x') + x)))
    ax[1].plot(convert_units(np.arange(obj.nsteps)*obj.dt, 'au', 'fs'), np.sum(np.abs(r2), axis=1), label='R$^2$-tot')
    ax[1].legend()
    ax[0].set_ylabel('Energy (eV)')
    ax[1].set_ylabel('<R$^2>-<R_0>^2$ ($\\AA^2$)')
    ax[1].set_xlabel('Time (fs)')


    save_c2_steps = [0, 10, 100, 500, 1000, 5000]
    save_c2_steps = [x for x in save_c2_steps if x < obj.nsteps]
    length = obj.edstep.length
    length = np.linalg.norm(length, axis=1)
    idx = np.argsort(length)
    n = len(obj.c2[0])
    ns, nd = int(n//2-10), int(n//2+10)
    for i, c2 in enumerate(obj.c2[save_c2_steps]):
        i = save_c2_steps[i]
        c2 = c2[idx]
        ax[2].plot(range(ns, nd), c2[ns:nd], label='t$=$ %d (fs)' % convert_units(i*obj.dt, 'au', 'fs'))
    ax[2].set_ylabel('c$^2$')
    ax[2].set_xlabel('Site index')
    ax[2].set_xticks(range(ns, nd))
    ax[2].legend()
    plt.tight_layout()

    plt.show()
