import warnings

from lumeq import np, itertools
from lumeq.utils import (
        parser, print_matrix, convert_units, Sampler,
        put_keys_kwargs_to_object, put_kwargs_to_keys,
        load_json, save_json, monitor_performance)
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
        Args:
            key (dict): Input parameters.
            **kwargs: Additional input parameters.

        Notes:
            The required parameters are listed below with default values.
        """
        # default parameters
        self.debug = 1
        self.temperature = 298 # system temperature in K
        self.exciton_dt = 1 # exciton dynamics time step in au
        self.nstate = 0 # exciton number of each site

        self.propagator = 'magnus' # magnus (midpoint), split (left), magnus_split (midpoint)

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

        self.onsite_disorder = getattr(self, 'onsite_disorder',
                                       bool(np.any(self.coupling_g)))
        self.coupling_disorder = getattr(self, 'coupling_disorder',
                                         bool(np.any(self.coupling_a)))


    def process_parameters(self):
        r"""Process the input parameters and get real parameters."""
        # number of molecules in an unit cell
        self.n_mol = n_mol = self.unit_cell['n_mol']

        # build the list of unit cells
        nx, ny, nz = self.n_cell
        # Do not eagerly build the full nx*ny*nz box when set_model already
        # supplied a small periodic image envelope (or a finite model path).
        if not hasattr(self, 'cells'):
            self.cells = list(itertools.product(range(nx), range(ny), range(nz)))
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
                if jc is not None:
                    # ic, jc are the indices of hamiltonian elements
                    # l, d are the molecule indices in the unit cell
                    # x is the index for coupling_j
                    sites.append(ic*n_mol+l)
                    sites.append(jc*n_mol+d)
                    hamil_index.append([ic, l, jc, d, x])
                    #print('index: ', ic, '[',i,j,k,l,'];', jc, '[',a,b,c,d,'];',x)

        self.hamil_index = hamil_index
        self.sites = sites = sorted(set(sites))
        print('%4d out of %4d molecules are included in the Hamiltonian.'
              % (len(self.sites), int((ic+1)*n_mol)))
        hamil_filter = (np.reshape(sites, (-1,1))*self.nstate + np.arange(self.nstate)[None,:]).ravel()
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
        if not xyzfile.endswith('.xyz'):
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
        r"""Return and store a copy of the onsite energies.

        Args:
            energy (numpy.ndarray or None): Energies in Hartree, shape ``(M,)``;
                None uses self.energy, where M=n_site*nstate.
            **kwargs: Unused additional builder arguments.

        Returns:
            numpy.ndarray: Copied energies, shape ``(M,)``, stored as h_diagonal.
        """
        if energy is None: energy = self.energy
        self.h_diagonal = energy.copy()
        return self.h_diagonal


    def exciton_couplings(self, coupling_j=None, neighbor_index=None, **kwargs):
        r"""Build the Hermitian coupling matrix for the retained sites.

        The precomputed hamil_index and hamil_filter determine the topology and
        basis order. The assembled matrix is added to its conjugate transpose.
        For Strang splitting, the coupling topology must give diag(J)=0.

        Args:
            coupling_j (numpy.ndarray or None): Couplings in Hartree, shape
                ``(n_cell, ndimer, nstate, nstate)``; None uses self.coupling_j.
            neighbor_index (list or None): Unused compatibility argument;
                this implementation uses the precomputed hamil_index.
            **kwargs: Unused additional builder arguments.

        Returns:
            numpy.ndarray: Coupling matrix J, shape ``(M, M)``, where
                M=n_site*nstate. A separate copy is stored as h_coupling.
        """
        if coupling_j is None: coupling_j = self.coupling_j
        hamil_index = self.hamil_index

        n_cell, nt, ns = self.n_cell, self.n_mol, self.nstate
        # n_site = n_cell*nt
        hamiltonian = np.zeros((n_cell, nt, ns, n_cell, nt, ns),
                               dtype=np.result_type(coupling_j, float))

        for (left, l, right, d, x) in hamil_index:
            hamiltonian[left,l,:,right,d] = coupling_j[left, x]

        hamiltonian = hamiltonian.reshape(n_cell*nt*ns, -1)
        hamiltonian = hamiltonian[self.hamil_filter]
        hamiltonian += hamiltonian.conj().T

        self.h_coupling = hamiltonian.copy()
        return hamiltonian


    def get_hamiltonian(self, energy=None, coupling_j=None,
                        neighbor_index=None, **kwargs):
        r"""Assemble the Hamiltonian from cached onsite and coupling components.

        Each component is rebuilt when its explicit argument is supplied, its
        disorder flag is enabled, or its cache is absent. With disorder disabled,
        changing self.energy or self.coupling_j alone does not refresh the cache;
        pass energy or coupling_j explicitly. Explicit overrides remain cached
        until the corresponding component is rebuilt again.

        Args:
            energy (numpy.ndarray or None): Onsite energies in Hartree, shape
                ``(M,)``. A supplied array refreshes h_diagonal; None uses
                self.energy only when the diagonal is rebuilt.
            coupling_j (numpy.ndarray or None): Couplings in Hartree, shape
                ``(n_cell, ndimer, nstate, nstate)``. A supplied array rebuilds
                the coupling matrix; None uses self.coupling_j on a rebuild.
            neighbor_index (list or None): Passed to exciton_couplings() when
                rebuilding; unused by the base builder.
            **kwargs: Forwarded to component builders only when they are called.
                Additional arguments alone do not trigger a rebuild.

        Returns:
            numpy.ndarray: Stored Hamiltonian, shape ``(M, M)``, where
                M=n_site*nstate. Its diagonal is replaced by h_diagonal.
        """
        if (energy is not None or self.onsite_disorder
                or not hasattr(self, 'h_diagonal')):
            self.h_diagonal = self.exciton_diagonal(energy, **kwargs)

        if (coupling_j is not None or self.coupling_disorder
                or not hasattr(self, 'h_coupling')):
            if coupling_j is None: coupling_j = self.coupling_j
            self.hamiltonian = self.exciton_couplings(coupling_j,
                                                      neighbor_index, **kwargs)

        np.fill_diagonal(self.hamiltonian, self.h_diagonal)
        return self.hamiltonian


    def get_initial_coefficients(self, H=None, method=None, **kwargs):
        r"""Initialize a normalized pure-state exciton wavefunction.

        ``random`` assigns random complex amplitudes to every state of the sites
        selected by n_site_init. Integer indices refer to the retained-site order;
        ``center``, ``edge0``, and ``edge1`` select the middle, first, or last site
        in lexicographic coordinate order. ``equal`` populates all basis states
        with equal amplitudes. ``ground`` selects the lowest-energy eigenvector.
        ``thermal`` draws one eigenvector with probability

        .. math::

            p_k=\frac{e^{-\beta(E_k-E_{\min})}}
                      {\sum_j e^{-\beta(E_j-E_{\min})}}.

        A thermal draw is a pure state; the thermal mixture is obtained by
        averaging independent draws.

        Args:
            H (numpy.ndarray or None): Hamiltonian in Hartree, shape ``(M, M)``;
                None calls get_hamiltonian().
            method (str or None): ``random``, ``equal``, ``ground``, or ``thermal``.
                None uses initial_method, defaulting to ``random``.
            **kwargs: Builder arguments and optional seed. The seed overrides
                random_seed for random amplitudes and thermal selection.

        Returns:
            numpy.ndarray: Initial populations ``abs(coefficients)**2``, shape
                ``(M,)``, where M=n_site*nstate. Stores the complex coefficients.
        """
        if H is None: H = self.get_hamiltonian(**kwargs)
        if method is None: method = getattr(self, 'initial_method', 'random')

        if method == 'random':
            idx = getattr(self, 'n_site_init', 0)
            rng = np.random.default_rng(kwargs.get('seed', getattr(self, 'random_seed', None)))
            if isinstance(idx, str):
                order = np.lexsort(tuple(self.length.T[::-1]))
                if idx == 'center':
                    idx = order[self.n_site//2]
                elif idx == 'edge0':
                    idx = order[0]
                elif idx == 'edge1':
                    idx = order[-1]
                idx = int(idx)
            if isinstance(idx, (int, np.integer)): idx = [idx]
            idx = np.asarray(idx, dtype=int).ravel()
            if idx.size == 0 or np.any((idx < 0) | (idx >= self.n_site)):
                raise ValueError('Initial site indices must lie in [0, n_site).')
            idx = [self.nstate*i + state for i in idx for state in range(self.nstate)]
            print('Wavefunction initializes at sites*states:', idx)
            values = rng.random(len(idx)) + 1j * rng.random(len(idx))
            values /= np.linalg.norm(values)
            coeffs0 = np.zeros(self.n_site*self.nstate, dtype=complex)
            for i, v in zip(idx, values):
                coeffs0[i] += v

        elif method == 'equal':
            coeffs0 = np.ones(self.n_site*self.nstate, dtype=complex)
            coeffs0 /= np.sqrt(coeffs0.size)

        elif method in {'ground', 'thermal'}:
            evals, evecs = np.linalg.eigh(H)

            if method == 'ground':
                coeffs0 = evecs[:,0] # ground state

            elif method == 'thermal':
                weights = np.exp(-self.beta_b * (evals - evals.min()))
                weights /= np.sum(weights)
                rng = np.random.default_rng(kwargs.get('seed', getattr(self, 'random_seed', None)))
                index = rng.choice(len(evals), p=weights)
                coeffs0 = evecs[:,index]

        else:
            raise ValueError(f'Unknown initial method: {method}')

        self.coefficients = coeffs0
        return np.abs(self.coefficients)**2 # c2


    def update_coefficients(self, dt=None, **kwargs):
        r"""Propagate coefficients using the current sampled Hamiltonian.

        For H=D+J with diag(J)=0, split modes apply Strang factors directly:

        .. math::

            c_{n+1}=e^{-ihD/2}\left[e^{-ihJ}\left(e^{-ihD/2}c_n\right)\right].

        Both diagonal factors use the same sample. The coupling exponential is
        cached until h or the assembled h_coupling changes. Parameter changes
        reach this cache only after get_hamiltonian() refreshes the component.
        Other modes apply the full Hamiltonian exponential. Names containing
        ``eigen`` use a Hermitian eigendecomposition without storing eigenpairs.

        The MC driver selects the left endpoint for ``split`` and the midpoint
        for ``magnus`` and ``magnus_split``. Magnus retains its first term with
        midpoint quadrature, omitting higher commutator terms:

        .. math::

            \Omega_1=-i\int_{t_n}^{t_n+h}H(s)\,ds\approx-ihH(t_n+h/2).

        Atomic units set hbar=1. For smooth time dependence, the midpoint
        methods have global order two and left-endpoint splitting has order
        one. Stochastic paths require separate time-step convergence checks.

        Args:
            dt (float or None): Propagation step in au; None uses exciton_dt.
                MC propagation uses a positive full step, even for midpoint sampling.
            **kwargs: Parameters passed to get_hamiltonian().

        Returns:
            numpy.ndarray: Updated populations ``abs(coefficients)**2``, shape
                ``(M,)``, where M=n_site*nstate. Updates self.coefficients.
        """
        from scipy.linalg import expm

        if dt is None: dt = self.exciton_dt

        self.get_hamiltonian(**kwargs)

        if 'split' in self.propagator:
            cache = getattr(self, '_exp_j_cache', None)
            if (cache is None or dt != cache[0]
                    or not np.array_equal(self.h_coupling, cache[1])):
                exp_j = expm(-1j * dt * self.h_coupling)
                self._exp_j_cache = (dt, self.h_coupling.copy(), exp_j)
            else:
                exp_j = cache[2]

            exp_d = np.exp(-0.5j * dt * self.h_diagonal)
            self.coefficients = exp_d * (exp_j @ (exp_d * self.coefficients))

        elif 'eigen' in self.propagator:
            evals, evecs = np.linalg.eigh(self.hamiltonian)
            phases = np.exp(-1j * dt * evals)
            self.coefficients = evecs @ (phases * (evecs.conj().T @ self.coefficients))

        else:
            self.coefficients = expm(-1j * dt * self.hamiltonian) @ self.coefficients

        return np.abs(self.coefficients)**2


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

        Args:
            dipole (numpy.ndarray, optional): Transition dipoles for each site
                with shape ``(nsite*nstate, 3)``.
            coefficients (numpy.ndarray, optional): Exciton coefficients with
                shape ``(nsite*nstate,)``.

        Returns:
            tuple: Exciton eigenvalues and oscillator strengths, each with shape
            ``(nsite*nstate,)``.
        """
        evals, evecs = np.linalg.eigh(self.hamiltonian)
        self.h_eigens = [evals, evecs] # save eigenvalues and eigenvectors

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
    r"""Exciton dynamics driven by sampled Gaussian disorder.

    Call get_sampler() once to configure the onsite and coupling channels.
    update_parameters() samples their enabled fluctuations; the inherited
    Hamiltonian and propagation methods advance the exciton state.
    """
    def get_sampler(self, variance=0.1, seed=123, **kwargs):
        r"""Configure onsite and coupling samplers and reset their fluctuations.

        The channels use independent random streams with the same variance, mean,
        and tau_c. Their samples match energy.shape and coupling_j.shape.
        A positive tau_c selects temporal correlation; zero or None selects
        independent draws. Channel flags default to their current settings.
        Spatial lengths use coordinate units; default site positions use Angstrom.

        Args:
            variance (float): Stationary variance in Hartree squared.
            seed (int or None): Seed for the two independent random streams.
            **kwargs: Channel and distribution options listed below.

        Keyword Args:
            onsite_disorder (bool): Enable onsite fluctuations.
            coupling_disorder (bool): Enable coupling fluctuations.
            sigma (float): Standard deviation in Hartree; overrides variance.
            mean (float): Mean fluctuation in Hartree; default 0.
            covariance (bool): Enable onsite spatial correlation; default False.
            coordinate (numpy.ndarray): Onsite coordinates, shape ``(energy.size, 3)``;
                defaults to site positions repeated for each state.
            ell_c (float): Positive onsite spatial correlation length.
            coupling_covariance (bool): Enable coupling spatial correlation;
                default False.
            coupling_coordinate (numpy.ndarray): Required for coupling covariance,
                shape ``(coupling_j.size, 3)``, following flattened coupling order.
            coupling_ell_c (float): Positive coupling spatial correlation length.
            kind (str): Spatial kernel, ``exp`` (default) or ``gauss``.
            nugget (float): Covariance diagonal regularization; default 1e-12.

        Returns:
            None: Sets onsite_sampler, coupling_sampler, and their sampling callables.
                Copies current parameters to energy0 and coupling_j0 and zeros delta_e
                and delta_j. Correlated noise starts from zero fluctuations rather
                than a stationary draw.
        """
        self.onsite_disorder = kwargs.get(
            'onsite_disorder', getattr(self, 'onsite_disorder', True))
        self.coupling_disorder = kwargs.get(
            'coupling_disorder', getattr(self, 'coupling_disorder', False))
        self.tau_c = getattr(self, 'tau_c', None)

        seeds = np.random.SeedSequence(seed).spawn(2)
        common = {name: kwargs[name] for name in ('sigma', 'mean', 'kind', 'nugget')
                  if name in kwargs}
        onsite_options = dict(common)
        if self.onsite_disorder and kwargs.get('covariance', False):
            coordinate = kwargs.get('coordinate')
            if coordinate is None:
                coordinate = np.repeat(self.length, self.nstate, axis=0)
            onsite_options.update(covariance=True, coordinate=coordinate,
                                  ell_c=kwargs.get('ell_c'))

        coupling_options = dict(common)
        if self.coupling_disorder and kwargs.get('coupling_covariance', False):
            coupling_options.update(covariance=True,
                                    coordinate=kwargs.get('coupling_coordinate'),
                                    ell_c=kwargs.get('coupling_ell_c'))

        self.onsite_sampler = Sampler(size=self.energy.shape, variance=variance,
                                      seed=seeds[0], **onsite_options)
        self.coupling_sampler = Sampler(size=self.coupling_j.shape, variance=variance,
                                        seed=seeds[1], **coupling_options)
        method = 'correlated_sample' if self.tau_c else 'sample'
        self.onsite_sample = getattr(self.onsite_sampler, method)
        self.coupling_sample = getattr(self.coupling_sampler, method)
        if self.tau_c:
            print('Using correlated sampling with tau_c = %.3f fs.'
                  % convert_units(self.tau_c, 'au', 'fs'))

        self.energy0 = np.copy(self.energy)
        self.coupling_j0 = np.copy(self.coupling_j)
        self.delta_e = np.zeros_like(self.energy, dtype=float)
        self.delta_j = np.zeros_like(self.coupling_j, dtype=float)


    def update_parameters(self, dt=None):
        r"""Sample enabled fluctuations and update the exciton parameters.

        .. math::

            E(t)=E_0+\delta E(t),\qquad J(t)=J_0+\delta J(t).

        Disabled channels retain their values. Call get_hamiltonian() separately
        to assemble the updated Hamiltonian.

        Args:
            dt (float or None): Elapsed time in au; None uses exciton_dt.
                Correlated samplers use dt; independent samplers ignore it.

        Returns:
            None: Updates the enabled delta_e/energy and delta_j/coupling_j arrays
                without changing their shapes.
        """
        if dt is None: dt = self.exciton_dt

        if self.onsite_disorder:
            self.delta_e = self.onsite_sample(self.delta_e, self.tau_c, dt)
            self.energy = self.energy0 + self.delta_e

        if self.coupling_disorder:
            self.delta_j = self.coupling_sample(self.delta_j, self.tau_c, dt)
            self.coupling_j = self.coupling_j0 + self.delta_j


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

        Args:
            coupling_j (numpy.ndarray, optional): Exciton-exciton couplings with shape
                ``(ndimer, nstate, nstate)``.
            neighbor_index (list, optional): Neighbor indices for the couplings.
            coupling_a (numpy.ndarray, optional): Off-diagonal exciton-phonon couplings with
                shape ``(ndimer, nmode, nstate, nstate)``.
            coordinate (numpy.ndarray, optional): Oscillator coordinates with shape
                ``(nsite, nmode)``.

        Returns:
            numpy.ndarray: Exciton Hamiltonian with shape ``(nsite*nstate, nsite*nstate)``.
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
        Args:
            key (dict): Input parameters.
            **kwargs: Extra input options.

        Notes:
            The required parameters are listed below with default values.
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
    r"""Run one exciton trajectory with Monte Carlo disorder sampling.

    Constructs an ExcitonMC step, propagates it with kernel(), and saves its
    observables. Ensemble averaging requires multiple independent trajectories.
    """
    def set_dynamics_class(self, key):
        r"""Construct the ExcitonMC step and configure its samplers.

        Args:
            key (dict): Exciton parameters and sampler options. mc_covariance enables
                onsite spatial correlation; mc_coupling_covariance enables coupling
                spatial correlation. A supplied wrapper dt overrides exciton_dt.

        Returns:
            None: Sets edstep with the requested disorder channels and samplers.
        """
        if self.dt is not None:
            key['exciton_dt'] = self.dt
        self.edstep = ExcitonMC(key)

        options = {name: key[name] for name in (
            'coordinate', 'ell_c', 'kind', 'nugget', 'sigma', 'mean',
            'coupling_coordinate', 'coupling_ell_c') if name in key}
        options.update(
            onsite_disorder=key.get('onsite_disorder', True),
            coupling_disorder=key.get('coupling_disorder', False),
            covariance=key.get('mc_covariance', False),
            coupling_covariance=key.get('mc_coupling_covariance', False))

        self.edstep.get_sampler(variance=key.get('mc_variance', 0.1),
                                seed=key.get('random_seed', None), **options)


    @monitor_performance(level=0)
    def kernel(self):
        r"""Propagate one disorder trajectory and save its observables.

        For H(t)=D(t)+J(t), let h=dt and t_m=t_n+h/2. Atomic units set hbar=1:

        .. math::

            U_n^{\mathrm{magnus}}=e^{-ihH(t_m)},

            U_n^{\mathrm{split}}\approx e^{-ihD(t_n)/2}
            e^{-ihJ(t_n)}e^{-ihD(t_n)/2},

            U_n^{\mathrm{magnus\_split}}\approx e^{-ihD(t_m)/2}
            e^{-ihJ(t_m)}e^{-ihD(t_m)/2}.

        ``split`` propagates with the left-endpoint Hamiltonian, then advances
        parameters by h. Midpoint modes advance parameters by h/2 before and after
        a full propagation step. Both diagonal half steps use the same sample.
        Each sampling call draws fresh disorder when temporal correlation is disabled.

        Magnus retains the first term with midpoint quadrature. For smooth time
        dependence, global orders are two for magnus and magnus_split and one for
        split. Stochastic paths require separate time-step convergence checks.

        Returns:
            None: Writes output_name+'.npz', including the initial time point.
                Saves site populations, state participation ratios, spatial moments
                in Angstrom squared, and endpoint energies in eV. Bath energy is
                excluded. debug=-2 additionally saves onsite energies in Hartree.
        """
        # assign local variables for class attributes
        edstep = self.edstep

        c2 = edstep.get_initial_coefficients()
        correlation, ipr = edstep.analyze_wf_property(c2=c2)

        n_site = edstep.n_site
        self.c2[0] = np.sum(c2.reshape(n_site, -1), axis=1)
        self.correlation[0] = correlation
        self.ipr[0] = ipr

        self.total_energy[0] = edstep.cal_energy()

        propagator = edstep.propagator
        if propagator not in {'magnus', 'split', 'magnus_split'}:
            raise ValueError("propagator must be 'magnus', 'split', or 'magnus_split'.")

        dt = edstep.exciton_dt
        midpoint = 'magnus' in propagator
        sample_dt = dt/2 if midpoint else 0.

        onsite_energy = None
        if self.debug == -2:
            onsite_energy = np.empty((self.nsteps, n_site*edstep.nstate))
            onsite_energy[0] = edstep.energy

        for ti in range(1, self.nsteps):
            # split uses the existing left-endpoint Hamiltonian; the other
            # methods first advance parameters for hamiltonian to the midpoint.
            if midpoint:
                edstep.update_parameters(dt=sample_dt)
            c2 = edstep.update_coefficients(dt=dt)

            # Advance parameters to the endpoint used for observables.
            edstep.update_parameters(dt=dt-sample_dt)
            edstep.get_hamiltonian() # update hamiltonian for energy analysis

            correlation, ipr = edstep.analyze_wf_property(c2=c2)

            self.c2[ti] = np.sum(c2.reshape(n_site, -1), axis=1)
            self.correlation[ti] = correlation
            self.ipr[ti] = ipr

            self.total_energy[ti] = edstep.cal_energy()

            if self.debug == -2:
                onsite_energy[ti] = edstep.energy

            if ti % self.printing_nsteps == 0:
                print('Step %4d has energy %12.6f eh' % (ti, self.total_energy[ti]))

        self.total_energy = convert_units(self.total_energy, 'eh', 'ev')
        if self.debug >= 1:
            print_matrix('total_energy (eV):', self.total_energy)
            print_matrix('correlation (AA^2):', self.correlation)

        data_dict = {'total_energy': self.total_energy,
                     'c2': self.c2,
                     'correlation': self.correlation,
                     'ipr': self.ipr,}
        if self.debug == -2:
            data_dict['onsite_energy'] = onsite_energy

        np.savez_compressed(self.output_name+'.npz', **data_dict)



def setup_exciton_dynamics(infile, key=None, nsteps=300, periodic=None):
    r"""Configure exciton dynamics from the input sections and model parameters.

    A non-None monte_carlo section selects MC dynamics, including an empty
    section that uses defaults. Otherwise, the phonon section configures
    explicit oscillator dynamics. This function constructs the driver;
    call its kernel() to propagate.

    Args:
        infile (str): Input parameter file.
        key (dict or None): Additional parameters, copied before input
            sections are merged. File-section values override matching keys.
        nsteps (int): Default propagation-step count when absent from key
            and input. The driver stores one additional initial time point.
        periodic (bool or None): Override the periodic setup flag. Periodic
            setup supplies the cells needed for Bloch extraction; the dynamics
            kernel still propagates the finite real-space Hamiltonian.

    Returns:
        ExcitonDynamicsMC or ExcitonDynamics: Configured MC or explicit
            oscillator dynamics driver.
    """
    key = {} if key is None else dict(key)
    parameters = parser(infile)
    key.update(parameters.get('exciton', {}))

    do_mc = parameters.get('monte_carlo') is not None
    if do_mc: # use Monte Carlo sampling
        key.update(parameters.get('monte_carlo', {}))
    else: # use explicit vibrational phonon modes
        key.update(parameters.get('phonon', {}))

    key['debug'] = key.get('debug', 0)
    key['random_seed'] = key.get('random_seed', None)
    key['nsteps'] = key.get('nsteps', nsteps)
    key['exciton_dt'] = key.get('exciton_dt', 10)
    key['n_cell'] = key.get('n_cell', [5, 5, 5])
    key['n_cell_param'] = key.get('n_cell_param', [5, 5, 5])
    key['nstate'] = key.get('nstate', 2)
    key['propagator'] = key.get('propagator', 'split')

    key['cif'] = key.get('cif', 'H2OBPc.cif')
    key['json'] = key.get('json', 'H2OBPc.cif.json')
    key['param_dir'] = key.get('param_dir', './')
    key['write_coords'] = key.get('write_coords', False)

    data = None
    if key.get('json', None):
        data = load_json(key['param_dir']+key['json'], to_numpy=True)
    if data is None and key.get('cif', None):
        data = process_parameters(key['cif'], key['n_cell_param'],
                                  key['param_dir'], key['nstate'])
        save_json(key['cif']+'.json', data)
    if periodic is not None:
        key['periodic'] = periodic
    cells, neighbor_index = set_model(
        data['neighbor_index'], data['distances'], key.get('model', 'AB'),
        key.get('n_cell', [5,5,5]), key.get('r_cutoff', 10),
        periodic=key.get('periodic', False))
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

    time_line = convert_units(np.arange(obj.nsteps)*obj.dt, 'au', 'fs')

    fig, ax = plt.subplots(3,1)
    ax[0].plot(time_line, obj.total_energy)
    r2 = obj.correlation[:,0] - obj.correlation[:,1]
    for x in range(3):
        ax[1].plot(time_line, r2[:,x], label='R$^2$-'+str(chr(ord('x') + x)))
    ax[1].plot(time_line, np.sum(np.abs(r2), axis=1), label='R$^2$-tot')
    ax[1].legend()
    ax[0].set_ylabel('Energy (eV)')
    ax[1].set_ylabel('<R$^2>-<R>^2$ ($\\AA^2$)')
    ax[1].set_xlabel('Time (fs)')


    save_c2_steps = [0, 10, 100, 500, 1000, 5000]
    save_c2_steps = [x for x in save_c2_steps if x < obj.nsteps]
    for i, c2 in enumerate(obj.c2[save_c2_steps]):
        i = save_c2_steps[i]
        nsites = range(len(c2))
        #nsites = np.arange(min(nsites), max(nsites)+1)
        ax[2].plot(nsites, c2[nsites], label='t$=$ %d (fs)' % convert_units(i*obj.dt, 'au', 'fs'))
    ax[2].set_ylabel('c$^2$')
    ax[2].set_xlabel('Site index')
    #ax[2].set_xticks(nsites[::10])
    ax[2].legend()
    plt.tight_layout()

    plt.show()
