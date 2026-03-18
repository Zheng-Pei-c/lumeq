from lumeq import np, itertools
from lumeq.utils import monitor_performance
from lumeq.plot import plt

from lumeq.spins import cirq, sympy
from lumeq.spins import create_qubits_2d, create_circuit

class QAOA(object):
    r"""Base class for Quantum Approximate Optimization Algorithm models.

    Notes:
        The variational state has the form
        ``prod_k U(beta_k, B) U(gamma_k, C) |s>``,
        where ``|s>`` is the equal-superposition initial state.

        ``U(gamma, C) = exp(-i * gamma * C)`` is the problem-Hamiltonian
        evolution and ``U(beta, B) = exp(-i * beta * B)`` is the mixer
        evolution.
    """
    def __init__(self, nsite, **kwargs):
        self.nsite = nsite
        self.nrow, self.ncol = nsite

        # initialize qubits on 2d array
        self.qubits = create_qubits_2d(self.nrow, self.ncol)

        # define circuit parameters
        kwargs = self.resolve_circuit_params(**kwargs)

        # initialize circuit
        self.circuit = create_circuit(self.qubits, gamma=self.gamma,
                                      beta=self.beta, nlayers=self.nlayers,
                                      **kwargs)


    def resolve_circuit_params(self, **kwargs):
        r"""
        Define the circuit parameters.

        Args:
            **kwargs: Circuit parameters such as ``gamma``, ``beta``, and ``nlayers``.
        """
        self.gamma = None
        self.beta = None
        self.nlayers = 0

        return kwargs


    @property
    def draw_circuit(self):
        r"""Draw the quantum circuit."""
        print('circuit:\n', self.circuit)
        #cirq.testing.assert_has_diagram(self.circuit)


    def energy_expectation(self, wf, **kwargs):
        r"""
        Calculate the energy expectation value for a given wavefunction.

        Args:
            wf (numpy.ndarray): Wavefunction array of size ``2**(nrow*ncol)``.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError


    @monitor_performance
    def energy_evaluation(self, gamma, beta):
        r"""
        Evaluate the energy expectation value for given gamma and beta on circuit.

        Args:
            gamma (float): Rotation angle of the target Hamiltonian.
            beta (float): Rotation angle of the initial Hamiltonian.

        Returns:
            float: Energy expectation value of the simulated wavefunction.
        """
        # start simulate
        simulator = cirq.Simulator()
        params = cirq.ParamResolver({self.gamma: gamma, self.beta: beta})
        result = simulator.simulate(self.circuit, param_resolver=params)
        wf = result.final_state_vector

        # calculate energy expectation value
        return self.energy_expectation(wf)


    def fd_gradient_evaluation(self, gamma, beta, eps=10**-3):
        # gamma gradient
        grad_g  = self.energy_evaluation(gamma+eps, beta)
        grad_g -= self.energy_evaluation(gamma-eps, beta)
        grad_g /= 2*eps

        # beta gradient
        grad_b  = self.energy_evaluation(gamma, beta+eps)
        grad_b -= self.energy_evaluation(gamma, beta-eps)
        grad_b /= 2*eps

        return grad_g, grad_b


    def optimizer(self, gamma, beta, eps=10**-3, max_steps=150, thresh=10**-5):
        r"""
        Optimize rotation angles

        Args:
            gamma (float): Rotation angle for the target Hamiltonian.
            beta (float): Rotation angle for the initial Hamiltonian.
            eps (float): Finite-difference step size.
            max_steps (int): Maximum number of optimization steps.
            thresh (float): Convergence threshold for the energy.

        Returns:
            tuple: Optimized ``(gamma, beta, energy)``.
        """
        energy = self.energy_evaluation(gamma, beta)
        for i in range(max_steps):
            grad_g, grad_b = self.fd_gradient_evaluation(gamma, beta, eps)

            # update circuit parameters
            gamma -= eps * grad_g
            beta  -= eps * grad_b

            e = self.energy_evaluation(gamma, beta)
            print('step: %3d, gamma: %4.2f, beta: %4.2f, grad_g: %5.3f, grad_b: %5.3f, energy: %8.5f'
                  % (i, gamma, beta, grad_g, grad_b, energy))

            if np.abs(energy - e) < thresh:
                break
            energy = e

        self.energy = energy
        return gamma, beta, energy



class QAOA_Ising(QAOA):
    r"""
    `E = - \sum_{ij} \sigma_i \sigma_j - h_i \sigma_i`
    h is the magnetic field on each spin
    """
    def resolve_circuit_params(self, **kwargs):
        r"""Define the circuit parameters for Ising model."""
        self.gamma = sympy.Symbol('gamma')
        self.beta = sympy.Symbol('beta')
        self.nlayers = kwargs.get('nlayers', 1)

        # magnetic field on each site
        h_field = kwargs.get('h_field', None)
        if h_field is None:
            h_field = .5*np.ones(self.nsite)
            kwargs['h_field'] = h_field # pass to circuit creation

        # j couplings
        j_coupling = kwargs.get('j_coupling', None)
        if j_coupling is None:
            j_coupling = np.ones(self.nsite)
            kwargs['j_coupling'] = j_coupling

        self.h_field = h_field
        self.j_coupling = j_coupling

        return kwargs


    def energy_expectation(self, wf, **kwargs):
        r"""
        Calculate the energy expectation value for a given wavefunction.

        Args:
            wf (numpy.ndarray): Wavefunction array of size ``2**(nrow*ncol)``.
            **kwargs: Additional keyword arguments.
        """
        nrow, ncol = self.nrow, self.ncol
        nsite = nrow * ncol

        # pauli-z operator diag(1, -1) introduces a phase flip
        # onsite Z has the shape (nsite, 2**nrow*ncol)
        Z = np.array([(-1) ** (np.arange(2**nsite) >> i) for i in range(nsite-1, -1, -1)])

        # neighboring coupling
        ZZ = np.zeros_like(wf)
        for r, c, in itertools.product(range(nrow), range(ncol)):
            if r < nrow - 1:
                ZZ += Z[r*ncol+c] * Z[(r+1)*ncol+c]
            if c < ncol - 1:
                ZZ += Z[r*ncol+c] * Z[r*ncol+(c+1)]

        energy = - ZZ - np.einsum('i,ij->j', self.h_field.ravel(), Z)

        # expectation value of the energy per site
        return np.einsum('i,i,i->', wf.conj(), wf, energy).real / nsite



if __name__ == '__main__':
    from lumeq.utils import set_performance_log
    set_performance_log(debug=True)

    nsite = [3,2]
    ising = QAOA_Ising(nsite)
    #ising.draw_circuit

    gamma, beta = -.21, -.28
    ising.optimizer(gamma, beta)
