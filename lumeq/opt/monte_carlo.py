from lumeq import sys, np
from lumeq.utils.print_matrix import print_statistics

def log_prob(action):
    r"""
    Abstract function to compute the logarithmic probability density
    p(x) = e^{-S(x)} such that log p(x) = -S(x)
    """
    return -action


def metropolis(log_prob, x, step_size=1.0, n_steps=5000, seed=None):
    r"""
    Metropolis sampler generates new points
    based on Gaussian proposals (random walk)
    provided a target distribution
    with the acceptance ratio: A = min(1, p(x')/p(x))

    Args:
        log_prob (callable): Logarithmic probability density function.
        x: Initial position.
        step_size (float): Step size for the proposal distribution.
        n_steps (int): Number of steps to sample.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        numpy.ndarray: Sampled positions.
    """
    rng = np.random.default_rng(seed)

    action = log_prob(x)

    samples = []
    for _ in range(n_steps):
        x_prop = x + rng.normal(scale=step_size)
        action_new = log_prob(x_prop)
        if np.log(rng.random()) < action_new - action:
            x = x_prop
            action = action_new

        samples.append(x)

    return np.array(samples)


def metropolis_pi(log_prob, x, step_size=0.5, n_steps=5000,
                  seed=None, **kwargs):
    r"""
    Metropolis walker for path integral method generates new points
    based on Gaussian proposals (random walk)
    provided a target distribution
    with the acceptance ratio: A = min(1, p(x')/p(x))

    Args:
        log_prob (callable): Logarithmic probability density function.
        x: Initial bead positions.
        step_size (float): Step size for bead updates.
        n_steps (int): Number of steps to sample.
        seed (int, optional): Random seed for reproducibility.
        **kwargs: Additional keyword arguments passed to ``log_prob``.

    Returns:
        numpy.ndarray: Sampled bead positions.
    """
    rng = np.random.default_rng(seed)

    n_beads = len(x) # number of beads
    action = log_prob(x, **kwargs)

    samples = []
    for _ in range(n_steps):
        i = rng.integers(0, n_beads) # choose a bead to update
        old = x[i]
        new = old + rng.normal(scale=step_size)
        x[i] = new
        action_new = log_prob(x, **kwargs)
        if np.log(rng.random()) < action_new - action:
            action = action_new
        else: # reject move
            x[i] = old

        samples.append(np.copy(x))

    return np.array(samples)


def importance_sampling(log_prob, proposal_log_prob):
    r"""
    Importance sampling generates samples from a target distribution p(x)
    using a proposal distribution q(x) that is easier.
    <f> = \int f(x) p(x) dx = \int f(x) (p(x)/q(x)) q(x) dx = <f(x) w(x)>
    where w(x) = p(x)/q(x) are the importance weights.

    Args:
        log_prob (callable): Logarithmic target probability density function.
        proposal_log_prob (callable): Logarithmic proposal density function.

    Returns:
        numpy.ndarray: Importance weights.
    """
    log_weights = log_prob(samples) - proposal_log_prob(samples)
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= np.sum(weights)

    return weights


def diffusion_walker(potential, n_walkers=1000, dt=0.01, n_steps=500,
                     seed=None):
    r"""Run a simple diffusion Monte Carlo walker simulation.

    The algorithm alternates drift-diffusion proposals, branching weights,
    and walker resampling with a reference-energy update.

    Args:
        potential (callable): Potential-energy function.
        n_walkers (int): Number of walkers.
        dt (float): Time step.
        n_steps (int): Number of diffusion steps.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: Final walker positions and estimated energies.

    Notes:
        The diffusion Monte Carlo propagation follows
        ``- d psi / dt = - (T + V - E_ref) psi``.

        The update is split into three stages:

        1. drift-diffusion move from the kinetic term:
           ``x' = x + chi * sqrt(dt)``,
        2. branching weight from the potential term:
           ``w = exp(-(V(x') - E_ref) * dt)``,
        3. resampling walkers according to the branching weights.

        States with local energy closer to ``E_ref`` are therefore favored.
    """
    sqrt_dt = np.sqrt(dt)
    rng = np.random.default_rng(seed)

    # Initialize walkers
    walkers = rng.normal(scale=1.0, size=n_walkers)
    # Initial reference energy
    ref_energy = np.mean(potential(walkers))

    energy_list = []
    for i in range(n_steps):
        # Drift-diffusion move
        walkers += rng.normal(scale=sqrt_dt, size=n_walkers)

        # Branching weight
        energy = potential(walkers)
        weights = np.exp(dt * (ref_energy - energy))

        # Resample walkers based on weights
        probs = weights / np.max(weights)
        survivors = walkers[rng.random(n_walkers) < probs]
        if len(survivors) == 0: # repopulate if all walkers die
            survivors = rng.normal(size=n_walkers)

        walkers = rng.choice(survivors, size=n_walkers, replace=True)

        # Update reference energy
        ref_energy = np.mean(energy) - np.log(weights.mean()) / dt
        energy_list.append(ref_energy)

    return walkers, energy_list


class toy_qm():
    r"""
    A trivial quantum mechanics example using a Gaussian wavefunction
    """
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def psi(self, x):
        return np.exp(-self.alpha * x**2)

    def log_psi(self, x):
        return -self.alpha * x**2

    def log_prob(self, x):
        return -2.*self.alpha * x**2

    def kinetic_energy(self, x):
        r"""-1/2 * (d^2 \psi / dx^2) / psi"""
        alpha = self.alpha
        self.kinetic = alpha - 2. * alpha**2 * x**2
        return self.kinetic

    def potential_energy(self, x):
        self.potential = 0.5 * x**2 #+ x**4
        return self.potential

    def local_energy(self, x):
        self.energy = self.kinetic_energy(x) + self.potential_energy(x)
        return self.energy


    def action(self, x, beta=1.0, M=100):
        r"""
        Action S(x) = \int m/2 * \dot(x)^2 + V(x)
                    = \sum m/(2*dt) * (x(t+dt) - x(t))^2 + V(x) dt

        Args:
            x (numpy.ndarray): Discretized points on the integration path.
            beta (float): Inverse temperature.
            M (int): Number of beads (Trotter slices).
        """
        dt = beta / M
        kinetic = np.sum((x - np.roll(x, 1))**2) / (2.*dt)
        potential = np.sum(self.potential_energy(x)) * dt
        return kinetic + potential



def run_vmc(model, step_size=1.0, n_steps=10000, n_burn=1000):
    r"""
    Run a Variational Monte Carlo simulation

    Args:
        model: Quantum model with ``log_prob`` and ``local_energy`` methods.
        step_size (float): Step size for Metropolis sampling.
        n_steps (int): Number of Metropolis steps.
        n_burn (int): Number of initial samples to discard.

    Returns:
        tuple: Sampled points and estimated energies.
    """
    # Metropolis sampling
    x0 = 0.0
    samples = metropolis(model.log_prob, x0, step_size, n_steps)
    samples = samples[n_burn:]

    # Compute local energies
    energy = model.local_energy(samples)

    print_statistics('VMC Estimated Energy', energy)

    return samples, energy


def run_pimc(model, beta=1.0, M=100, step_size=1.0, n_steps=10000, n_burn=1000):
    r"""
    Run a Path Integral Monte Carlo simulation

    Args:
        model: Quantum model with ``log_prob`` and ``local_energy`` methods.
        beta (float): Inverse temperature.
        M (int): Number of beads on the integration path.
        step_size (float): Step size for Metropolis sampling.
        n_steps (int): Number of Metropolis steps.
        n_burn (int): Number of initial samples to discard.

    Returns:
        tuple: Sampled paths and estimated energies.
    """
    # Metropolis sampling
    x0 = np.zeros(M)
    samples = metropolis_pi(model.action, x0, step_size, n_steps, beta=beta, M=M)
    samples = samples[n_burn:]

    # Compute local energies
    energy = None
    raise NotImplementedError('path-integral energy is under development')

    return samples, energy


def run_dmc(model, n_walkers=3000, dt=0.01, n_steps=7000, n_burn=2000):
    r"""
    run a Diffusion Monte Carlo simulation
    """

    potential = model.potential_energy

    walkers, energy = diffusion_walker(potential, n_walkers=n_walkers, dt=dt, n_steps=n_steps)
    walkers, energy = walkers[n_burn:], energy[n_burn:]

    print_statistics('DMC Estimated Energy', energy)
    return walkers, energy



if __name__ == "__main__":
    alpha = 0.5
    model = toy_qm(alpha)
    run_vmc(model)
    run_dmc(model)
    run_pimc(model)
