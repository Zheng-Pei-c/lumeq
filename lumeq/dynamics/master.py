r"""Master-equation utilities for single-exciton transport models.

This module provides a small set of helpers for evolving density matrices
under unitary and Lindblad dynamics.  It is intended as a minimal extension to
wavefunction-based exciton dynamics workflows where one wants to include
dephasing and thermal relaxation in a controlled way.

The functions here are basis agnostic.  For a site-basis exciton Hamiltonian

.. math::

    H = \sum_n \epsilon_n |n\rangle\langle n| + \sum_{n \ne m}
        J_{nm} |n\rangle\langle m|,

the density matrix :math:`\rho` evolves as

.. math::

    \frac{d\rho}{dt} = -i[H, \rho] + \mathcal{D}(\rho),

where :math:`\mathcal{D}` is a Lindblad dissipator built from the jump
operators returned by ``make_site_dephasing_jumps`` or
``make_thermal_jumps``.
"""

from __future__ import annotations

import numpy as np


def density_from_state(state: np.ndarray) -> np.ndarray:
    """Return a pure-state density matrix.

    Args:
        state: Complex state vector with shape ``(n,)``.

    Returns:
        Density matrix ``|psi><psi|`` with shape ``(n, n)``.
    """

    state = np.asarray(state, dtype=complex)
    return np.outer(state, state.conj())


def normalize_density(rho: np.ndarray) -> np.ndarray:
    """Hermitize and normalize a density matrix.

    Numerical integration can introduce small Hermiticity and trace errors.
    This helper projects the matrix back to the Hermitian, unit-trace manifold.

    Args:
        rho: Density matrix with shape ``(n, n)``.

    Returns:
        Hermitian density matrix with unit trace.
    """

    rho = 0.5 * (rho + rho.conj().T)
    trace = np.trace(rho).real
    if np.isclose(trace, 0.0):
        raise ValueError("Density matrix has zero trace and cannot be normalized.")
    return rho / trace


def lindblad_rhs(rho: np.ndarray, hamiltonian: np.ndarray, jumps) -> np.ndarray:
    """Evaluate the Lindblad master-equation right-hand side.

    Args:
        rho: Density matrix with shape ``(n, n)``.
        hamiltonian: System Hamiltonian with shape ``(n, n)``.
        jumps: Iterable of ``(operator, rate)`` pairs.  Each operator has shape
            ``(n, n)`` and each rate is a non-negative scalar.

    Returns:
        Time derivative ``d rho / dt``.
    """

    rho = np.asarray(rho, dtype=complex)
    hamiltonian = np.asarray(hamiltonian, dtype=complex)

    drho = -1j * (hamiltonian @ rho - rho @ hamiltonian)
    for operator, rate in jumps:
        if rate == 0:
            continue
        operator = np.asarray(operator, dtype=complex)
        left = operator.conj().T @ operator
        drho += rate * (
            operator @ rho @ operator.conj().T
            - 0.5 * (left @ rho + rho @ left)
        )
    return drho


def rk4_step(rho: np.ndarray, hamiltonian: np.ndarray, jumps, dt: float) -> np.ndarray:
    """Advance a density matrix by one fourth-order Runge-Kutta step.

    Args:
        rho: Density matrix with shape ``(n, n)``.
        hamiltonian: System Hamiltonian with shape ``(n, n)``.
        jumps: Iterable of ``(operator, rate)`` pairs.
        dt: Time step.

    Returns:
        Updated density matrix after one RK4 step.
    """

    k1 = lindblad_rhs(rho, hamiltonian, jumps)
    k2 = lindblad_rhs(rho + 0.5 * dt * k1, hamiltonian, jumps)
    k3 = lindblad_rhs(rho + 0.5 * dt * k2, hamiltonian, jumps)
    k4 = lindblad_rhs(rho + dt * k3, hamiltonian, jumps)
    rho_next = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return normalize_density(rho_next)


def propagate_density_matrix(
    rho0: np.ndarray,
    hamiltonian: np.ndarray,
    jumps,
    dt: float,
    nsteps: int,
) -> np.ndarray:
    """Propagate a density matrix trajectory.

    Args:
        rho0: Initial density matrix with shape ``(n, n)``.
        hamiltonian: System Hamiltonian with shape ``(n, n)``.
        jumps: Iterable of ``(operator, rate)`` pairs.
        dt: Time step.
        nsteps: Number of stored steps, including the initial state.

    Returns:
        Array of density matrices with shape ``(nsteps, n, n)``.
    """

    if nsteps < 1:
        raise ValueError("nsteps must be at least 1.")

    rho0 = normalize_density(np.asarray(rho0, dtype=complex))
    dimension = rho0.shape[0]
    trajectory = np.empty((nsteps, dimension, dimension), dtype=complex)
    trajectory[0] = rho0

    for step in range(1, nsteps):
        trajectory[step] = rk4_step(trajectory[step - 1], hamiltonian, jumps, dt)

    return trajectory


def make_site_dephasing_jumps(nsite: int, gamma) -> list[tuple[np.ndarray, float]]:
    r"""Build site-basis pure-dephasing jump operators.

    This is the standard Lindblad form corresponding to

    .. math::

        L_n = |n\rangle\langle n|.

    It damps off-diagonal coherences while leaving site populations unchanged
    directly.

    Args:
        nsite: Hilbert-space dimension.
        gamma: Scalar or array-like dephasing rates.

    Returns:
        List of ``(L_n, gamma_n)`` pairs.
    """

    rates = np.broadcast_to(np.asarray(gamma, dtype=float), (nsite,))
    jumps = []
    for index, rate in enumerate(rates):
        operator = np.zeros((nsite, nsite), dtype=complex)
        operator[index, index] = 1.0
        jumps.append((operator, float(rate)))
    return jumps


def make_thermal_jumps(
    hamiltonian: np.ndarray,
    beta: float,
    rate_scale: float = 1.0e-3,
    spectral_density=None,
) -> list[tuple[np.ndarray, float]]:
    r"""Build thermal relaxation jumps in the Hamiltonian eigenbasis.

    The jump operator for a transition ``alpha -> beta`` is

    .. math::

        L_{\beta\alpha} = |\beta\rangle\langle\alpha|,

    with rates chosen to satisfy detailed balance.  By default, downhill rates
    are set by ``rate_scale`` and uphill rates are Boltzmann suppressed.

    Args:
        hamiltonian: System Hamiltonian with shape ``(n, n)``.
        beta: Inverse temperature ``1 / (k_B T)`` in units consistent with
            ``hamiltonian``.
        rate_scale: Base relaxation rate for downhill transitions.
        spectral_density: Optional callable ``f(abs_delta_e))`` that modulates
            the transition rate by energy gap.

    Returns:
        List of ``(L, rate)`` pairs for thermal relaxation.
    """

    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    evals, evecs = np.linalg.eigh(hamiltonian)
    if spectral_density is None:
        spectral_density = lambda delta_e: 1.0

    jumps = []
    for alpha, energy_alpha in enumerate(evals):
        for beta_index, energy_beta in enumerate(evals):
            if alpha == beta_index:
                continue

            delta_e = energy_beta - energy_alpha
            weight = float(spectral_density(abs(delta_e)))
            if delta_e < 0:
                rate = rate_scale * weight
            else:
                rate = rate_scale * weight * np.exp(-beta * delta_e)

            operator = np.outer(evecs[:, beta_index], evecs[:, alpha].conj())
            jumps.append((operator, float(rate)))
    return jumps


def populations_from_density(rho: np.ndarray) -> np.ndarray:
    """Return populations from a density matrix.

    Args:
        rho: Density matrix with shape ``(n, n)``.

    Returns:
        Real populations given by the diagonal of ``rho``.
    """

    return np.real(np.diag(rho))


def position_correlation(length: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Compute ``<r^2> - <r>^2`` from site populations.

    Args:
        length: Site coordinates with shape ``(nsite, ndim)``.
        rho: Density matrix with shape ``(nsite, nsite)``.

    Returns:
        Array with shape ``(2, ndim)`` storing ``<r^2>`` and ``<r>^2``.
    """

    pop = populations_from_density(rho)
    r2 = np.einsum("nx,nx,n->x", length, length, pop)
    r_2 = np.einsum("nx,n->x", length, pop) ** 2
    return np.array([r2, r_2], dtype=float)
