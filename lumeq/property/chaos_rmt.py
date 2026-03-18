from lumeq import sys, np
from lumeq.dynamics import MolecularDynamics
from lumeq.utils import print_matrix, parser, collect_lists
from lumeq.utils.sec_mole import read_symbols_coords
from lumeq.utils.pyscf_helper import run_pyscf_final

r"""
Random matrix theory (RMT) gives Wigner-Dyson statistics for (unfolded) energy level,
showing quantum chaos: eigenvalue ensemble and eigenstate thermalization.

Reference:
    Z. Tao, https://arxiv.org/abs/2602.21299
"""

def stat_wigner(s, itype='goe'):
    r"""
    Wigner-Dyson distribution for unfolded level spacing s.
    ``P(s) = \frac{\pi}{2} s e^{-\frac{\pi}{4} s^2}`` for GOE
    ``P(s) = \frac{32}{\pi^2} s^2 e^{-\frac{4}{\pi} s^2}`` for GUE

    Args:
        s (array-like): The unfolded level spacings to be evaluated.
        itype (str, optional): The type of random matrix ensemble.
            Can be 'goe' (Gaussian Orthogonal Ensemble),
            'gue' (Gaussian Unitary Ensemble). Default is 'goe'.

    Returns:
        P (array): The probability density P(s) for the given unfolded level spacings.
    """
    if itype == 'goe':
        return 0.5 * np.pi * s * np.exp(-0.25 * np.pi * s**2)
    elif itype == 'gue':
        return 32 / np.pi**2 * s**2 * np.exp(-4 * s**2 / np.pi)


def stat_poisson(s):
    r"""
    Poisson distribution for unfolded level spacing s: P(s) = exp(-s)
    """
    return np.exp(-s)


def unfold_level_spacing(levels, deg=5, center_window=False):
    r"""
    Unfold the level spacing s = dN/dE * dE

    Args:
        levels (array-like): The energy levels to be unfolded.
        deg (int, optional): The degree of the polynomial fit to the staircase function N(E). Default is 5.
        center_window (tuple, optional): Fractional level window used for unfolding.
            For example, ``(0.2, 0.8)`` uses levels between the 20th and 80th percentiles.
            Default is False, which uses all levels.

    Returns:
        eps (array): The unfolded energy levels.
        s (array): The unfolded level spacings.
    """
    levels = np.asarray(levels)
    if levels.ndim == 2:
        eps, s = collect_lists(unfold_level_spacing, levels, deg, center_window)
        return np.asarray(eps), np.asarray(s)

    levels = np.sort(levels)
    if center_window:
        a, b = center_window
        i0, i1 = int(a*len(levels)), int(b*len(levels))
        levels = levels[i0:i1]

    # staircase counting N(E): level index as a function of energy
    N = np.arange(1, len(levels) + 1, dtype=float)

    # smooth fit to N(E)
    coeff = np.polyfit(levels, N, deg)
    eps = np.polyval(coeff, levels)

    # unfolded spacing
    d = np.diff(eps)
    s = d / d.mean(keepdims=True)
    #r = np.minimum(d[:-1], d[1:]) / np.maximum(d[:-1], d[1:])
    return eps, s


def spectral_form_factor(levels, times):
    r"""Compute the spectral form factor from unfolded energy levels.

    Args:
        levels (2d array (nsnap, nlevels)): The unfolded energy levels to be analyzed.
        times (array): The time points at which to compute K(t) for Fourier transform.

    Returns:
        K (array): The spectral form factor K(t) at the corresponding time points.

    Notes:
        The spectral form factor is
        ``K(t) = <sum_{i,j} exp(-i (E_i - E_j) t)> / D^2``.
        Equivalently,
        ``K(t) = <|sum_i exp(-i E_i t)|^2> / D^2``.
    """
    if len(levels.shape) == 1: levels = levels.reshape(1,-1)

    # np.exp works elementwise
    # sum over energy levels for each snapshot, then average over snapshots
    K = np.exp(-1j * times[:, None, None] * levels[None, :, :]).sum(axis=2)
    K = np.abs(K)**2 / levels.shape[1]**2 # (nt, nsnap)
    return K.mean(axis=1) # average over snapshots


def unfold_parameter(eps, param, ref=None):
    r"""
    Rescale an external parameter using the variance of unfolded level velocities.

    Figure 6 of Tao and Galitski, arXiv:2602.21299, uses
    ``x = sqrt(C0) * (lambda - lambda0)``,
    where ``C0 = <(d eps / d lambda)^2>``.

    Args:
        eps (2d array (nparam, nlevels)): Unfolded energy levels along the parameter grid.
        param (1d array): External parameter values, e.g. electric field or time.
        ref (float, optional): Reference parameter value ``lambda0``. Default is ``param[0]``.

    Returns:
        x (1d array): Unfolded parameter.
        c0 (float): Mean squared level velocity.
        vel (2d array): Level velocities ``d eps / d lambda``.
    """
    eps = np.asarray(eps)
    param = np.asarray(param, dtype=float)
    if eps.ndim != 2:
        raise ValueError('eps must be a 2D array of shape (nparam, nlevels)')
    if param.ndim != 1 or param.shape[0] != eps.shape[0]:
        raise ValueError('param must be a 1D array with len(param) == eps.shape[0]')

    vel = np.gradient(eps, param, axis=0, edge_order=2)
    c0 = np.mean(vel**2)
    if ref is None:
        ref = param[0]
    x = np.sqrt(c0) * (param - ref)
    return x, c0, vel


if __name__ == '__main__':
    from lumeq.plot import plt
    from lumeq.utils import convert_units
    from matplotlib.ticker import MaxNLocator

    jobtype = sys.argv[2]
    if jobtype == 'static':
        infile = sys.argv[1]
        parameters = parser(infile)
        results = run_pyscf_final(parameters)
        mf = results['mf']
        mo_energy = mf.mo_energy
        #print_matrix('mo_energy', mo_energy)

        occidx = np.where(mf.mo_occ > 0)[0]
        viridx = ~occidx
        energy = mo_energy[viridx] - mo_energy[occidx, None]
        energy = energy.ravel()

        td_e = results['td'].e

        label = ['MO energy', '1e excitation', 'excitation energy']
        _, s0 = unfold_level_spacing(mo_energy, deg=5, center_window=(0.2, 0.8))
        _, s1 = unfold_level_spacing(energy, deg=5, center_window=(0.2, 0.8))
        _, s2 = unfold_level_spacing(td_e, deg=5, center_window=(0.2, 0.8))

        np.savez('level_spacing.npz', mo_energy=mo_energy, energy=energy, td_e=td_e,
                 s0=s0, s1=s1, s2=s2)

        fig, axes = plt.subplots(1, 3, tight_layout=True)
        for i, s in enumerate([s0, s1, s2]):
            ax = axes[i]
            ax.hist(s, bins=20, density=True, alpha=0.5, label=label[i], facecolor='C0')

            x = np.linspace(0, max(4, np.max(s)), 400)
            ax.plot(x, stat_wigner(x, 'goe'), label='GOE')
            ax.plot(x, stat_poisson(x), '--', label='Poisson')

            ax.set_xlabel('Unfolded spacing s')
            if i==0: ax.set_ylabel('P(s)')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend()

        plt.show()

    elif jobtype == 'md':
        dt = 10
        data = np.load('mo_energy.npz')
        mo_energy = data['mo_energy']
        unfolded_e = np.asarray([unfold_level_spacing(e, deg=5, center_window=(0.2, 0.8))[0] for e in mo_energy])
        times = np.linspace(0.0, 2.0, 5000)
        k = spectral_form_factor(unfolded_e[0], times)

        fig, ax = plt.subplots(tight_layout=True)
        ax.plot(times, k)
        ax.set_xlabel('Spectral time t')
        ax.set_ylabel('K(t)')
        plt.show()

    elif jobtype == 'md_flow':
        data = np.load(sys.argv[1])
        levels_key = sys.argv[3] if len(sys.argv) > 3 else 'mo_energy'
        field_key = sys.argv[4] if len(sys.argv) > 4 else 'field'
        ref = float(sys.argv[5]) if len(sys.argv) > 5 else None

        levels = data[levels_key]
        if field_key in data:
            field = data[field_key]
        else:
            dt = 10
            field = np.linspace(0.0, levels.shape[0]*dt, levels.shape[0])
        eps, _ = unfold_level_spacing(levels, deg=5, center_window=(0.2, 0.8))
        x, c0, _ = unfold_parameter(eps, field, ref=ref)
        max_levels = min(80, eps.shape[1])
        i0 = (eps.shape[1] - max_levels) // 2
        i1 = i0 + max_levels

        fig, ax = plt.subplots(tight_layout=True)
        ax.plot(x[:, None], eps[:, i0:i1], color='C0', alpha=0.6, lw=0.8)
        ax.set_xlabel(r'Unfolded parameter $x$')
        ax.set_ylabel(r'Unfolded energy $\epsilon_I$')
        ax.set_title(rf'Field-driven level flow, $C_0={c0:.3e}$')
        plt.show()
