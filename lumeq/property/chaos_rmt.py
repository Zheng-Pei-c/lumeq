from lumeq import sys, np
from lumeq.dynamics import MolecularDynamics
from lumeq.utils import print_matrix, parser
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


def _select_level_window(levels, center_window=False):
    r"""
    Select and sort levels in a fractional center window.
    """
    levels = np.sort(np.asarray(levels))
    if center_window:
        a, b = center_window
        i0, i1 = int(a*len(levels)), int(b*len(levels))
        levels = levels[i0:i1]
    return levels


def _unfold_levels(levels, deg=5):
    r"""
    Smoothly unfold sorted levels with a polynomial staircase fit.
    """
    levels = np.asarray(levels)
    if len(levels) == 0:
        return levels.copy()

    # staircase counting N(E): level index as a function of energy
    N = np.arange(1, len(levels) + 1, dtype=float)
    fit_deg = min(deg, len(levels) - 1)
    coeff = np.polyfit(levels, N, fit_deg)
    return np.polyval(coeff, levels)


def _level_spacing_from_levels(levels):
    r"""
    Calculate mean-normalized spacings from sorted or unfolded levels.
    """
    d = np.diff(levels)
    if d.size == 0:
        return d

    mean = d.mean()
    if np.abs(mean) < 1e-14:
        return d
    return d / mean


def _adjacent_gap_ratio_from_levels(levels):
    r"""
    Calculate adjacent-gap ratios from sorted or unfolded levels.
    """
    d = np.diff(np.sort(levels))
    if len(d) < 2:
        return np.asarray([]), np.nan

    # Keep exact or near degeneracies from creating divide-by-zero warnings.
    mask = (np.abs(d[:-1]) > 1e-14) & (np.abs(d[1:]) > 1e-14)
    if not np.any(mask):
        return np.asarray([]), np.nan

    d0, d1 = d[:-1][mask], d[1:][mask]
    r = np.minimum(d0, d1) / np.maximum(d0, d1)
    return r, np.mean(r)


def _stack_stat_values(values):
    r"""
    Convert a list of per-spectrum outputs to an array when possible.
    """
    try:
        return np.asarray(values)
    except ValueError:
        return np.asarray(values, dtype=object)


def eigenvalue_statistics(levels, deg=5, center_window=False, unfold=True,
                          level_spacing=True, adjacent_gap_ratio=True):
    r"""
    Calculate selected eigenvalue statistics with shared preprocessing.

    The input levels are sorted and optionally filtered by ``center_window``.
    If ``unfold=True``, the statistics are evaluated on polynomial-unfolded
    levels; otherwise they are evaluated on the raw selected levels.  The
    level spacing is mean-normalized, while the adjacent-gap ratio is
    ``r_i = min(d_i, d_{i+1}) / max(d_i, d_{i+1})``.

    Args:
        levels (array-like): One spectrum ``(nlevels,)`` or snapshots
            ``(nsnap, nlevels)``.
        deg (int, optional): Polynomial degree used for unfolding.
        center_window (tuple, optional): Fractional level window.
        unfold (bool, optional): Whether to unfold levels before statistics.
        level_spacing (bool, optional): Whether to calculate spacing ``s``.
        adjacent_gap_ratio (bool, optional): Whether to calculate the
            adjacent-gap ratio.

    Returns:
        data (dict): Dictionary with selected raw ``levels``, analysis
        levels ``eps``, and optional ``s``, ``r``, and ``r_mean`` fields.
    """
    levels = np.asarray(levels)
    if levels.ndim == 2:
        rows = [eigenvalue_statistics(x, deg=deg, center_window=center_window,
                                      unfold=unfold,
                                      level_spacing=level_spacing,
                                      adjacent_gap_ratio=adjacent_gap_ratio)
                for x in levels]
        if len(rows) == 0:
            data = {'levels': np.asarray([]), 'eps': np.asarray([])}
            if level_spacing:
                data['s'] = np.asarray([])
            if adjacent_gap_ratio:
                data['r'] = np.asarray([])
                data['r_mean'] = np.asarray([])
            return data

        return {key: _stack_stat_values([row[key] for row in rows])
                for key in rows[0]}

    levels = _select_level_window(levels, center_window)
    eps = _unfold_levels(levels, deg=deg) if unfold else levels
    data = {'levels': levels, 'eps': eps}

    if level_spacing:
        data['s'] = _level_spacing_from_levels(eps)

    if adjacent_gap_ratio:
        data['r'], data['r_mean'] = _adjacent_gap_ratio_from_levels(eps)

    return data


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
    data = eigenvalue_statistics(levels, deg=deg, center_window=center_window,
                                 unfold=True, level_spacing=True,
                                 adjacent_gap_ratio=False)
    return data['eps'], data['s']


def adjacent_gap_ratio(levels, deg=5, center_window=False, unfold=False):
    r"""
    Calculate adjacent-gap ratios from sorted levels.

    The adjacent-gap ratio is
    ``r_i = min(d_i, d_{i+1}) / max(d_i, d_{i+1})``.
    It is usually computed from raw sorted levels because the local density
    of states cancels in the ratio.  Set ``unfold=True`` only when one wants
    to use the same unfolded levels as in ``unfold_level_spacing``.

    Reference:
        Y. Y. Atas, E. Bogomolny, O. Giraud, and G. Roux,
        Phys. Rev. Lett. 110, 084101 (2013),
        https://doi.org/10.1103/PhysRevLett.110.084101

    Args:
        levels (array-like): Energy levels.
        deg (int, optional): Polynomial degree used when ``unfold=True``.
        center_window (tuple, optional): Fractional level window.
        unfold (bool, optional): Whether to compute ratios from unfolded levels.

    Returns:
        r (array): Adjacent-gap ratios.
        r_mean (float): Mean adjacent-gap ratio.
    """
    data = eigenvalue_statistics(levels, deg=deg, center_window=center_window,
                                 unfold=unfold, level_spacing=False,
                                 adjacent_gap_ratio=True)
    return data['r'], data['r_mean']


def site_population(vectors, nstate=1):
    r"""
    Calculate site populations for each eigenstate.

    Args:
        vectors (2d array (nbasis, nlevels)): Eigenvectors in columns.
        nstate (int, optional): Number of local excited states on each site.

    Returns:
        pop (2d array (nsite, nlevels)): Population of each eigenstate on each site.
    """
    vectors = np.asarray(vectors)
    if vectors.ndim == 3:
        pop = [site_population(v, nstate) for v in vectors]
        return np.asarray(pop)

    if vectors.ndim != 2:
        raise ValueError('vectors must be a 2D array of shape (nbasis, nlevels)')
    if vectors.shape[0] % nstate != 0:
        raise ValueError('nbasis must be divisible by nstate')

    nsite = vectors.shape[0] // nstate
    pop = np.abs(vectors.reshape(nsite, nstate, -1))**2
    return pop.sum(axis=1)


def eigenstate_ipr(vectors, nstate=1):
    r"""
    Calculate inverse participation ratio and participation ratio.

    Args:
        vectors (2d array (nbasis, nlevels)): Eigenvectors in columns.
        nstate (int, optional): Number of local excited states on each site.

    Returns:
        ipr (array): ``sum_n |psi_a(n)|^4`` for each eigenstate.
        pr (array): Participation ratio ``1 / ipr``.
        entropy (array): Shannon entropy of the site population.
    """
    pop = site_population(vectors, nstate=nstate)
    if pop.ndim == 3:
        out = [eigenstate_ipr(v, nstate=nstate) for v in vectors]
        ipr, pr, entropy = zip(*out)
        return np.asarray(ipr), np.asarray(pr), np.asarray(entropy)

    ipr = np.sum(pop**2, axis=0)
    pr = 1. / ipr
    safe_pop = np.where(pop > 0., pop, 1.)
    entropy = -np.sum(np.where(pop > 0., pop*np.log(safe_pop), 0.), axis=0)
    return ipr, pr, entropy


def local_projector_matrix(vectors, site=0, nstate=1):
    r"""
    Matrix elements of local site projector P_n in the energy eigenbasis.

    ``P_ab = <psi_a|P_n|psi_b>``.  For ``nstate > 1``, the site projector is
    the sum over the local excited states on that site.

    Args:
        vectors (2d array (nbasis, nlevels)): Eigenvectors in columns.
        site (int): Site index.
        nstate (int, optional): Number of local excited states on each site.

    Returns:
        P (2d array (nlevels, nlevels)): Projector matrix in eigenbasis.
    """
    vectors = np.asarray(vectors)
    if vectors.ndim != 2:
        raise ValueError('vectors must be a 2D array of shape (nbasis, nlevels)')

    i0, i1 = site*nstate, (site+1)*nstate
    v = vectors[i0:i1]
    return v.conj().T @ v


def eth_projector_statistics(levels, vectors, site=0, nstate=1, nbins=20,
                             center_window=False):
    r"""
    ETH diagnostics for a local site projector.

    The local projector is ``P_n = |n><n|`` for ``nstate=1`` and the sum over
    local states for ``nstate > 1``.  This routine bins the diagonal ETH
    component ``P_aa`` by energy and the off-diagonal envelope ``|P_ab|^2`` by
    ``|omega| = |E_a - E_b|``.

    Reference:
        L. D'Alessio, Y. Kafri, A. Polkovnikov, and M. Rigol,
        Adv. Phys. 65, 239 (2016),
        https://doi.org/10.1080/00018732.2016.1198134

    Args:
        levels (array-like): Energy eigenvalues.
        vectors (2d array (nbasis, nlevels)): Eigenvectors in columns.
        site (int): Site index for the local projector.
        nstate (int, optional): Number of local excited states on each site.
        nbins (int, optional): Number of energy/frequency bins.
        center_window (tuple, optional): Fractional level window.

    Returns:
        data (dict): Binned diagonal and off-diagonal ETH statistics.
    """
    levels = np.asarray(levels)
    vectors = np.asarray(vectors)
    if levels.ndim != 1:
        raise ValueError('levels must be a 1D array')
    if vectors.ndim != 2:
        raise ValueError('vectors must be a 2D array')
    if vectors.shape[1] != levels.size:
        raise ValueError('vectors.shape[1] must equal len(levels)')

    order = np.argsort(levels)
    levels = levels[order]
    vectors = vectors[:, order]
    if center_window:
        a, b = center_window
        i0, i1 = int(a*len(levels)), int(b*len(levels))
        levels = levels[i0:i1]
        vectors = vectors[:, i0:i1]

    P = local_projector_matrix(vectors, site=site, nstate=nstate)
    p_diag = np.real(np.diag(P))

    e_edges = np.linspace(levels.min(), levels.max(), nbins + 1)
    e_center = 0.5 * (e_edges[:-1] + e_edges[1:])
    diagonal_mean = np.zeros(nbins)
    diagonal_var = np.zeros(nbins)
    diagonal_count = np.zeros(nbins, dtype=int)
    for i in range(nbins):
        mask = (levels >= e_edges[i]) & (levels < e_edges[i+1])
        if i == nbins - 1:
            mask = (levels >= e_edges[i]) & (levels <= e_edges[i+1])
        diagonal_count[i] = np.count_nonzero(mask)
        if diagonal_count[i] > 0:
            diagonal_mean[i] = np.mean(p_diag[mask])
            diagonal_var[i] = np.var(p_diag[mask])

    ia, ib = np.triu_indices(len(levels), k=1)
    omega = np.abs(levels[ia] - levels[ib])
    offdiag = np.abs(P[ia, ib])**2

    if omega.size == 0:
        omega_center = np.asarray([])
        offdiag_mean = np.asarray([])
        offdiag_var = np.asarray([])
        offdiag_count = np.asarray([], dtype=int)
    else:
        w_edges = np.linspace(0., omega.max(), nbins + 1)
        omega_center = 0.5 * (w_edges[:-1] + w_edges[1:])
        offdiag_mean = np.zeros(nbins)
        offdiag_var = np.zeros(nbins)
        offdiag_count = np.zeros(nbins, dtype=int)
        for i in range(nbins):
            mask = (omega >= w_edges[i]) & (omega < w_edges[i+1])
            if i == nbins - 1:
                mask = (omega >= w_edges[i]) & (omega <= w_edges[i+1])
            offdiag_count[i] = np.count_nonzero(mask)
            if offdiag_count[i] > 0:
                offdiag_mean[i] = np.mean(offdiag[mask])
                offdiag_var[i] = np.var(offdiag[mask])

    return {
        'levels': levels,
        'diagonal': p_diag,
        'energy': e_center,
        'diagonal_mean': diagonal_mean,
        'diagonal_var': diagonal_var,
        'diagonal_count': diagonal_count,
        'omega': omega_center,
        'offdiag_mean': offdiag_mean,
        'offdiag_var': offdiag_var,
        'offdiag_count': offdiag_count,
    }


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
        Here ``times`` are spectral times conjugate to unfolded energy, not
        molecular-dynamics times.  A Thouless-time analysis asks when the
        form factor crosses over from system-specific dynamics to the
        universal RMT ramp.
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
        window = (0.2, 0.8)
        spectra = [mo_energy, energy, td_e]
        stats = [eigenvalue_statistics(e, deg=5, center_window=window,
                                       unfold=True, level_spacing=True,
                                       adjacent_gap_ratio=True)
                 for e in spectra]
        raw_stats = [eigenvalue_statistics(e, deg=5, center_window=window,
                                           unfold=False, level_spacing=False,
                                           adjacent_gap_ratio=True)
                     for e in spectra]

        s0, s1, s2 = [x['s'] for x in stats]
        r0, r1, r2 = [x['r'] for x in raw_stats]
        ru0, ru1, ru2 = [x['r'] for x in stats]
        r_mean = np.asarray([x['r_mean'] for x in raw_stats])
        ru_mean = np.asarray([x['r_mean'] for x in stats])

        np.savez('level_spacing.npz', mo_energy=mo_energy, energy=energy, td_e=td_e,
                 s0=s0, s1=s1, s2=s2, r0=r0, r1=r1, r2=r2,
                 ru0=ru0, ru1=ru1, ru2=ru2,
                 r_mean=r_mean, ru_mean=ru_mean)
        print('Adjacent-gap ratio <r>:', r_mean)
        print('Unfolded adjacent-gap ratio <r>:', ru_mean)

        fig, axes = plt.subplots(1, 3, tight_layout=True)
        for i, s in enumerate([s0, s1, s2]):
            ax = axes[i]
            ax.hist(s, bins=20, density=True, alpha=0.5, label=label[i], facecolor='C0')

            x = np.linspace(0, max(4, np.max(s)), 400)
            ax.plot(x, stat_wigner(x, 'goe'), label='GOE')
            ax.plot(x, stat_poisson(x), '--', label='Poisson')

            ax.set_xlabel('Unfolded spacing s')
            if i==0: ax.set_ylabel('P(s)')
            ax.set_title(r'$\langle r\rangle=%.3f$' % r_mean[i])
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
