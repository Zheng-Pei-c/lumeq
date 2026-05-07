from lumeq import np, itertools

from pyscf import lib, gto, scf
from pyscf.lib import logger
from pyscf.scf import cphf
from pyscf.dft import libxc


def _as_atom_list(mol, atmlst):
    if atmlst is None:
        return np.arange(mol.natm, dtype=int)
    if np.isscalar(atmlst):
        atmlst = [atmlst]
    atmlst = np.asarray(list(atmlst), dtype=int)
    if atmlst.ndim != 1:
        raise ValueError('atmlst must be a one-dimensional sequence of atom indices')
    if np.any(atmlst < 0) or np.any(atmlst >= mol.natm):
        raise ValueError('atom index in atmlst is out of range')
    return atmlst


def _check_supported_reference(mf, mo_coeff, mo_occ):
    if mo_coeff is None or mo_occ is None or getattr(mf, 'mo_energy', None) is None:
        raise RuntimeError('SCF object is not initialized')
    if isinstance(mo_coeff, (tuple, list)):
        raise NotImplementedError('unrestricted and generalized references are not supported')
    if np.iscomplexobj(mo_coeff):
        raise NotImplementedError('complex orbitals are not supported')
    if np.any((mo_occ != 0) & (mo_occ != 2)):
        raise NotImplementedError('only closed-shell restricted references with occupations 0 or 2 are supported')
    mol = mf.mol
    if mol.has_ecp() or bool(mol._pseudo):
        raise NotImplementedError('ECP and pseudopotential charge derivatives are not implemented')
    if getattr(mf, 'converged', True) is False:
        logger.warn(mf, 'SCF object is not converged')


def electronic_potential(mol, atmlst, mo_coeff=None, mo_occ=None):
    r"""AO, full-MO, or VO matrix for electronic potential on nuclei,
    \partial hcore / \partial Z_A."""
    potential_ao = []
    for ia in atmlst:
        ia = int(ia)
        coord = mol.atom_coord(ia)
        zeta = mol._env[mol._atm[ia, gto.PTR_ZETA]]
        with mol.with_rinv_origin(coord), mol.with_rinv_zeta(zeta):
            potential_ao.append(-mol.intor('int1e_rinv'))

    potential_ao = np.asarray(potential_ao)
    if mo_coeff is None:
        return potential_ao

    potential_mo = np.einsum('pi,xpq,qj->xij',
                             mo_coeff, potential_ao, mo_coeff)
    if mo_occ is None:
        return potential_ao, potential_mo

    occidx = mo_occ > 0
    viridx = mo_occ == 0
    # virtual-occupied block used as CPHF right-hand side
    return potential_ao, potential_mo[:, viridx, :][:, :, occidx]


def nuclear_charge_gradient(mol, atmlst):
    r"""Gradient of nuclear repulsion energy with respect to nuclear charges."""
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    rij = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(rij, np.inf)
    return np.sum(charges[None, :] / rij[atmlst], axis=1)


def nuclear_charge_hessian(mol, atmlst):
    r"""Hessian of nuclear repulsion energy with respect to nuclear charges."""
    coords = mol.atom_coords()
    rij = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    out = np.zeros((len(atmlst), len(atmlst)))
    for ia0, ia in enumerate(atmlst):
        for ib0, ib in enumerate(atmlst):
            if ia != ib:
                out[ia0, ib0] = 1.0 / rij[ia, ib]
    return out


def _gen_response_vind(mf, mo_coeff, mo_occ):
    r"""orbital Hessian matrix-vector product function for CPHF equation."""
    occidx = mo_occ > 0
    viridx = mo_occ == 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:, viridx]
    nmo = mo_coeff.shape[1]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    vresp = mf.gen_response(mo_coeff, mo_occ, singlet=None, hermi=1)

    def fvind(mo1):
        mo1 = mo1.reshape(-1, nvir, nocc)
        dm1 = np.einsum('pa,xai,qi->xpq', orbv, mo1 * 2.0, orbo)
        dm1 = dm1 + dm1.transpose(0, 2, 1)
        v1ao = vresp(dm1)
        return np.einsum('pa,xpq,qi->xai', orbv, v1ao, orbo)

    return fvind


def _xc_kernel(mf, dmvo, singlet=True, max_memory=2000):
    r"""Compute the XC kernel third-order derivative contribution.
    ``dmvo`` is the symmetrized virtual-occupied density response from the first-order charge perturbation.
    The returned ``k1ao`` is the AO representation of the XC kernel contribution to the third derivative, which is contracted with two first-order density response to yield the XC kernel contribution to the third derivative.
    The returned ``k1ao`` has shape ``(natm_sel, nao, nao)`` and is tagged with ``k1ao.atmlst``.
    """
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, getattr(mf, 'max_memory', 4000) * .8 - mem_now)

    mol = mf.mol
    grids = mf.grids
    xc_code = mf.xc

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    # dmvo is assumed to be symmetrized
    # hermi=1
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dmvo, 1, False, grids)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    ao_loc = mol.ao_loc_nr()

    deriv = 3
    k1 = np.zeros((nset, nset, nset))

    if xctype == 'HF':
        return k1
    elif xctype == 'LDA':
        ao_deriv = 1
    elif xctype == 'GGA':
        ao_deriv = 2
    elif xctype == 'MGGA':
        ao_deriv = 2
        logger.warn(mf, 'MGGA is not tested and may be inaccurate due to grids response')
    else:
        raise NotImplementedError(f'Third response for functional {xc_code}')

    if mf.do_nlc():
        raise NotImplementedError("NLC contribution is not supported yet. "
                                  "Please set exclude_nlc field of tdscf object to True, "
                                  "which will turn off NLC contribution in the whole response calculation.")

    if singlet:
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            if xctype == 'LDA':
                ao0 = ao[0]
            else:
                ao0 = ao
            rho = ni.eval_rho2(mol, ao0, mo_coeff, mo_occ, mask, xctype, with_lapl=False)
            kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[-1]

            if xctype == 'LDA':
                ao = ao[0]
            elif xctype == 'GGA':
                ao = ao[:4]
            elif xctype == 'MGGA':
                ao = ao[:10]

            rho1 = []
            for idm in range(nset):
                _rho1 = make_rho(idm, ao, mask, xctype)
                if xctype == 'LDA':
                    _rho1 = _rho1[np.newaxis]
                rho1.append(_rho1)
            rho1 = np.asarray(rho1)
            k1 += np.einsum('axg,byg,czg,xyzg,g->abc', rho1, rho1, rho1, kxc, weight)
    else:
        logger.warn(mf, 'Triplet contribution is not tested!!!')
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            if xctype == 'LDA':
                ao0 = ao[0]
            else:
                ao0 = ao
            rho = ni.eval_rho2(mol, ao0, mo_coeff, mo_occ, mask, xctype, with_lapl=False)
            rho *= .5
            kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[-1]

            if xctype == 'LDA':
                ao = ao[0]
            elif xctype == 'GGA':
                ao = ao[:4]
            elif xctype == 'MGGA':
                ao = ao[:10]

            rho1 = []
            for idm in range(nset):
                _rho1 = make_rho(idm, ao, mask, xctype)
                if xctype == 'LDA':
                    _rho1 = _rho1[np.newaxis]
                rho1.append(_rho1)
            rho1 = np.asarray(rho1)

            # kxc in terms of the triplet coupling
            # 1/2 int (tia - tIA) kxc (tjb - tJB) = tia kxc_t tjb
            kxc = kxc[0,:,0] - kxc[0,:,1]
            kxc = kxc[:,:,0] - kxc[:,:,1]
            k1 += np.einsum('axg,byg,czg,xyzg,g->abc', rho1, rho1, rho1, kxc, weight)

    return k1


def alchemical_energy_gradient(mf, atmlst=None, h1ao=None,
                               include_electronic=True, include_nuclear=True):
    r"""First alchemical derivative \partial E / \partial Z_A.

    Implements Eq. (16) of Lesiuk, Balawender, and Zachara,
    J. Chem. Phys. 136, 034104 (2012), DOI: 10.1063/1.3674163,
    for closed-shell restricted HF/KS at fixed geometry, electron count,
    and AO basis:

    \[
        \frac{\partial E}{\partial Z_A}
        =
        \mathrm{Tr}[D h^A]
        + \sum_{B \ne A} \frac{Z_B}{R_{AB}},
        \qquad
        h^A_{\mu\nu} = \langle \mu | -r_A^{-1} | \nu \rangle .
    \]

    The density matrix is spin-summed.  All derivatives are evaluated at
    fixed geometry, fixed electron count, and fixed AO basis.  For RKS, the
    XC functional, grid, and AO basis parameters are not differentiated with
    respect to nuclear charge.

    Args:
        mf: Converged PySCF RHF/RKS object.  The reference must be real,
            closed-shell, and all-electron.
        atmlst: Atom indices for which derivatives are requested.  If
            ``None``, all atoms are used.
        h1ao: Optional precomputed AO potential matrices ``(natm_sel, nao, nao)``.
        include_electronic: Include the electronic Hellmann-Feynman term.
        include_nuclear: Include the nuclear repulsion derivative.

    Returns:
        Tagged array ``d1`` with shape ``(natm_sel,)``.  Attributes:
        ``d1.electronic``, ``d1.nuclear``.
    """
    mol = mf.mol
    atmlst = _as_atom_list(mol, atmlst)
    electronic = np.zeros(len(atmlst))
    nuclear = np.zeros(len(atmlst))

    if include_electronic:
        if h1ao is None:
            h1ao = electronic_potential(mol, atmlst)
        dm0 = mf.make_rdm1()
        electronic = np.einsum('xpq,pq->x', h1ao, dm0).real

    if include_nuclear:
        nuclear = nuclear_charge_gradient(mol, atmlst)

    total = electronic + nuclear
    return lib.tag_array(total, electronic=electronic, nuclear=nuclear)


def solve_charge_response(mf, atmlst=None, mo_energy=None, mo_coeff=None,
                          mo_occ=None, h1vo=None, max_cycle=50, tol=None,
                          level_shift=0.0, verbose=None):
    r"""Solve CPSCF response to nuclear-charge perturbations.

    Uses PySCF's CPHF/CPKS solver for Eq. (24) of Lesiuk, Balawender,
    and Zachara, J. Chem. Phys. 136, 034104 (2012), DOI:
    10.1063/1.3674163:

    \[
        \sum_{bj} A_{ai,bj} U^A_{bj} = -h^A_{ai},
        \qquad
        h^A_{ai} = \langle a | -r_A^{-1} | i \rangle .
    \]

    ``A`` is the closed-shell electronic Hessian/orbital Hessian represented
    as a matrix-vector product by ``mf.gen_response``.  The returned
    ``U`` is in the virtual-occupied MO block only, with array convention
    ``mo1[A, a, i]`` for atom ``A``, virtual orbital ``a``, and occupied
    orbital ``i``.  No overlap-response term is present because changing
    nuclear charges does not move centers or AO basis functions.

    Args:
        mf: Converged PySCF RHF/RKS object.
        atmlst: Atom indices for charge perturbations.  If ``None``, all
            atoms are used.
        mo_energy: Optional MO energies ``(nmo,)``.
        mo_coeff: Optional MO coefficient matrix ``(nao, nmo)``.
        mo_occ: Optional MO occupation vector ``(nmo,)``.
        h1vo: Optional virtual-occupied MO perturbation matrices
            ``(natm_sel, nvir, nocc)``.  If omitted, they are built from
            ``int1e_rinv`` by :func:`electronic_potential`.
        max_cycle: Maximum Krylov iterations for PySCF CPHF.
        tol: CPHF convergence tolerance.  Defaults to ``mf.conv_tol_cpscf``.
        level_shift: Level shift passed to ``pyscf.scf.cphf.solve``.
        verbose: PySCF logger verbosity.

    Returns:
        Tagged response array ``mo1`` with shape
        ``(natm_sel, nvir, nocc)``.  Attributes: ``mo1.h1vo`` and
        ``mo1.atmlst``.
    """
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    _check_supported_reference(mf, mo_coeff, mo_occ)

    mol = mf.mol
    atmlst = _as_atom_list(mol, atmlst)
    if h1vo is None:
        h1vo = electronic_potential(mol, atmlst, mo_coeff, mo_occ)[1]
    else:
        h1vo = np.asarray(h1vo)

    if tol is None:
        tol = getattr(mf, 'conv_tol_cpscf', 1e-9)
    if verbose is None:
        verbose = getattr(mf, 'verbose', logger.WARN)
    fvind = _gen_response_vind(mf, mo_coeff, mo_occ)
    mo1, _ = cphf.solve(
        fvind, mo_energy, mo_occ, h1vo, s1=None, max_cycle=max_cycle,
        tol=tol, hermi=False, verbose=verbose, level_shift=level_shift)

    return lib.tag_array(mo1, h1vo=h1vo, atmlst=atmlst)


def alchemical_energy_hessian(mf, atmlst=None, mo_energy=None,
                              mo_coeff=None, mo_occ=None,
                              response=None, h1vo=None,
                              include_electronic=True,
                              include_nuclear=True,
                              symmetrize=True, max_cycle=50,
                              tol=None, level_shift=0.0, verbose=None):
    r"""Mixed second derivative \partial^2 E / \partial Z_A \partial Z_B.

    Implements Eq. (17) of Lesiuk, Balawender, and Zachara,
    J. Chem. Phys. 136, 034104 (2012), DOI: 10.1063/1.3674163.
    The electronic term is \(4 (h^A)^T U^B\), equivalently
    \(-4 (\phi^A)^T A^{-1} \phi^B\) for \(\phi^A = -h^A\).  The factor of
    four is the restricted closed-shell spin factor and orbital-rotation
    normalization used by PySCF's CPHF convention.  The returned Hessian is
    tagged with ``electronic``, ``nuclear`` components.

    \[
        \frac{\partial^2 E}{\partial Z_A \partial Z_B}
        =
        4 \sum_{ai} h^A_{ai} U^B_{ai}
        + (1-\delta_{AB}) R_{AB}^{-1}.
    \]

    Args:
        mf: Converged PySCF RHF/RKS object.
        atmlst: Atom indices for the mixed derivatives.  If ``None``, all
            atoms are used.
        mo_energy: Optional MO energies ``(nmo,)``.
        mo_coeff: Optional MO coefficient matrix ``(nao, nmo)``.
        mo_occ: Optional MO occupation vector ``(nmo,)``.
        response: Optional precomputed ``mo1`` from
            :func:`solve_charge_response`.
        h1vo: Optional MO perturbation matrices ``(natm_sel, nvir, nocc)``.
        include_electronic: Include the CPSCF electronic response term.
        include_nuclear: Include the nuclear repulsion mixed derivative.
        symmetrize: Symmetrize the returned Hessian over ``A`` and ``B``.
        max_cycle: Maximum CPHF iterations if ``response`` is not supplied.
        tol: CPHF convergence tolerance if ``response`` is not supplied.
        level_shift: Level shift passed to ``pyscf.scf.cphf.solve``.
        verbose: PySCF logger verbosity.

    Returns:
        Tagged array ``d2`` with shape ``(natm_sel, natm_sel)``.
        Attributes: ``d2.electronic``, ``d2.nuclear``.
    """
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    _check_supported_reference(mf, mo_coeff, mo_occ)

    mol = mf.mol
    atmlst = _as_atom_list(mol, atmlst)
    electronic = np.zeros((len(atmlst), len(atmlst)))
    nuclear = np.zeros_like(electronic)

    if include_electronic:
        if h1vo is None and response is not None and hasattr(response, 'h1vo'):
            h1vo = response.h1vo
        if h1vo is None:
            h1vo = electronic_potential(mol, atmlst, mo_coeff, mo_occ)[1]
        else:
            h1vo = np.asarray(h1vo)
        if response is None:
            response = solve_charge_response(
                mf, atmlst=atmlst, mo_energy=mo_energy,
                mo_coeff=mo_coeff, mo_occ=mo_occ, h1vo=h1vo,
                max_cycle=max_cycle, tol=tol, level_shift=level_shift,
                verbose=verbose)
        electronic = 4.0 * np.einsum('xai,yai->xy', h1vo, response).real

    if include_nuclear:
        nuclear = nuclear_charge_hessian(mol, atmlst)

    if symmetrize:
        electronic = 0.5 * (electronic + electronic.T)
        nuclear = 0.5 * (nuclear + nuclear.T)

    total = electronic + nuclear
    return lib.tag_array(total, electronic=electronic, nuclear=nuclear)


def alchemical_energy_third_order(
        mf, atmlst=None, mo_energy=None, mo_coeff=None, mo_occ=None,
        response=None, h1ao=None, h1vo=None, symmetrize=True, max_cycle=50,
        tol=None, level_shift=0.0, verbose=None):
    r"""Third derivative from the Wigner-rule Eq. (32).

    Implements Eq. (32) of Lesiuk, Balawender, and Zachara,
    J. Chem. Phys. 136, 034104 (2012), DOI: 10.1063/1.3674163,
    using the second-order inhomogeneity \(B^{\beta\gamma}\) of Eq. (29).
    No second-order CPSCF equation is solved; only the first-order charge
    response \(U^\alpha_{ai}\) from :func:`solve_charge_response` is used.

    \[
        E^{ABC}_{e}
        =
        -4 \sum_{pi} B^{BC}_{pi} U^A_{pi}
        +4 \sum_{pqi} U^B_{pi} U^C_{qi} h^A_{pq},
        \qquad h^A_{pq} = \langle p | -r_A^{-1} | q \rangle .
    \]

    The response amplitudes use the convention ``response[A, a, i] =
    U^A_{ai}``; they are expanded to a skew-symmetric full MO matrix with
    zero occupied-occupied and virtual-virtual gauge blocks.  The two-electron
    pieces in \(B^{\beta\gamma}\) are evaluated by ``mf.gen_response`` from
    product-density seeds.  The PySCF MO convention also requires the
    first-order orbital-energy derivative terms from Eq. (25); these are
    the diagonal of the first-order Fock derivative ``g1mo``.  The
    \(\delta K_{xc}/\delta\rho\) contribution is evaluated through
    ``pyscf.grad.tdrks._contract_xc_kernel``.  No MO two-electron integral
    transformation is formed.

    Args:
        mf: Converged PySCF RHF/RKS object.
        atmlst: Atom indices for the third derivatives.  If ``None``, all
            atoms are used.
        mo_energy: Optional MO energies ``(nmo,)``.
        mo_coeff: Optional MO coefficient matrix ``(nao, nmo)``.
        mo_occ: Optional MO occupation vector ``(nmo,)``.
        response: Optional precomputed reference-charge ``mo1`` from
            :func:`solve_charge_response`.
        h1vo: Optional MO perturbation matrices at the reference charge.
        symmetrize: Symmetrize over the three nuclear-charge indices.
        max_cycle: Maximum CPHF iterations if ``response`` is not supplied.
        tol: CPHF convergence tolerance if ``response`` is not supplied.
        level_shift: Level shift passed to ``pyscf.scf.cphf.solve``.
        verbose: PySCF logger verbosity.

    Returns:
        Tagged array ``d3`` with shape
        ``(natm_sel, natm_sel, natm_sel)``.  Attributes:
        ``d3.electronic``, ``d3.nuclear``.
    """
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    _check_supported_reference(mf, mo_coeff, mo_occ)

    mol = mf.mol
    atmlst = _as_atom_list(mol, atmlst)
    occidx = mo_occ > 0
    viridx = mo_occ == 0
    occ = np.where(occidx)[0]
    vir = np.where(viridx)[0]
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:, viridx]
    nmo = mo_coeff.shape[1]
    nocc = len(occ)
    nvir = len(vir)

    if h1vo is None and response is not None and hasattr(response, 'h1vo'):
        h1vo = response.h1vo
    if h1ao is None and h1vo is None:
        h1ao, h1vo = electronic_potential(mol, atmlst, mo_coeff, mo_occ)
    elif h1ao is None:
        h1ao = electronic_potential(mol, atmlst, mo_coeff, mo_occ)[0]
    elif h1vo is None:
        h1vo = electronic_potential(mol, atmlst, mo_coeff, mo_occ)[1]

    if response is None:
        response = solve_charge_response(
            mf, atmlst=atmlst, mo_energy=mo_energy, mo_coeff=mo_coeff,
            mo_occ=mo_occ, h1vo=h1vo, max_cycle=max_cycle, tol=tol,
            level_shift=level_shift, verbose=verbose)

    vresp = mf.gen_response(mo_coeff, mo_occ, singlet=None, hermi=1)
    dmvo = np.einsum('pa,xai,qi->xpq', orbv, response * 2.0, orbo)
    dmvo += dmvo.transpose(0, 2, 1)

    v1ao = np.asarray(h1ao) + vresp(dmvo)
    v1oo = np.einsum('xpq,pi,qj->xij', v1ao, orbo, orbo)
    v1vv = np.einsum('xpq,pa,qb->xab', v1ao, orbv, orbv)
    electronic  = np.einsum('xab,yai,zbi->xyz', v1vv, response, response)
    electronic -= np.einsum('xji,yai,zaj->xyz', v1oo, response, response)

    electronic *= 2.0
    electronic = (electronic
                  + electronic.transpose(0, 2, 1)
                  + electronic.transpose(1, 0, 2)
                  + electronic.transpose(1, 2, 0)
                  + electronic.transpose(2, 0, 1)
                  + electronic.transpose(2, 1, 0))
    electronic += _xc_kernel(mf, dmvo)

    nuclear = np.zeros_like(electronic)
    total = electronic + nuclear
    return lib.tag_array(total, electronic=electronic, nuclear=nuclear)


def alchemical_derivatives(mf, atmlst=None, mo_energy=None, mo_coeff=None,
                           mo_occ=None, max_cycle=50, tol=None,
                           level_shift=0.0, verbose=None, third_order=True):
    r"""Compute first, second, and third charge derivatives.

    Uses the closed-shell CPSCF formulas of Lesiuk, Balawender, and
    Zachara, J. Chem. Phys. 136, 034104 (2012), DOI:
    10.1063/1.3674163.

    This wrapper builds the virtual-occupied charge perturbations once,
    solves the first-order charge response once, and reuses those
    intermediates for
    \(\partial E / \partial Z_A\), \(\partial^2 E / \partial Z_A \partial Z_B\),
    and \(\partial^3 E / \partial Z_A \partial Z_B \partial Z_C\)

    Args:
        mf: Converged PySCF RHF/RKS object.
        atmlst: Atom indices.  If ``None``, all atoms are used.
        mo_energy: Optional MO energies ``(nmo,)``.
        mo_coeff: Optional MO coefficient matrix ``(nao, nmo)``.
        mo_occ: Optional MO occupation vector ``(nmo,)``.
        max_cycle: Maximum CPHF iterations.
        tol: CPHF convergence tolerance.
        level_shift: Level shift passed to ``pyscf.scf.cphf.solve``.
        verbose: PySCF logger verbosity.

    Returns:
        ``(d1, d2, d3)`` where ``d1`` is the tagged first-derivative array
        from :func:`alchemical_energy_gradient`, ``d2`` is the tagged
        second-derivative array from
        :func:`alchemical_energy_hessian`, and ``d3`` is the tagged
        third-derivative array from
        :func:`alchemical_energy_third_order`.
    """
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mol = mf.mol
    atmlst = _as_atom_list(mol, atmlst)
    h1ao, h1vo = electronic_potential(mol, atmlst, mo_coeff, mo_occ)
    response = solve_charge_response(
        mf, atmlst=atmlst, mo_energy=mo_energy, mo_coeff=mo_coeff,
        mo_occ=mo_occ, h1vo=h1vo, max_cycle=max_cycle, tol=tol,
        level_shift=level_shift, verbose=verbose)

    first = alchemical_energy_gradient(mf, atmlst=atmlst, h1ao=h1ao)
    second = alchemical_energy_hessian(
        mf, atmlst=atmlst, mo_energy=mo_energy, mo_coeff=mo_coeff,
        mo_occ=mo_occ, response=response, h1vo=h1vo)

    third = None
    if third_order:
        third = alchemical_energy_third_order(
            mf, atmlst=atmlst, mo_energy=mo_energy, mo_coeff=mo_coeff,
            mo_occ=mo_occ, response=response, h1ao=h1ao, h1vo=h1vo,
            max_cycle=max_cycle, tol=tol, level_shift=level_shift,
            verbose=verbose)

    return first, second, third


def fractional_charge_mol(mol0, charges):
    mol1 = mol0.copy()
    mol1._atm = mol1._atm.copy()
    mol1._env = mol1._env.copy()
    env = list(mol1._env)
    for ia, charge in enumerate(charges):
        ptr = len(env)
        env.append(float(charge))
        mol1._atm[ia, gto.NUC_MOD_OF] = gto.NUC_FRAC_CHARGE
        mol1._atm[ia, gto.PTR_FRAC_CHARGE] = ptr
    mol1._env = np.asarray(env, dtype=float)
    mol1.nelectron = mol0.nelectron
    mol1.enuc = mol1.energy_nuc()
    return mol1

def make_rks(mol, functional, pcm_options=None):
    mf = scf.RKS(mol)
    mf.xc = functional
    if pcm_options is not None:
        mf = mf.PCM()
        for key, value in pcm_options.items():
            setattr(mf.with_solvent, key, value)
        mf.with_solvent.equilibrium_solvation = True
    return mf


def energy_with_charges(mol0, charges, cache, functional,
                        pcm_options=None):
    key = tuple(np.round(charges, 12))
    if key not in cache:
        mol1 = fractional_charge_mol(mol0, charges)
        mf1 = make_rks(mol1, functional, pcm_options=pcm_options)
        mf1.init_guess = '1e'
        mf1.conv_tol = 1e-12
        mf1.max_cycle = 100
        mf1.kernel()
        if not mf1.converged:
            raise RuntimeError('finite-difference SCF did not converge')
        cache[key] = mf1.e_tot
    return cache[key]


def finite_difference_derivatives(mol0, functional, pcm_options=None,
                                  step=1e-4, step3=2e-3, third_order=True):
    charges = mol0.atom_charges().astype(float)
    natm = mol0.natm
    cache = {}
    e0 = energy_with_charges(
        mol0, charges, cache, functional, pcm_options=pcm_options)
    fd1 = np.zeros(natm)
    fd2 = np.zeros((natm, natm))
    fd3 = np.zeros((natm, natm, natm))

    for ia in range(natm):
        zp = charges.copy()
        zm = charges.copy()
        zp[ia] += step
        zm[ia] -= step
        ep = energy_with_charges(
            mol0, zp, cache, functional, pcm_options=pcm_options)
        em = energy_with_charges(
            mol0, zm, cache, functional, pcm_options=pcm_options)
        fd1[ia] = (ep - em) / (2.0 * step)
        fd2[ia, ia] = (ep - 2.0 * e0 + em) / step ** 2

    for ia in range(natm):
        for ib in range(ia):
            zpp = charges.copy()
            zpm = charges.copy()
            zmp = charges.copy()
            zmm = charges.copy()
            zpp[[ia, ib]] += step
            zpm[ia] += step
            zpm[ib] -= step
            zmp[ia] -= step
            zmp[ib] += step
            zmm[[ia, ib]] -= step
            epp = energy_with_charges(
                mol0, zpp, cache, functional, pcm_options=pcm_options)
            epm = energy_with_charges(
                mol0, zpm, cache, functional, pcm_options=pcm_options)
            emp = energy_with_charges(
                mol0, zmp, cache, functional, pcm_options=pcm_options)
            emm = energy_with_charges(
                mol0, zmm, cache, functional, pcm_options=pcm_options)
            fd2[ia, ib] = fd2[ib, ia] = (
                epp - epm - emp + emm) / (4.0 * step ** 2)

    if not third_order:
        return fd1, fd2, fd3

    for ia, ib, ic in itertools.product(range(natm), repeat=3):
        value = 0.0
        for sa, sb, sc in itertools.product((-1, 1), repeat=3):
            z = charges.copy()
            z[ia] += sa * step3
            z[ib] += sb * step3
            z[ic] += sc * step3
            value += (
                sa * sb * sc
                * energy_with_charges(
                    mol0, z, cache, functional,
                    pcm_options=pcm_options))
        fd3[ia, ib, ic] = value / (8.0 * step3 ** 3)

    return fd1, fd2, fd3


if __name__ == '__main__':
    atom = '''
    O  0.000000  0.000000  0.000000
    H  0.000000 -0.757160  0.586260
    H  0.000000  0.757160  0.586260
    '''
    functional = 'wb97xd'
    basis = '6-31g*'
    pcm_options = None
    #pcm_options = {'eps': 78.3553}

    third_order = True

    mol = gto.M(
        atom=atom,
        basis=basis,
        unit='Angstrom',
        verbose=0,
    )
    mf = make_rks(mol, functional, pcm_options=pcm_options)
    mf.run(conv_tol=1e-12)
    d1, d2, d3 = alchemical_derivatives(mf, tol=1e-12, third_order=third_order)

    print('SCF energy:', mf.e_tot)
    print('dE/dZ:')
    print(d1)
    print('dE/dZ electronic:')
    print(d1.electronic)
    print('dE/dZ nuclear:')
    print(d1.nuclear)
    print('d2E/dZdZ:')
    print(d2)
    print('d2E/dZdZ electronic:')
    print(d2.electronic)
    print('d2E/dZdZ nuclear:')
    print(d2.nuclear)
    print('Eq. 32 d3E/dZdZdZ:')
    print(d3)

    fd1, fd2, fd3 = finite_difference_derivatives(
        mol, functional, pcm_options=pcm_options, third_order=third_order)
    print('finite-difference dE/dZ:')
    print(fd1)
    print('max |analytic - finite-difference| dE/dZ:',
          np.max(np.abs(d1 - fd1)))
    print('finite-difference d2E/dZdZ:')
    print(fd2)
    print('max |analytic - finite-difference| d2E/dZdZ:',
          np.max(np.abs(d2 - fd2)))
    print('finite-difference d3E/dZdZdZ:')
    print(fd3)
    print('max |analytic - finite-difference| d3E/dZdZdZ:',
          np.max(np.abs(d3 - fd3)))
