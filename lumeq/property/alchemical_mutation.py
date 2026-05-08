from lumeq import np, itertools
from lumeq.utils import print_matrix

from pyscf import lib, gto, scf
from pyscf.lib import logger
from pyscf.scf import cphf
from pyscf.dft import rks
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
        with mol.with_rinv_at_nucleus(int(ia)):
            potential_ao.append(-mol.intor('int1e_rinv'))

    potential_ao = np.asarray(potential_ao)
    if mo_coeff is None:
        return potential_ao

    if mo_occ is None:
        potential_mo = np.einsum('pi,xpq,qj->xij',
                             mo_coeff, potential_ao, mo_coeff)
        return potential_ao, potential_mo

    else:
        # virtual-occupied block used as CPHF right-hand side
        orbo = mo_coeff[:, mo_occ > 0]
        orbv = mo_coeff[:, mo_occ == 0]
        potential_mo = np.einsum('pa,xpq,qi->xai',
                                 orbv, potential_ao, orbo)
        return potential_ao, potential_mo


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


def _response_density(theta, mo_coeff, mo_occ, symmetric=True):
    r"""Density response from a given orbital rotation ``theta``."""
    orbo = mo_coeff[:, mo_occ > 0]
    orbv = mo_coeff[:, mo_occ == 0]
    dm = np.einsum('pa,xai,qi->xpq', orbv, theta * 2.0, orbo)
    if symmetric:
        dm += dm.transpose(0, 2, 1)
    return dm


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


def solve_charge_response(mf, atmlst=None, h1vo=None, max_cycle=50, tol=None,
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
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
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


def nuclear_potential_on_surface(pcmobj, mol, atmlst):
    r"""Compute the nuclear potential on PCM surface points for a given list of atoms."""
    coords = pcmobj.surface['grid_coords']
    charge_exp = pcmobj.surface['charge_exp']
    fakemol = gto.fakemol_for_charges(coords, expnt=charge_exp**2)
    atom_coords = mol.atom_coords(unit='B')
    fakemol_nuc = gto.fakemol_for_charges(atom_coords[atmlst])
    int2c2e = mol._add_suffix('int2c2e')
    v_nuc = gto.mole.intor_cross(int2c2e, fakemol_nuc, fakemol)
    return v_nuc


def surface_charges_from_potential(pcmobj, v_grids, sym=True):
    r"""Compute surface charges from potential on grids by solving PCM equations.
    """
    if not getattr(pcmobj, "_intermediates", None):
        pcmobj.build()

    K = pcmobj._intermediates["K"]
    R = pcmobj._intermediates["R"]

    v = np.asarray(v_grids)
    one_vector = v.ndim == 1
    v = v.reshape(1, -1) if one_vector else v

    q = np.linalg.solve(K, np.dot(R, v.T)).T

    if sym:
        vK_1 = np.linalg.solve(K.T, v.T)
        qt = np.dot(R.T, vK_1).T
        q = 0.5 * (q + qt)

    return q[0] if one_vector else q



class Alchem(rks.RKS):
    r"""
    Alchemical derivatives with respect to nuclear charges.

    Reference: Lesiuk, Balawender, and Zachara, J. Chem. Phys. 136, 034104 (2012),
    DOI:  10.1063/1.3674163
    """
    def set_alchem_atmlst(self, atmlst=None):
        self.alchem_atmlst = _as_atom_list(self.mol, atmlst)
        return self.alchem_atmlst


    def electronic_potential(self, atmlst=None):
        r"""AO and MO representations of the electronic potential on nuclei."""
        if atmlst is None: atmlst = self.alchem_atmlst

        potential = electronic_potential(self.mol, atmlst, self.mo_coeff,
                                         self.mo_occ)
        self.potential_ao = potential[0]
        self.potential_mo = potential[1]
        return potential


    def electronic_charge_gradient(self, atmlst=None):
        r"""Electronic potential on nuclei."""
        if atmlst is None: atmlst = self.alchem_atmlst
        if not hasattr(self, 'potential_ao'):
            self.electronic_potential(atmlst=atmlst)[0]
        dm0 = self.make_rdm1()
        return np.einsum('xpq,pq->x', self.potential_ao, dm0).real


    def electronic_charge_hessian(self, atmlst=None, response=None):
        r"""Second derivative of electronic energy with respect to nuclear charges."""
        if atmlst is None: atmlst = self.alchem_atmlst
        if not hasattr(self, 'potential_mo'):
            self.electronic_potential(atmlst=atmlst)[1]
        if response is None:
            if not hasattr(self, 'response'):
                self.solve_charge_response(atmlst=atmlst)
            response = self.response

        h1vo = self.potential_mo
        return 4.0 * np.einsum('xai,yai->xy', h1vo, response).real


    def solvent_charge_gradient(self, atmlst=None):
        r"""Gradient of solvent contribution to energy with respect to nuclear charges."""
        if atmlst is None: atmlst = self.alchem_atmlst
        if not hasattr(self, 'with_solvent'):
            return np.zeros(len(atmlst))

        # get pcm class variable
        pcmobj = self.with_solvent

        # get pcm scf surface charges
        if not getattr(pcmobj, '_intermediates', None):
            pcmobj.build()
        if 'q_sym' not in pcmobj._intermediates:
            pcmobj.kernel(self.make_rdm1())
        q = pcmobj._intermediates.get('q_sym', pcmobj._intermediates.get('q'))

        # get nuclear potential on surface points
        if not hasattr(self, 'v_nuc_surface'):
            self.v_nuc_surface = nuclear_potential_on_surface(
                    pcmobj, self.mol, atmlst)
        v_nuc = self.v_nuc_surface

        # get the gradient contribution
        return np.einsum('g,xg->x', q, v_nuc).real


    def solvent_charge_hessian(self, atmlst=None, dmvo=None):
        r"""Hessian of solvent contribution to energy with respect to nuclear charges."""
        if atmlst is None: atmlst = self.alchem_atmlst
        if not hasattr(self, 'with_solvent'):
            return np.zeros((len(atmlst), len(atmlst)))

        if dmvo is None:
            if not hasattr(self, 'dmvo'):
                self.solve_charge_response(atmlst=atmlst)
            dmvo = self.dmvo

        # get pcm class variable
        pcmobj = self.with_solvent

        # get electronic potential from induced charges at surface points
        v_ele = pcmobj._get_v(dmvo)
        # get nuclear potential on surface points
        if not hasattr(self, 'v_nuc_surface'):
            self.v_nuc_surface = nuclear_potential_on_surface(
                    pcmobj, self.mol, atmlst)
        v_nuc = self.v_nuc_surface
        # get induced surface charges
        qind = surface_charges_from_potential(pcmobj, v_nuc - v_ele)

        return np.einsum('xg,yg->xy', qind, v_nuc).real


    def nuclear_charge_gradient(self, atmlst=None):
        r"""Gradient of nuclear repulsion energy with respect to nuclear charges."""
        if atmlst is None: atmlst = self.alchem_atmlst
        return nuclear_charge_gradient(self.mol, atmlst)


    def nuclear_charge_hessian(self, atmlst=None):
        r"""Hessian of nuclear repulsion energy with respect to nuclear charges."""
        if atmlst is None: atmlst = self.alchem_atmlst
        return nuclear_charge_hessian(self.mol, atmlst)


    def solve_charge_response(self, atmlst=None, h1vo=None, max_cycle=50,
                              tol=None, level_shift=0.0, verbose=None):
        r"""Solve CPSCF response to nuclear-charge perturbations."""
        if atmlst is None: atmlst = self.alchem_atmlst
        if h1vo is None:
            if not hasattr(self, 'potential_mo'):
                self.electronic_potential(atmlst=atmlst)
            h1vo = self.potential_mo.copy()

            # add PCM contribution to the right-hand-side
            if hasattr(self, 'with_solvent') and self.with_solvent.equilibrium_solvation:
                pcmobj = self.with_solvent
                # get nuclear potential on surface points
                if not hasattr(self, 'v_nuc_surface'):
                    self.v_nuc_surface = nuclear_potential_on_surface(
                            pcmobj, self.mol, atmlst)
                v_nuc = self.v_nuc_surface
                # get induced surface charges from nuclear potential
                qind = surface_charges_from_potential(pcmobj, v_nuc)
                # get potential on AO basis from induced charges
                vmat = pcmobj._get_vmat(qind)
                orbo = self.mo_coeff[:, self.mo_occ > 0]
                orbv = self.mo_coeff[:, self.mo_occ == 0]
                h1vo += np.einsum('xpq,pa,qi->xai', vmat, orbv, orbo)

                self.solvent_h1ao = vmat
                self.solvent_h1vo = h1vo - self.potential_mo

        self.response = solve_charge_response(
            self, atmlst=atmlst, h1vo=h1vo, max_cycle=max_cycle, tol=tol,
            level_shift=level_shift, verbose=verbose)

        # get density matrix as well
        self.dmvo = _response_density(self.response, self.mo_coeff, self.mo_occ)
        return self.response


    def alchemical_gradient(self, atmlst=None):
        r"""Gradient of total energy with respect to nuclear charges."""
        if atmlst is None: atmlst = self.alchem_atmlst
        electronic = self.electronic_charge_gradient(atmlst=atmlst)
        nuclear = self.nuclear_charge_gradient(atmlst=atmlst)
        solvent = self.solvent_charge_gradient(atmlst=atmlst)
        total = electronic + nuclear + solvent

        self.z_grad = lib.tag_array(total, electronic=electronic,
                                    nuclear=nuclear, solvent=solvent)
        return self.z_grad


    def alchemical_hessian(self, atmlst=None, symmetrize=True,
                           max_cycle=50, tol=None, level_shift=0.0,
                           verbose=None):
        r"""Hessian of total energy with respect to nuclear charges."""
        if atmlst is None: atmlst = self.alchem_atmlst
        if not hasattr(self, 'response'):
            self.solve_charge_response(atmlst=atmlst, max_cycle=max_cycle,
                                       tol=tol, level_shift=level_shift,
                                       verbose=verbose)

        electronic = self.electronic_charge_hessian(atmlst=atmlst)
        nuclear = self.nuclear_charge_hessian(atmlst=atmlst)
        solvent = self.solvent_charge_hessian(atmlst=atmlst)

        if symmetrize:
            electronic = 0.5 * (electronic + electronic.T)
            nuclear = 0.5 * (nuclear + nuclear.T)
            solvent = 0.5 * (solvent + solvent.T)

        total = electronic + nuclear + solvent
        self.z_hess = lib.tag_array(total, electronic=electronic,
                                    nuclear=nuclear, solvent=solvent)
        return self.z_hess


    def alchemical_third_order(self, atmlst=None, max_cycle=50, tol=None,
                               level_shift=0.0, verbose=None):
        r"""Third derivative of total energy with respect to nuclear charges.
        2n+1 rule is used."""
        if not hasattr(self, 'potential_ao') or not hasattr(self, 'potential_mo'):
            self.electronic_potential(atmlst=atmlst)

        # add PCM contribution to the right-hand-side
        if not hasattr(self, 'solvent_h1ao') or not hasattr(self, 'solvent_h1vo'):
            if hasattr(self, 'with_solvent') and self.with_solvent.equilibrium_solvation:
                pcmobj = self.with_solvent
                # get nuclear potential on surface points
                if not hasattr(self, 'v_nuc_surface'):
                    self.v_nuc_surface = nuclear_potential_on_surface(
                            pcmobj, self.mol, atmlst)
                v_nuc = self.v_nuc_surface
                # get induced surface charges from nuclear potential
                qind = surface_charges_from_potential(pcmobj, v_nuc)
                # get potential on AO basis from induced charges
                vmat = pcmobj._get_vmat(qind)
                orbo = self.mo_coeff[:, self.mo_occ > 0]
                orbv = self.mo_coeff[:, self.mo_occ == 0]
                h1vo = np.einsum('xpq,pa,qi->xai', vmat, orbv, orbo)
                self.solvent_h1ao = vmat
                self.solvent_h1vo = h1vo
            else:
                self.solvent_h1ao = np.zeros_like(self.potential_ao)
                self.solvent_h1vo = np.zeros_like(self.potential_mo)

        h1ao = self.potential_ao + self.solvent_h1ao
        h1vo = self.potential_mo + self.solvent_h1vo
        if not hasattr(self, 'response') or not hasattr(self, 'dmvo'):
            self.solve_charge_response(atmlst=atmlst, max_cycle=max_cycle,
                                       tol=tol, level_shift=level_shift,
                                       verbose=verbose)

        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        orbo = mo_coeff[:, mo_occ > 0]
        orbv = mo_coeff[:, mo_occ == 0]

        dmvo = self.dmvo
        response = self.response
        vresp = self.gen_response(mo_coeff, mo_occ, singlet=None, hermi=1)

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
        electronic += _xc_kernel(self, dmvo)

        nuclear = np.zeros_like(electronic)
        total = electronic + nuclear
        return lib.tag_array(total, electronic=electronic, nuclear=nuclear)


    def alchemical_derivatives(self, atmlst=None, max_cycle=50, tol=None,
                               level_shift=0.0, verbose=None, third_order=True,
                               solvent_response=True):
        r"""Compute first, second, and optionally third derivatives of total energy with respect to nuclear charges."""
        if atmlst is None:
            if not hasattr(self, 'alchem_atmlst'):
                self.set_alchem_atmlst()
            atmlst = self.alchem_atmlst

        if solvent_response and hasattr(self, 'with_solvent'):
            # turn on solvent response to orbital Hessian for CPHF
            self.with_solvent.equilibrium_solvation = True

        if not hasattr(self, 'response'):
            self.solve_charge_response(atmlst=atmlst, max_cycle=max_cycle,
                                       tol=tol, level_shift=level_shift,
                                       verbose=verbose)

        first = self.alchemical_gradient(atmlst=atmlst)
        second = self.alchemical_hessian(atmlst=atmlst, max_cycle=max_cycle,
                                         tol=tol, level_shift=level_shift,
                                         verbose=verbose)

        if not third_order:
            return first, second, None

        third = self.alchemical_third_order(
            atmlst=atmlst, max_cycle=max_cycle, tol=tol,
            level_shift=level_shift, verbose=verbose)
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


def _parse_solvent_options(solvent_options):
    if solvent_options is None:
        return None, {}

    options = dict(solvent_options)
    model = options.pop('model', None)
    model_alias = options.pop('solvent_model', None)
    if model is None:
        model = model_alias
    elif model_alias is not None and model.lower() != model_alias.lower():
        raise ValueError('model and solvent_model specify different solvents')

    if model is None:
        model = 'pcm'
    return model.lower(), options


def apply_solvent(mf, solvent_options=None):
    model, options = _parse_solvent_options(solvent_options)
    if model is None:
        return mf

    if model == 'pcm':
        mf = mf.PCM()
    elif model == 'smd':
        mf = mf.SMD()
    else:
        raise ValueError("solvent model must be None, 'pcm', or 'smd'")

    for key, value in options.items():
        setattr(mf.with_solvent, key, value)
    mf.with_solvent.equilibrium_solvation = True
    return mf


def make_rks(mol, functional, solvent_options=None):
    mf = scf.RKS(mol)
    mf.xc = functional
    return apply_solvent(mf, solvent_options)


def energy_with_charges(mol0, charges, cache, functional,
                        solvent_options=None):
    key = tuple(np.round(charges, 12))
    if key not in cache:
        mol1 = fractional_charge_mol(mol0, charges)
        mf1 = make_rks(mol1, functional, solvent_options=solvent_options)
        mf1.init_guess = '1e'
        mf1.conv_tol = 1e-12
        mf1.max_cycle = 100
        mf1.kernel()
        if not mf1.converged:
            raise RuntimeError('finite-difference SCF did not converge')
        cache[key] = mf1.e_tot
    return cache[key]


def finite_difference_derivatives(mol0, functional, solvent_options=None,
                                  step=1e-4, step3=2e-3, third_order=True):
    charges = mol0.atom_charges().astype(float)
    natm = mol0.natm
    cache = {}
    e0 = energy_with_charges(
        mol0, charges, cache, functional, solvent_options=solvent_options)
    fd1 = np.zeros(natm)
    fd2 = np.zeros((natm, natm))
    fd3 = np.zeros((natm, natm, natm))

    for ia in range(natm):
        zp = charges.copy()
        zm = charges.copy()
        zp[ia] += step
        zm[ia] -= step
        ep = energy_with_charges(
            mol0, zp, cache, functional, solvent_options=solvent_options)
        em = energy_with_charges(
            mol0, zm, cache, functional, solvent_options=solvent_options)
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
                mol0, zpp, cache, functional, solvent_options=solvent_options)
            epm = energy_with_charges(
                mol0, zpm, cache, functional, solvent_options=solvent_options)
            emp = energy_with_charges(
                mol0, zmp, cache, functional, solvent_options=solvent_options)
            emm = energy_with_charges(
                mol0, zmm, cache, functional, solvent_options=solvent_options)
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
                    solvent_options=solvent_options))
        fd3[ia, ib, ic] = value / (8.0 * step3 ** 3)

    return fd1, fd2, fd3


if __name__ == '__main__':
    atom = '''
    O  0.000000  0.000000  0.000000
    H  0.000000 -0.757160  0.586260
    H  0.000000  0.757160  0.586260
    '''
    functional = 'wb97xd'
    basis = '6-311++g**'
    # solvent_options = None
    solvent_options = {'model': 'pcm', 'eps': 78.3553}
    # solvent_options = {'model': 'smd', 'solvent': 'water'}

    third_order = True

    mol = gto.M(
        atom=atom,
        basis=basis,
        unit='Angstrom',
        verbose=0,
    )
    mf = Alchem(mol)
    mf.xc = functional
    mf = apply_solvent(mf, solvent_options)
    mf.run(conv_tol=1e-12)
    print('SCF energy:', mf.e_tot)

    d1, d2, d3 = mf.alchemical_derivatives(
        tol=1e-12, third_order=third_order)

    print_matrix('dE/dZ:', d1)
    print_matrix('dE/dZ electronic:', d1.electronic)
    print_matrix('dE/dZ nuclear:', d1.nuclear)
    print_matrix('d2E/dZdZ:', d2)
    print_matrix('d2E/dZdZ electronic:', d2.electronic)
    print_matrix('d2E/dZdZ nuclear:', d2.nuclear)
    print_matrix('Eq. 32 d3E/dZdZdZ:', d3)

    fd1, fd2, fd3 = finite_difference_derivatives(
        mol, functional, solvent_options=solvent_options,
        third_order=third_order)
    #print_matrix('finite-difference dE/dZ:', fd1)
    print('max |analytic - finite-difference| dE/dZ:',
          np.max(np.abs(d1 - fd1)))
    #print_matrix('finite-difference d2E/dZdZ:', fd2)
    print('max |analytic - finite-difference| d2E/dZdZ:',
          np.max(np.abs(d2 - fd2)))
    #print_matrix('finite-difference d3E/dZdZdZ:', fd3)
    print('max |analytic - finite-difference| d3E/dZdZdZ:',
          np.max(np.abs(d3 - fd3)))
