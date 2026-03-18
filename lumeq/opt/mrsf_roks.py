from lumeq import os, sys
from lumeq.utils import print_matrix

import numpy
from pyscf import scf, tdscf, gto, lib
from pyscf.lib import logger
from pyscf.tdscf._lr_eig import eigh as lr_eigh, eig as lr_eig, real_eig
from pyscf.scf import hf, rohf, uhf
from pyscf.data import nist
from pyscf import __config__

MO_BASE = getattr(__config__, 'MO_BASE', 1)

"""
Multi-reference spin-flip (MRSF) DFT uses higher spin reference state
and calculates the real ground-state and excited-states with spin flipping.
references from Cheol Ho Choi
JCP 2019. 10.1063/1.5086895
JCTC 2021. 10.1021/acs.jctc.0c01074
JPCA 2024. 10.1021/acs.jpca.4c04521
"""

def mrsf_dimension_transform(nocc, nvir, nbas, singlet=True):
    r"""Build the MRSF amplitude-dimension transform matrices.

    The returned matrices reorder and combine spin-flip excitation blocks for
    the singlet or triplet MRSF working equations.

    Args:
        nocc: Numbers of alpha and beta occupied orbitals.
        nvir: Numbers of alpha and beta virtual orbitals.
        nbas (int): Number of AO basis functions.
        singlet (bool): Whether to calculate the singlet (`True`) or triplet (`False`).

    Returns:
        U (numpy.ndarray): Transform matrix for singlet and triplet excited states.
            The rows collect the relevant excitation blocks for the bra side.
        Ut (numpy.ndarray): Reordered transform matrix for the ket side.
            This matrix is the reordered partner used on the ket side.

    Notes:
        Reference: JCP 2019, 10.1063/1.5086895.

        The spin-pairing structure of the A matrix follows Fig. 1:

        .. code-block:: text

            [U S  ] [alpha->beta,  alpha->beta ] [U S]
            [UCO  ] [alpha->alpha, alpha->alpha] [UOV]
            [UOV  ] [beta ->beta,  beta ->beta ] [UCO]
            [UCOCO] [alpha->beta,  beta ->alpha] [UCOCO]
            [UOVOV] [beta ->alpha, alpha->beta ] [UOVOV]
    """
    nocca, noccb = nocc
    nvira, nvirb = nvir
    #print('nocca:', nocca, 'noccb:', noccb, 'nvira:', nvira, 'nvirb:', nvirb)
    nova = novb = nocca * nvirb # excitation space dimension
    nov = nova * 2 # maximum number of roots

    U = numpy.zeros((7, nocca, nvirb)) # dimention transformation
    U[1, :noccb, 0]   = 1. # UCO1  # C->O excitation Eqs. 2.6a and 2.7a
    U[2, :noccb, 1]   = 1. # UCO2
    U[3, noccb, 2:]   = 1. # UO1V  # O->V excitation Eqs. 2.6b and 2.7b
    U[4, noccb+1, 2:] = 1. # UO2V
    U[5] = U[1] - U[2]   # UCO1CO2 where O is unoccupied
    U[6] = U[3] - U[4]   # UO1VO2V where O is occupied

    # CV + G + D + OO
    # Eqs. 2.4, 2.5, and 2.6c in JCP 2019.
    U[0] = numpy.ones((nocca, nvirb))
    #if singlet:
    #    U[0, noccb, 0]   = 1.
    #    U[0, noccb+1, 1] = -1.
    #else: # triplet
    #    U[0, noccb, 0]   = 1.
    #    U[0, noccb, 1]   = 0.
    #    U[0, noccb+1, 0] = 0.
    #    U[0, noccb+1, 1] = 1.

    Ut = numpy.array((U[0], U[4], U[3], U[2], U[1], U[5], U[6])) # reorder U
    #U, Ut = U[1:], Ut[1:]  # remove CV row
    return U, Ut


# based on scf/_response_function.py
def _gen_rhf_response(mf, mo_coeff=None, mo_occ=None,
                      singlet=True, hermi=0, max_memory=None):
    r'''Generate a function to compute the product of UHF response function and
    UHF density matrices.
    '''
    assert isinstance(mf, (rohf.ROHF, uhf.UHF))
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if isinstance(mf, hf.KohnShamDFT):
        from pyscf.pbc.dft import multigrid
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if mf.do_nlc():
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'derivative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        #print('functional:', mf.xc, 'omega:', omega, 'alpha:', alpha, 'hyb:', hyb)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        # mf might be pbc.dft.RKS object with multigrid
        if not hybrid and isinstance(getattr(mf, 'with_df', None), multigrid.MultiGridFFTDF):
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_rhf_response(mf, dm0, singlet, hermi)

        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, 1)
        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        fxc *= .5
        def vind(dm1):
            #if hermi == 2:
            #    v1 = numpy.zeros_like(dm1)
            #else:
            #    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
            #    v1 = ni.nr_rks_fxc_st(mol, mf.grids, mf.xc, dm0, dm1, 0, True,
            #                       rho0, vxc, fxc, max_memory=max_memory)
            if hybrid:
                if omega == 0:
                    vj, vk = mf.get_jk(mol, dm1, hermi)
                    vj *= hyb
                    vk *= hyb
                elif alpha == 0: # LR=0, only SR exchange
                    vj = mf.get_j(mol, dm1, hermi)
                    vk = mf.get_k(mol, dm1, hermi, omega=-omega)
                    vj *= hyb
                    vk *= hyb
                elif hyb == 0: # SR=0, only LR exchange
                    vj = mf.get_j(mol, dm1, hermi)
                    vk = mf.get_k(mol, dm1, hermi, omega=omega)
                    vj *= alpha
                    vk *= alpha
                else: # SR and LR exchange with different ratios
                    vj, vk = mf.get_jk(mol, dm1, hermi)
                    vj *= hyb
                    vk *= hyb
                    if alpha-hyb != 0:
                        vj2, vk2 = mf.get_jk(mol, dm1, hermi, omega=omega) * (alpha-hyb)
                        vj += vj2
                        vk += vk2
            elif hermi != 2:
                vj, vk = numpy.zeros_like(dm1), numpy.zeros_like(dm1)
            return vj, -vk

    elif hermi != 2:
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            return vj, -vk

    else:
        def vind(dm1):
            return -mf.get_k(mol, dm1, hermi=hermi)

    return vind


# for MRSF adopted from tdscf/rhf.py
def gen_tda_operation(mf, fock_ao=None, singlet=True, wfnsym=None):
    r'''A x

    Kwargs:
        wfnsym : int or str
            Point group symmetry irrep symbol or ID for excited CIS wavefunction.
    '''
    assert isinstance(mf, rohf.ROHF)

    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert (mo_coeff.dtype == numpy.double)
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidxa = mo_occ > 0
    occidxb = mo_occ == 2
    viridxa = ~occidxa
    viridxb = ~occidxb
    orboa = mo_coeff[:,occidxa]
    orbob = mo_coeff[:,occidxb]
    orbva = mo_coeff[:,viridxa]
    orbvb = mo_coeff[:,viridxb]
    nocca = orboa.shape[1]
    noccb = orbob.shape[1]
    nvira = orbva.shape[1]
    nvirb = orbvb.shape[1]

    # transform matrix
    sign = 1 if singlet else -1 # scale integrals Eq. 2.13
    U, Ut = mrsf_dimension_transform([nocca, noccb], [nvira, nvirb], nao)

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        x_sym_a, x_sym_b = _get_x_sym_table(mf)
        sym_forbid = numpy.append(x_sym_a.ravel(), x_sym_b.ravel()) != wfnsym

    fock_mo = lib.einsum('mp,imn,nq->ipq', mo_coeff.conj(), fock_ao, mo_coeff)
    Fa_o, Fb_v = fock_mo[0][numpy.ix_(occidxa,occidxa)], fock_mo[1][numpy.ix_(viridxb,viridxb)]
    # beta virtual energy - alpha occupied energy
    e_ia = hdiag = numpy.diag(Fb_v) - numpy.diag(Fa_o)[:,None]
    # numerically there is no difference after averaging O1O1 and O2O2 transitions
    #e_ia[noccb,0] = e_ia[noccb+1,1] = (e_ia[noccb,0] + e_ia[noccb+1,1]) / 2.
    #print_matrix('e_ia:', e_ia, nind=1)
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = e_ia.ravel()

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.8-mem_now)
    vresp = _gen_rhf_response(mf, hermi=0, max_memory=max_memory)

    def vind(zs):
        nz = len(zs)
        zs = numpy.asarray(zs)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0

        # transform amplitudes
        zs = zs.reshape(nz, nocca, nvirb)
        zs_new = lib.einsum('kov,xov->kxov', Ut, zs)
        zs_new[0,:,noccb,0] /= numpy.sqrt(2.)
        if singlet:
            zs_new[0,:,noccb+1,1] = -zs_new[0,:,noccb,0]
        else: # triplet
            zs_new[0,:,noccb,1] = 0.
            zs_new[0,:,noccb+1,0] = 0.
            zs_new[0,:,noccb+1,1] = zs_new[0,:,noccb,0]
        zs_new[5,:,:, [0, 1]] = zs_new[5,:,:, [1, 0]] # swap columns of virtual O1 and O2 for UCO1CO2
        zs_new[6,:, [noccb, noccb+1]] = zs_new[6,:, [noccb+1, noccb]] # swap rows of occupied O1 and O2 for UO1VO2V
        dms = lib.einsum('kxov,pv,qo->kxpq', zs_new, orbvb, orboa.conj())
        dms[5:] = dms[5:].transpose(0,1,3,2) # use transpose for the last two dms

        vj, vk = vresp(dms.reshape(-1, nao, nao)) # vk has included -1
        vj = vj.reshape((7, nz, nao, nao))
        vk = vk.reshape((7, nz, nao, nao))
        # Eq. 2.14
        vk[1:5] = sign * (vj[1:5] + vk[1:5])
        vk[5:] = -sign * vk[5:]
        v1ao = vk

        v1mo = lib.einsum('kxpq,qo,pv->kxov', v1ao, orboa, orbvb.conj())

        # add one-electron terms
        zs = zs_new[0]
        v1mo[0] += lib.einsum('xib,ab->xia', zs, Fb_v)
        v1mo[0] -= lib.einsum('xja,ji->xia', zs, Fa_o)

        if singlet:
            v1mo[0,:,noccb,0] = (v1mo[0,:,noccb,0] - v1mo[0,:,noccb+1,1]) / numpy.sqrt(2.)
            v1mo[0,:,noccb+1,1] = 0.
        else: # triplet
            v1mo[0,:,noccb,0] = (v1mo[0,:,noccb,0] + v1mo[0,:,noccb+1,1]) / numpy.sqrt(2.)
            v1mo[0,:,noccb,1] = 0.
            v1mo[0,:,noccb+1,0] = 0.
            v1mo[0,:,noccb+1,1] = 0.

        v1mo[5,:,:, [0, 1]] = v1mo[5,:,:, [1, 0]] # swap columns back
        v1mo[6,:, [noccb, noccb+1]] = v1mo[6,:, [noccb+1, noccb]] # swap rows back
        v1mo = lib.einsum('kov,kxov->xov', U, v1mo)  # sum over the contributions

        return v1mo.reshape(v1mo.shape[0],-1)

    return vind, hdiag
gen_tda_hop = gen_tda_operation


# based on tdscf/rhf.py
def analyze(tdobj, verbose=None):
    log = logger.new_logger(tdobj, verbose)
    mol = tdobj.mol
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    nocca = numpy.count_nonzero(mo_occ > 0)
    noccb = numpy.count_nonzero(mo_occ == 2)

    e_ev = numpy.asarray(tdobj.e) * nist.HARTREE2EV
    e_wn = numpy.asarray(tdobj.e) * nist.HARTREE2WAVENUMBER
    wave_length = 1e7/e_wn

    if tdobj.singlet:
        log.note('\n** Singlet excitation energies and oscillator strengths **')
    else:
        log.note('\n** Triplet excitation energies and oscillator strengths **')

    if mol.symmetry:
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        x_sym = symm.direct_prod(orbsym[mo_occ==2], orbsym[mo_occ==0], mol.groupname)
    else:
        x_sym = None

    f_oscillator = tdobj.oscillator_strength()
    for i, ei in enumerate(tdobj.e):
        x, y = tdobj.xy[i]
        if x_sym is None:
            log.note('Excited State %3d: %12.5f eV %9.2f nm  f=%.4f',
                     i+1, e_ev[i], wave_length[i], f_oscillator[i])
        else:
            wfnsym = _analyze_wfnsym(tdobj, x_sym, x)
            log.note('Excited State %3d: %4s %12.5f eV %9.2f nm  f=%.4f',
                     i+1, wfnsym, e_ev[i], wave_length[i], f_oscillator[i])

        if log.verbose >= logger.INFO:
            o_idx, v_idx = numpy.where(abs(x) > 0.1)
            for o, v in zip(o_idx, v_idx):
                log.info('    %4d -> %-4d %12.5f',
                         o+MO_BASE, v+MO_BASE+noccb, x[o,v])

    if log.verbose >= logger.INFO:
        log.info('\n** Transition electric dipole moments (AU) **')
        log.info('state          X           Y           Z        Dip. S.      Osc.')
        trans_dip = tdobj.transition_dipole()
        for i, ei in enumerate(tdobj.e):
            dip = trans_dip[i]
            log.info('%3d    %11.4f %11.4f %11.4f %11.4f %11.4f',
                     i+1, dip[0], dip[1], dip[2], numpy.dot(dip, dip),
                     f_oscillator[i])

        log.info('\n** Transition velocity dipole moments (imaginary part, AU) **')
        log.info('state          X           Y           Z        Dip. S.      Osc.')
        trans_v = tdobj.transition_velocity_dipole()
        f_v = tdobj.oscillator_strength(gauge='velocity', order=0)
        for i, ei in enumerate(tdobj.e):
            v = trans_v[i]
            log.info('%3d    %11.4f %11.4f %11.4f %11.4f %11.4f',
                     i+1, v[0], v[1], v[2], numpy.dot(v, v), f_v[i])

        log.info('\n** Transition magnetic dipole moments (imaginary part, AU) **')
        log.info('state          X           Y           Z')
        trans_m = tdobj.transition_magnetic_dipole()
        for i, ei in enumerate(tdobj.e):
            m = trans_m[i]
            log.info('%3d    %11.4f %11.4f %11.4f',
                     i+1, m[0], m[1], m[2])
    return tdobj


# based on tdscf/rhf.py
# change molecular orbitals
def _contract_multipole(tdobj, ints, hermi=True, xy=None):
    r'''ints is the integral tensor of a spin-independent operator'''
    if xy is None: xy = tdobj.xy
    nstates = len(xy)
    pol_shape = ints.shape[:-2]
    nao = ints.shape[-1]

    if not tdobj.singlet:
        return numpy.zeros((nstates,) + pol_shape)

    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    orbo = mo_coeff[:,mo_occ>0]
    orbv = mo_coeff[:,mo_occ<2]

    #Incompatible to old numpy version
    #ints = numpy.einsum('...pq,pi,qj->...ij', ints, orbo, orbv.conj())
    ints = lib.einsum('xpq,pi,qj->xij', ints.reshape(-1,nao,nao), orbo, orbv.conj())
    pol = numpy.array([numpy.einsum('xij,ij->x', ints, x) for x,y in xy])
    if isinstance(xy[0][1], numpy.ndarray):
        if hermi:
            pol += [numpy.einsum('xij,ij->x', ints, y) for x,y in xy]
        else:  # anti-Hermitian
            pol -= [numpy.einsum('xij,ij->x', ints, y) for x,y in xy]
    pol = pol.reshape((nstates,)+pol_shape)
    return pol


def spin_square(tdobj):
    r"""
    Calculate <S^2> expectation value for each excited state.
    S^2 = Sz^2 + 0.5 * (S+ S- + S- S+)
    """
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    nocca = numpy.count_nonzero(mo_occ > 0)
    noccb = numpy.count_nonzero(mo_occ == 2)
    nmo = mo_coeff.shape[1]

    ovlp = tdobj._scf.get_ovlp()


    ss = []
    for x, y in tdobj.xy:
        s_plus = lib.einsum('ia,ja->ij', x, x)
        s_minus = lib.einsum('ia,ja->ij', y, y)
        s_z = 0.5 * (lib.einsum('ia,ia->', x, x) - lib.einsum('ia,ia->', y, y))
        s2 = (s_z * (s_z + 1)
              + lib.trace(s_plus @ s_minus))
        ss.append(s2)
    return numpy.array(ss)



class MRSF_TDA(tdscf.rks.TDA):

    singlet = True
    #positive_eig_threshold = -0.3 # keep ground-state

    def gen_vind(self, mf=None):
        r'''Generate function to compute Ax'''
        if mf is None:
            mf = self._scf
        fock_ao = mf.get_fock()
        fock_ao = numpy.array([fock_ao.focka, fock_ao.fockb])
        return gen_tda_hop(mf, fock_ao, singlet=self.singlet, wfnsym=self.wfnsym)

    def init_guess(self, mf, nstates=None, wfnsym=None, return_symmetry=False):
        if nstates is None: nstates = self.nstates
        if wfnsym is None: wfnsym = self.wfnsym

        mol = mf.mol
        mo_ea = mf.mo_energy.mo_ea
        mo_eb = mf.mo_energy.mo_eb
        mo_occ = mf.mo_occ
        occidxa = mo_occ > 0
        occidxb = mo_occ == 2
        viridxa = ~occidxa
        viridxb = ~occidxb
        e_ia = (mo_eb[viridxb] - mo_ea[occidxa,None]).ravel()
        nova = e_ia.size
        noccb = len(mo_eb[occidxb])
        nvirb = len(mo_eb[viridxb])
        nstates = min(nstates, nova)

        # Find the nstates-th lowest energy gap
        # add one more state (rather than using nstates-1)
        extra = 0 if self.singlet else 2
        e_threshold = numpy.partition(e_ia, nstates+extra)[nstates+extra]
        e_threshold += self.deg_eia_thresh

        #print_matrix('e_ia:', e_ia[e_ia<=e_threshold])
        idx = numpy.where(e_ia <= e_threshold)[0]
        x0 = numpy.zeros((idx.size, nova))
        skip = False
        for i, j in enumerate(idx):
            if j != noccb*nvirb and j != (noccb+1)*nvirb+1:
                x0[i, j] = 1  # Koopmans' excitations
            elif j == noccb*nvirb and not skip: # O1O1 excitation
                # forced to be same as O1O1 excitation later
                #x0[i, (noccb+1)*nvirb+1] = 1.
                x0[i, noccb*nvirb] = 1.
                skip = True
            elif j == (noccb+1)*nvirb+1 and not skip: # O2O2 excitation
                # forced to be same as O1O1 excitation later
                #x0[i, (noccb+1)*nvirb+1] = 1.
                x0[i, noccb*nvirb] = 1.
                skip = True
        x0 = x0[~(x0 == 0).all(axis=1)]  # remove zero columns

        if return_symmetry:
            if mf.mol.symmetry:
                x_sym = numpy.append(x_sym_a.ravel(), x_sym_b.ravel())
                x0sym = x_sym[idx]
            else:
                x0sym = None
            return x0, x0sym
        else:
            return x0

    def kernel(self, x0=None, nstates=None):
        r'''TDA diagonalization solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        log = logger.Logger(self.stdout, self.verbose)
        mol = self.mol

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        def pickeig(w, v, nroots, envs):
            #idx = numpy.where(w > self.positive_eig_threshold)[0]
            idx = numpy.where(numpy.abs(w) > 1e-5)[0]
            #idx = numpy.argsort(w)[:nroots]
            return w[idx], v[:,idx], idx

        x0sym = None
        if x0 is None:
            x0, x0sym = self.init_guess(
                self._scf, self.nstates, return_symmetry=True)
        elif mol.symmetry:
            x_sym_a, x_sym_b = _get_x_sym_table(self._scf)
            x_sym = numpy.append(x_sym_a.ravel(), x_sym_b.ravel())
            x0sym = [rhf._guess_wfnsym_id(self, x_sym, x) for x in x0]

        self.converged, self.e, x1 = lr_eigh(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        nmo = self._scf.mo_occ.size
        nocca = (self._scf.mo_occ>0).sum()
        noccb = (self._scf.mo_occ==2).sum()
        nvira = nmo - nocca
        nvirb = nmo - noccb

        # average the O1 and O2 amplitudes
        x1 = x1.reshape(-1, nocca, nvirb)
        x1[:,noccb,0] /= numpy.sqrt(2.)
        if self.singlet:
            x1[:,noccb+1,1] = -x1[:,noccb,0]
        else:  # triplet
            x1[:,noccb,1] = 0.
            x1[:,noccb+1,0] = 0.
            x1[:,noccb+1,1] = x1[:,noccb,0]

        self.xy = [(xi.reshape(nocca,nvirb),  # X_alpha_to_beta
                    0)  # (Y_alpha_to_beta
                   for xi in x1]

        #print_matrix('amplitudes:', x1.reshape(-1,nocca,nvirb).transpose(0,2,1), nind=1)
        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDA', *cpu0)
        self._finalize()
        return self.e, self.xy

    analyze = analyze
    _contract_multipole = _contract_multipole

    to_gpu = lib.to_gpu

MRSF_CIS = MRSF_TDA



if __name__ == '__main__':
    # ethylene tw-pyr CI
    #atom = """
    #C       -0.001716     0.029274    -0.037218
    #C        1.378613     0.007268    -0.008058
    #H        2.012945    -0.887831     0.020480
    #H        0.055158    -0.660513    -0.949800
    #H       -0.481163    -0.736532     0.584588
    #H        1.974826     0.910467    -0.129649
    #"""

    atom = """Be      0.000000      0.000000      0.000000"""
#    atom = """
#           O           0.000000    0.000000    0.1191992
#           H          -0.759081    0.000000   -0.4767968
#           H           0.759081    0.000000   -0.4767968
#    """

    spin = 2
    basis = '6-31g'
    functional = 'bhandhlyp'
    nstates = 12
    rpa = 0

    mol = gto.M(
            atom = atom,
            spin = spin,
            basis = basis,
            )

    mf = scf.ROKS(mol)
    mf.xc = functional
    e0 = mf.kernel()

    td = MRSF_TDA(mf)
    td.nstates = nstates
    td.verbose = 4
    e, xys = td.kernel()
    td.analyze()

    from lumeq.property import assemble_amplitudes, cal_rdm1, cal_dipoles
    dip_mat = mol.intor('int1e_r', comp=3)
    coeff = mf.mo_coeff
    xy = assemble_amplitudes(td.xy, rpa=rpa, itype='ro-sf')
    #rdm1 = cal_rdm1(xy, coeff, scale=1., itype='trans')
    #dipoles = cal_dipoles(dip_mat, rdm1)
    #print_matrix('transition dipoles:', dipoles)
    rdm1 = cal_rdm1(xy, coeff, scale=1., itype='diff')
    dipoles = cal_dipoles(dip_mat, rdm1)
    print_matrix('difference dipoles:', dipoles)

    mol = gto.M(
            atom = atom,
            spin = 0,
            basis = basis,
            )
    mf = scf.RKS(mol)
    mf.xc = functional
    e0 = mf.kernel()
    print('dip_moment:', mf.dip_moment(unit='AU'))
    td = tdscf.rks.TDA(mf)
    td.nstates = nstates
    td.verbose = 4
    e, xys = td.kernel()
    td.analyze()
