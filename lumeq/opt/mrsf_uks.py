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

from lumeq.opt.mrsf_roks import mrsf_dimension_transform
from lumeq.opt.mrsf_roks import _gen_rhf_response

"""
Multi-reference spin-flip (MRSF) DFT uses higher spin reference state
and calculates the real ground-state and excited-states with spin flipping.
references from Cheol Ho Choi
JCP 2019. 10.1063/1.5086895
JCTC 2021. 10.1021/acs.jctc.0c01074
JPCA 2024. 10.1021/acs.jpca.4c04521
"""

# based on scf/_response_function.py
def _gen_uhf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, max_memory=None):
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
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        # mf might be pbc.dft.RKS object with multigrid
        if not hybrid and isinstance(getattr(mf, 'with_df', None), multigrid.MultiGridFFTDF):
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_uhf_response(mf, dm0, with_j, hermi)

        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, 1)
        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        def vind(dm1):
            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                v1 = ni.nr_uks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                   rho0, vxc, fxc, max_memory=max_memory)
            if not hybrid:
                if with_j:
                    vj = mf.get_j(mol, dm1, hermi=hermi)
                    #v1 += vj[0] + vj[1]
            else:
                if omega == 0:
                    vj, vk = mf.get_jk(mol, dm1, hermi, with_j=with_j)
                    vk *= hyb
                elif alpha == 0: # LR=0, only SR exchange
                    if with_j:
                        vj = mf.get_j(mol, dm1, hermi)
                    vk = mf.get_k(mol, dm1, hermi, omega=-omega)
                    vk *= hyb
                elif hyb == 0: # SR=0, only LR exchange
                    if with_j:
                        vj = mf.get_j(mol, dm1, hermi)
                    vk = mf.get_k(mol, dm1, hermi, omega=omega)
                    vk *= alpha
                else: # SR and LR exchange with different ratios
                    vj, vk = mf.get_jk(mol, dm1, hermi, with_j=with_j)
                    vk *= hyb
                    vk += mf.get_k(mol, dm1, hermi, omega=omega) * (alpha-hyb)
                #if with_j:
                #    v1 += vj[0] + vj[1] - vk
                #else:
                #    v1 -= vk
            #return v1
            return vj, -vk

    elif with_j:
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            #v1 = vj[0] + vj[1] - vk
            #return v1
            return vj, -vk

    else:
        def vind(dm1):
            return -mf.get_k(mol, dm1, hermi=hermi)

    return vind


# for MRSF adopted from tdscf/rhf.py and tdscf/uhf.py
def gen_tda_operation(mf, fock_ao=None, singlet=True, wfnsym=None):
    r'''A x

    Kwargs:
        wfnsym : int or str
            Point group symmetry irrep symbol or ID for excited CIS wavefunction.
    '''
    assert isinstance(mf, uhf.UHF)

    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert (mo_coeff[0].dtype == numpy.double)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff[0].shape
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]  # not used except for noccb
    viridxa = numpy.where(mo_occ[0]==0)[0] # not used except for nvira
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbvb = mo_coeff[1][:,viridxb]
    orbob = mo_coeff[1][:,occidxa] # spin-image
    orbva = mo_coeff[0][:,viridxb]

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        x_sym_a, x_sym_b = _get_x_sym_table(mf)
        sym_forbid = numpy.append(x_sym_a.ravel(), x_sym_b.ravel()) != wfnsym

    # beta virtual energy - alpha occupied energy
    fock_mo = lib.einsum('imp,imn,inq->ipq', mo_coeff.conj(), fock_ao, mo_coeff)
    e_ia = hdiag = numpy.diag(fock_mo[1])[viridxb] - numpy.diag(fock_mo[0])[occidxa,None]
    #e_ia = hdiag = mo_energy[1][viridxb] - mo_energy[0][occidxa,None]
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = e_ia.ravel()

    # it seems using pointers so don't move it above e_ia!!!
    # otherwise one needs to use numpy.copy()
    Fa_o, Fb_v = fock_mo[0, :nocca, :nocca], fock_mo[1, noccb:, noccb:]
    # remove diagonal elements for 1e cotributions in vind()
    Fa_o -= numpy.diag(numpy.diag(Fa_o))
    Fb_v -= numpy.diag(numpy.diag(Fb_v))

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.8-mem_now)
    #vresp = _gen_uhf_response(mf, hermi=0, max_memory=max_memory)
    vresp = _gen_rhf_response(mf, hermi=0, max_memory=max_memory)

    # transform matrix
    sign = 1 if singlet else -1 # scale integrals Eq. 2.13
    U, Ut = mrsf_dimension_transform([nocca, noccb], [nvira, nvirb], nao,
                                     singlet=singlet)
    idx_ab = [[0, 6], [0, 5]] # [left, right] transform indices
    idx_ba = [[5], [6]] # [left, right] transform indices
    idx_aa = [1, 2]
    idx_bb = [3, 4]

    def vind(zs):
        nz = len(zs)
        zs = numpy.asarray(zs)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0

        # transform amplitudes
        zs = zs.reshape(nz,nocca,nvirb)
        zs_new = lib.einsum('kov,xov->kxov', Ut, zs)
        zs_new[5,:,:, [0, 1]] = zs_new[5,:,:, [1, 0]] # swap columns of virtual O1 and O2 for UCO1CO2
        zs_new[6,:, [noccb, noccb+1]] = zs_new[6,:, [noccb+1, noccb]] # swap rows of occupied O1 and O2 for UO1VO2V
        dms = numpy.zeros((7, nz, nao, nao), dtype=zs_new.dtype)
        dms[idx_ab[0]] = lib.einsum('kxov,pv,qo->kxpq', zs_new[idx_ab[0]], orbvb, orboa.conj())
        dms[idx_ba[0]] = lib.einsum('kxov,pv,qo->kxpq', zs_new[idx_ba[0]], orbva, orbob.conj())
        dms[idx_aa] = lib.einsum('kxov,pv,qo->kxpq', zs_new[idx_aa], orbva, orboa.conj())
        dms[idx_bb] = lib.einsum('kxov,pv,qo->kxpq', zs_new[idx_bb], orbvb, orbob.conj())
        dms[5:] = dms[5:].transpose(0,1,3,2) # use transpose for the last two dms
        dms = dms.reshape(-1, nao, nao)

        vj, vk = vresp(dms) # vk has included -1
        vj = vj.reshape((7, nz, nao, nao))
        vk = vk.reshape((7, nz, nao, nao))
        # Eq. 2.14
        vk[1:5] = sign * (vj[1:5] + vk[1:5])
        vk[5:] = -sign * vk[5:]
        v1ao = vk

        v1mo = numpy.zeros_like(zs_new, dtype=zs_new.dtype)
        v1mo[idx_ab[1]] = lib.einsum('kxpq,qo,pv->kxov', v1ao[idx_ab[1]], orboa, orbvb.conj())
        v1mo[idx_ba[1]] = lib.einsum('kxpq,qo,pv->kxov', v1ao[idx_ba[1]], orbob, orbva.conj())
        v1mo[idx_aa] = lib.einsum('kxpq,qo,pv->kxov', v1ao[idx_aa], orboa, orbva.conj())
        v1mo[idx_bb] = lib.einsum('kxpq,qo,pv->kxov', v1ao[idx_bb], orbob, orbvb.conj())
        v1mo[5,:,:, [0, 1]] = v1mo[5,:,:, [1, 0]] # swap columns back
        v1mo[6,:, [noccb, noccb+1]] = v1mo[6,:, [noccb+1, noccb]] # swap rows back
        v1mo = lib.einsum('kov,kxov->xov', U, v1mo) # sum over the contributions Eq. 2.15

        # add one-electron terms for alpha-beta interaction
        zs = zs_new[0]
        v1mo += lib.einsum('xia,ia->xia', zs, e_ia)
        v1mo += lib.einsum('xia,ab->xib', zs, Fb_v)
        v1mo -= lib.einsum('xia,ij->xja', zs, Fa_o)

        return v1mo.reshape(v1mo.shape[0],-1)

    return vind, hdiag
gen_tda_hop = gen_tda_operation


# based on tdscf/rhf.py
def analyze(tdobj, verbose=None):
    log = logger.new_logger(tdobj, verbose)
    mol = tdobj.mol
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    nocca = numpy.count_nonzero(mo_occ[0] == 1)
    noccb = numpy.count_nonzero(mo_occ[1] == 1)

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
    orbo = mo_coeff[0][:,mo_occ[0]==1]
    orbv = mo_coeff[1][:,mo_occ[1]==0]

    #Incompatible to old numpy version
    #ints = numpy.einsum('...pq,pi,qj->...ij', ints, orbo, orbv.conj())
    ints = lib.einsum('xpq,pi,qj->xij', ints.reshape(-1,nao,nao), orbo, orbv.conj())
    pol = numpy.array([numpy.einsum('xij,ij->x', ints, x) * 2 for x,y in xy])
    if isinstance(xy[0][1], numpy.ndarray):
        if hermi:
            pol += [numpy.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
        else:  # anti-Hermitian
            pol -= [numpy.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
    pol = pol.reshape((nstates,)+pol_shape)
    return pol



class MRSF_TDA(tdscf.uks.TDA):

    singlet = True
    positive_eig_threshold = -0.3 # keep ground-state

    def gen_vind(self, mf=None):
        r'''Generate function to compute Ax'''
        if mf is None:
            mf = self._scf
        fock_ao = mf.get_fock()
        return gen_tda_hop(mf, fock_ao, singlet=self.singlet, wfnsym=self.wfnsym)

    def init_guess(self, mf, nstates=None, wfnsym=None, return_symmetry=False):
        if nstates is None: nstates = self.nstates
        if wfnsym is None: wfnsym = self.wfnsym

        mol = mf.mol
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        occidxa = numpy.where(mo_occ[0]>0)[0]
        occidxb = numpy.where(mo_occ[1]>0)[0]
        viridxa = numpy.where(mo_occ[0]==0)[0]
        viridxb = numpy.where(mo_occ[1]==0)[0]
        e_ia = (mo_energy[1][viridxb] - mo_energy[0][occidxa,None]).ravel()
        nova = e_ia.size
        nstates = min(nstates, nova)

        # Find the nstates-th lowest energy gap
        e_threshold = numpy.partition(e_ia, nstates-1)[nstates-1]
        e_threshold += self.deg_eia_thresh

        print_matrix('e_ia:', e_ia[e_ia<=e_threshold])
        idx = numpy.where(e_ia <= e_threshold)[0]
        x0 = numpy.zeros((idx.size, nova))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations

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
            idx = numpy.where(w > self.positive_eig_threshold)[0]
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

        nmo = self._scf.mo_occ[0].size
        nocca = (self._scf.mo_occ[0]>0).sum()
        noccb = (self._scf.mo_occ[1]>0).sum()
        nvira = nmo - nocca
        nvirb = nmo - noccb
        self.xy = [(xi.reshape(nocca,nvirb),  # X_alpha_to_beta
                    0)  # (Y_alpha_to_beta
                   for xi in x1]

        print_matrix('amplitudes:', x1.reshape(-1,nocca,nvirb).transpose(0,2,1), nind=1)
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



def convert_roks_to_uks(mf):
    r'''Convert ROKS object to UKS object for TDDFT calculation.'''
    from pyscf import dft, scf
    assert isinstance(mf, dft.roks.ROKS)

    mf_uks = scf.UKS(mf.mol)
    mf_uks.xc = mf.xc
    mf_uks.grids = mf.grids
    mf_uks.conv_tol = mf.conv_tol
    mf_uks.max_cycle = mf.max_cycle

    mf_uks.mo_coeff = numpy.array([mf.mo_coeff, mf.mo_coeff])
    mf_uks.mo_energy = numpy.array([mf.mo_energy.mo_ea, mf.mo_energy.mo_eb])
    mo_occ = numpy.zeros((2, len(mf.mo_occ)), dtype=mf.mo_occ.dtype)
    for i, occ in enumerate(mf.mo_occ):
        if occ > 0:
            mo_occ[0, i] = 1
        if occ == 2:
            mo_occ[1, i] = 1
    mf_uks.mo_occ = mo_occ

    mf_uks.e_tot = mf.e_tot
    return mf_uks



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

    spin = 2
    basis = '6-31g'
    functional = 'hf' #'bhandhlyp'
    nstates = 5

    mol = gto.M(
            atom = atom,
            spin = spin,
            basis = basis,
            )

    unrestricted = False
    method = 'UKS' if unrestricted else 'ROKS'
    mf = getattr(scf, method)(mol)
    mf.xc = functional
    e0 = mf.kernel()

    if not unrestricted:
        mf = convert_roks_to_uks(mf)

    td = MRSF_TDA(mf)
    td.nstates = nstates
    td.verbose = 4
    e, xys = td.kernel()
    td.analyze()
