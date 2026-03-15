from lumeq import sys, np

# used in the _gen_rhf_response function below
from pyscf import lib, gto
from pyscf.lib import logger

# copied from scf/_response_functions.py
# the only difference is use original density for rho0, vxc, fxc
# replace scf.hf.RHF.gen_response when used for embedded mf
def _gen_rhf_response(mf, mo_coeff=None, mo_occ=None,
                      singlet=None, hermi=0, max_memory=None):
    from pyscf.scf import hf, rohf, uhf, ghf, dhf
    '''Generate a function to compute the product of RHF response function and
    RHF density matrices.

    Kwargs:
        singlet (None or boolean) : If singlet is None, response function for
            orbital hessian or CPHF will be generated. If singlet is boolean,
            it is used in TDDFT response kernel.
    '''
    assert (not isinstance(mf, (uhf.UHF, rohf.ROHF)))

    if mo_coeff is None: mo_coeff = mf.mo_coeff # embedded mo
    if mo_occ is None: mo_occ = mf.mo_occ       # embedded occ
    mo_coeff0 = mf.mo_coeff0             # original mo for dft
    mo_occ0 = mf.mo_occ0                # original occ for dft
    mol = mf.mol
    if isinstance(mf, hf.KohnShamDFT):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if mf.nlc or ni.libxc.is_nlc(mf.xc):
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        # mf can be pbc.dft.RKS object with multigrid
        if (not hybrid and
            'MultiGridFFTDF' == getattr(mf, 'with_df', None).__class__.__name__):
            from pyscf.pbc.dft import multigrid
            dm0 = mf.make_rdm1(mo_coeff0, mo_occ0)
            return multigrid._gen_rhf_response(mf, dm0, singlet, hermi)

        # use only singlet or triplet case, removed ground state hessian
        # use original mos
        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                            mo_coeff0, mo_occ0, spin=1)
        dm0 = None  #mf.make_rdm1(mo_coeff0, mo_occ0)
        # this ends the modification

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        if singlet is None:
            # Without specify singlet, used in ground state orbital hessian
            def vind(dm1):
                # The singlet hessian
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    v1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                       rho0, vxc, fxc, max_memory=max_memory)
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if abs(omega) > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                        v1 += vj - .5 * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(mol, dm1, hermi=hermi)
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1

        elif singlet:
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = ni.nr_rks_fxc_st(mol, mf.grids, mf.xc, dm0, dm1, 0, True,
                                          rho0, vxc, fxc, max_memory=max_memory)
                    v1 *= .5
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if abs(omega) > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                        v1 += vj - .5 * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(mol, dm1, hermi=hermi)
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1
        else:  # triplet
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = ni.nr_rks_fxc_st(mol, mf.grids, mf.xc, dm0, dm1, 0, False,
                                          rho0, vxc, fxc, max_memory=max_memory)
                    v1 *= .5
                if hybrid:
                    vk = mf.get_k(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if abs(omega) > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 += -.5 * vk
                return v1

    else:  # HF
        if (singlet is None or singlet) and hermi != 2:
            def vind(dm1):
                vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                return vj - .5 * vk
        else:
            def vind(dm1):
                return -.5 * mf.get_k(mol, dm1, hermi=hermi)

    return vind


if __name__ == '__main__':
    from pyscf import scf, tdscf
    from lumeq.utils.pyscf_helper import *
    from lumeq.utils import print_matrix
    from lumeq.embedding.fragment_entangle import EmbeddingMeanField

    infile = '../../samples/formic_acid_6_h2o.in'
    if len(sys.argv)>1: infile = sys.argv[1]
    parameters = parser(infile)
    results = run_pyscf_final(parameters)
    mol, mf = results['mol'], results['mf']
    print('nelec:', mol.nelectron//2, 'nbasis:', mf.mo_coeff.shape[0])
    print_matrix('mo_energy:', mf.mo_energy)

    nstates = 3
    td = tdscf.TDA(mf)
    td.kernel(nstates=nstates)
    print('td converged:', td.converged)
    print_matrix('td energy:', td.e)
    #td.analyze()


    frgm_idx = get_frgm_idx(parameters)
    print('frgm_idx:', frgm_idx)

    extra_orb = 0
    embed = EmbeddingMeanField(mf, frgm_idx, extra_orb=extra_orb)
    embed.emb_basis_dmet(mf, 0)
    eomf = embed.get_eomf(mf)


    td_emb = tdscf.TDA(eomf)
    scf.hf.RHF.gen_response = _gen_rhf_response # use embedding response
    td_emb.kernel(nstates=nstates)
    print('td_emb converged:', td_emb.converged)
    print_matrix('td_emb energy:', td_emb.e)
    #td_emb.analyze()

    from pydmet.embedding import Embedding
    from pydmet.dmet_tda import solve_tda, full_mol
    embed = Embedding(mf, frgm_idx, 1e-6)
    eomf = embed.get_eomf(mf)
    ene_eo_tda, amp_eo_tda = solve_tda(eomf, nstates, term='eo', verbose=5)
