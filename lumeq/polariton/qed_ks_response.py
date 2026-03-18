from lumeq import sys, np
from lumeq.utils import convert_units, print_matrix
from lumeq.utils.pyscf_helper import build_molecule
from lumeq.polariton import polariton_cs
from lumeq.polariton.qed_ks import print_qed_dse_energy

from scipy.optimize import newton_krylov

from pyscf import lib, scf, tdscf
from pyscf.lib import logger

# Solve the frequency-dependent CPHF problem
# [A-wI, B   ] [X] + [h1] = [0]
# [B   , A+wI] [Y]   [h1]   [0]
def cphf_with_freq(mf, h1, mo_energy=None, mo_coeff=None, mo_occ=None,
                   freq=0, level_shift=.1,
                   max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN):
    log = logger.new_logger(verbose=verbose)
    time0 = (logger.process_clock(), logger.perf_counter())

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ

    occidx = np.where(mo_occ==2)[0]
    viridx = np.where(mo_occ==0)[0]
    orbo = mo_coeff[:,occidx]
    orbv = mo_coeff[:,viridx]

    nocc, nvir = len(occidx), len(viridx)
    nao = nocc + nvir
    h1 = h1.reshape(-1, nao, nao)

    e_ia = mo_energy[viridx] - mo_energy[occidx,None]
    # e_ia - freq may produce very small elements which can cause numerical
    # issue in krylov solver
    level_shift = 0.1
    diag = (e_ia - freq,
            e_ia + freq)
    diag[0][diag[0] < level_shift] += level_shift
    diag[1][diag[1] < level_shift] += level_shift

    h1ov = -np.einsum('xpq,po,qv->xov', h1, orbo.conj(), orbv)
    h1vo = -np.einsum('xpq,qo,pv->xov', h1, orbo, orbv.conj())

    rhs = np.stack((h1ov, h1vo), axis=1)
    rhs = rhs.reshape(-1, nocc*nvir*2)
    guess = np.stack((h1ov/diag[0], h1vo/diag[1]), axis=1)
    guess = guess.reshape(-1, nocc*nvir*2)

    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=0)

    def vind(xys):
        xys = np.reshape(xys, (-1,2,nocc,nvir))
        xs, ys = xys.transpose(1,0,2,3)
        # dms = AX + BY
        # *2 for double occupancy
        dms  = lib.einsum('xov,qv,po->xpq', xs*2, orbv.conj(), orbo)
        dms += lib.einsum('xov,pv,qo->xpq', ys*2, orbv, orbo.conj())

        v1ao = vresp(dms)
        v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        v1vo = lib.einsum('xpq,qo,pv->xov', v1ao, orbo, orbv.conj())

        v1ov += np.einsum('xov,ov->xov', xs, diag[0])
        v1vo += np.einsum('xov,ov->xov', ys, diag[1])

        v = np.stack((v1vo, v1ov), axis=1)
        return v.reshape(-1, nocc*nvir*2) - rhs

    mo1 = newton_krylov(vind, guess, f_tol=tol)
    log.timer('krylov solver in CPHF', *time0)

    mo1 = np.reshape(mo1, (-1,2,nocc,nvir))
    mo1 = mo1.transpose(1,0,2,3)

    xs, ys = mo1
    dms  = lib.einsum('xov,qv,po->xpq', xs*2, orbv.conj(), orbo)
    dms += lib.einsum('xov,pv,qo->xpq', ys*2, orbv, orbo.conj())

    mo_e1 = lib.einsum('xpq,pi,qj->xij', vresp(dms), orbo, orbo)

    return mo1, mo_e1, dms


def get_polarizability(mol, mf, h1=None, freq=0., scale=False, method=1):
    # polarizability atomic unit is bohr**3
    print('freq:'+str(freq), end=' ')
    if method == 0:
        print('sum-of-state', end=' ')
    elif method == 1:
        print('linear-response', end=' ')
    elif method == 2:
        print('1e approximation', end=' ')

    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ

    if method == 0:
        td = tdscf.TDDFT(mf)
        td.kernel(nstates=1000)
        if np.any(td.converged is False):
            print('td converged:', td.converged)
        omega = td.e
        h1 = td.transition_dipole()

    else:
        if h1 is None:
            h1 = mol.intor('int1e_r', comp=3, hermi=0)

        if method == 2:
            occidx = np.where(mo_occ==2)[0]
            viridx = np.where(mo_occ==0)[0]

            e_ia = mo_energy[viridx] - mo_energy[occidx,None]
            omega = e_ia.ravel() #ov

            orbo = mo_coeff[:,occidx]
            orbv = mo_coeff[:,viridx]
            h1 = np.einsum('xpq,po,qv->xov', h1, orbo.conj(), orbv) #ov
            h1 = h1.reshape(-1, len(omega)).T


    if method == 1:
        dms = cphf_with_freq(mf, h1, mo_energy, mo_coeff, mo_occ, freq=freq)[2]
        alpha = -np.einsum('xpq,ypq->xy', h1, dms)

    else:
        alpha_res = np.einsum('nx,n,ny->xy', h1, 1./(omega-freq), h1)
        alpha_nonres = np.einsum('nx,n,ny->xy', h1, 1./(omega+freq), h1)
        alpha = alpha_res + alpha_nonres
        print_matrix('alpha_tot:', alpha)
        #print_matrix('alpha_res:', alpha_res)
        #print_matrix('alpha_nonres:', alpha_nonres)
        alpha = alpha_nonres


    if scale:
        alpha *= freq

    print_matrix('alpha (bohr**3):', alpha)
    return alpha



if __name__ == '__main__':
    atom = sys.argv[1]
    h2 = """
            H    0. 0. -0.373
            H    0. 0.  0.373
    """
    hf = """
            H    0. 0. -0.459
            F    0. 0.  0.459
    """
    lif = """
           Li    0. 0. -0.791
            F    0. 0.  0.791
    """
    h2o = """
    O          0.00000        0.00000        0.00000
    H          0.00000        0.75695        0.58588
    H          0.00000       -0.75695        0.58588
    """
    functional = 'hf'
    mol = build_molecule(locals()[atom], '6-31g*')
    mf = scf.RKS(mol)

    mf.xc = functional
    mf.grids.prune = True
    e_tot0 = mf.kernel()

    dm = mf.make_rdm1()

    frequency = 0.42978 # gs doesn't depend on frequency

    for freq in [0, 0.4]:
        alpha = get_polarizability(mol, mf, freq=freq, method=0)
        alpha = get_polarizability(mol, mf, freq=freq, method=1)
        alpha = get_polarizability(mol, mf, freq=freq, method=2)


#    scf_method = polariton_cs
#
#    x, c = 0, 0.01
#    coupling = np.zeros(3)
#    coupling[x] = c
#    mf1 = scf_method(mol) # in number (Fock) state
#
#    #mf1.verbose = 10
#    mf1.xc = functional
#    mf1.grids.prune = True
#    mf1.get_multipole_matrix(coupling)
#
#    e0 = mf1.get_coupling_energy(dm=dm)
#    e_tot = mf1.kernel()#(dm0=dm)
#    e1 = mf1.get_coupling_energy()
#
#    e_tot = np.array([e_tot0, e_tot])
#    e_tot = convert_units(e_tot, 'hartree', 'ev')
#    print_qed_dse_energy(coupling[x], e0, e1, e_tot)
#
#    for freq in [0, 0.4]:
#        alpha = get_polarizability(mol, mf1, freq=freq, method=0)
#        alpha = get_polarizability(mol, mf1, freq=freq, method=1)
