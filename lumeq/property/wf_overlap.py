from lumeq import sys, np, itertools
from lumeq.property.rdm_analysis import assemble_amplitudes
from lumeq.utils import print_matrix
from lumeq.utils import monitor_performance

from pyscf import scf, tdscf, gto

@monitor_performance
def cal_wf_overlap_r(Xm, Ym, Xn, Yn, Cm, Cn, S):
    r"""
    Calculate wavefunction overlap between excited states at different nuclear configurations
    for restricted case
    Ovlp = < Psi_m | Psi_n >

    Matrix determinant lemma is used.
    vec1: `S_vo S_oo^-1` in ov (transpose is used)
    vec2: `S_oo^-1 S_ov` in ov
    vec3: `(S_vv - S_vo S_oo^-1 S_ov)` in vv
    vec4: `S_oo^-1` in oo

    Args:
        Xm (numpy.ndarray): Excitation amplitudes at geometry ``m`` with shape
            ``(nroots, nocc, nvir)``.
        Ym (numpy.ndarray): De-excitation amplitudes with shape
            ``(nroots, nocc, nvir)``.
        Xn (numpy.ndarray): Excitation amplitudes at geometry ``n`` with shape
            ``(nroots, nocc, nvir)``.
        Yn (numpy.ndarray): De-excitation amplitudes with shape
            ``(nroots, nocc, nvir)``.
        Cm (numpy.ndarray): Molecular-orbital coefficients at geometry ``m``
            with shape ``(nbasis, nbasis)``.
        Cn (numpy.ndarray): Molecular-orbital coefficients at geometry ``n``
            with shape ``(nbasis, nbasis)``.
        S (numpy.ndarray): Overlap matrix between geometries ``m`` and ``n``
            with shape ``(nbasis, nbasis)``.

    Returns:
        numpy.ndarray: Wavefunction overlap matrix.
    """
    # restricted case has same orbitals for alpha and beta electrons
    has_m = True if isinstance(Xm, np.ndarray) else False
    has_n = True if isinstance(Xn, np.ndarray) else False
    has_y = True if (isinstance(Ym, np.ndarray) and isinstance(Yn, np.ndarray)) else False

    nroots, no, nv = Xm.shape

    smo = np.einsum('mp,mn,nq->pq', Cm, S, Cn)
    #print_matrix('smo:', smo, 5, 1)
    smo_oo = np.copy(smo[:no,:no])
    dot_0 = np.linalg.det(smo_oo)
    ovlp_00 = dot_0**2

    if not (has_m or has_n):
        return ovlp_00

    # Cramer's rule
    # Ax_j = b_j <=> x_j = det(A_j(b_j)) / det(A) where A_j(b_j) is replacing A's j-column with b_j
    vec0 = np.linalg.inv(smo_oo) # S_oo^-1
    vec1 = np.einsum('aj,ji->ia', smo[no:,:no], vec0) # replace rows
    vec2 = np.einsum('ij,jb->ib', vec0, smo[:no,no:]) # replace columns
    # rank-2 perturbation from matrix determinant lemma
    vec3 = smo[no:,no:] - np.einsum('aj,jb->ab', smo[no:,:no], vec2)

    # excited-ground
    if has_m:
        Xm = Xm.conj()
        ovlp1 = np.einsum('kia,ia->k', Xm, vec1)
        ovlp_m0 = np.copy(ovlp1)

        if has_y:
            Ym = Ym.conj()
            ovlp2 = np.einsum('kia,ia->k', Ym, vec2)
            ovlp_m0 += ovlp2

        ovlp_m0 *= 2.*ovlp_00
        if not has_n:
            return np.array([ovlp_00, ovlp_m0])

    # ground-excited
    if has_n:
        ovlp3 = np.einsum('kia,ia->k', Xn, vec2)
        ovlp_0n = np.copy(ovlp3)

        if has_y:
            ovlp4 = np.einsum('kia,ia->k', Yn, vec1)
            ovlp_0n += ovlp4

        ovlp_0n *= 2.*ovlp_00
        if not has_m:
            return np.array([ovlp_00, ovlp_0n])

    # excited-excited
    if has_m and has_n:
        # e-g * g-e
        ovlp_mn = np.einsum('m,n->mn', ovlp1, ovlp3)
        if has_y:
            ovlp_mn -= np.einsum('m,n->mn', ovlp2, ovlp4)

        # e-e * g-g
        ovlp_mn *= 2. # first contribution is same as e-g * g-e
        ovlp_mn += np.einsum('mia,njb,ji,ab->mn', Xm, Xn, vec0, vec3)
        if has_y:
            ovlp_mn -= np.einsum('mia,njb,ij,ba->mn', Ym, Yn, vec0, vec3)

        ovlp_mn *= 2.*ovlp_00
        return np.block([[ovlp_00, ovlp_0n.reshape(1,-1)], [ovlp_m0.reshape(-1,1), ovlp_mn]])


@monitor_performance
def cal_wf_overlap_u(Xm, Ym, Xn, Yn, Cm, Cn, S):
    r"""
    calculate wavefunction overlap between excited states at different nuclear configurations
    for unrestricted case
    """
    # unrestricted case, including alpha and beta spin orbitals
    has_m = (Xm is not None)
    has_n = (Xn is not None)
    has_y = ((Ym[0] is not None) and (Yn[0] is not None))

    nroots, no_a, nv_a = Xm[0].shape # alpha
    _,      no_b, nv_b = Xm[1].shape # beta
    nocc, nvir = [no_a, no_b], [nv_a, nv_b]

    smo = np.einsum('...mp,mn,...nq->...pq', Cm, S, Cn)
    assert smo.ndim == 3

    smo_oo = [None]*2
    for s in range(2): # loop over spins
        smo_oo[s] = np.copy(smo[s,:nocc[s],:nocc[s]])
    dot_0 = [np.linalg.det(x) for x in smo_oo] # loop over the first-index
    ovlp_00 = dot_0[0]*dot_0[1]

    if not (has_m or has_n):
        return ovlp_00

    # Cramer's rule
    # Ax_j = b_j <=> x_j = det(A_j(b_j)) / det(A) where A_j(b_j) is replacing A's j-column with b_j
    vec0, vec1, vec2, vec3 = [None]*2, [None]*2, [None]*2, [None]*2
    for s in range(2):
        no = nocc[s]
        vec0[s] = np.linalg.inv(smo_oo[s]) # S_oo^-1
        vec1[s] = np.einsum('aj,ji->ia', smo[s,no:, :no], vec0[s]) # replace rows
        vec2[s] = np.einsum('ij,jb->ib', vec0[s], smo[s,:no,no:]) # replace columns
        # rank-2 perturbation from matrix determinant lemma
        vec3[s] = smo[s,no:,no:] - np.einsum('aj,jb->ab', smo[s,no:,:no], vec2[s])


    # excited-ground
    if has_m:
        ovlp1 = np.empty((2, nroots))
        for s in range(2):
            Xm[s] = Xm[s].conj()
            ovlp1[s] = np.einsum('kia,ia->k', Xm[s], vec1[s])
        ovlp_m0 = np.copy(ovlp1)

        if has_y:
            ovlp2 = np.empty((2, nroots))
            for s in range(2):
                Ym[s] = Ym[s].conj()
                ovlp2[s] = np.einsum('kia,ia->k', Ym[s], vec2[s])
            ovlp_m0 += ovlp2

        ovlp_m0 = np.sum(ovlp_m0, axis=0)*ovlp_00
        if not has_n:
            return np.array([ovlp_00, ovlp_m0])

    # ground-excited
    if has_n:
        ovlp3 = np.empty((2, nroots))
        for s in range(2):
            ovlp3[s] = np.einsum('kia,ia->k', Xn[s], vec2[s])
        ovlp_0n = np.copy(ovlp3)

        if has_y:
            ovlp4 = np.empty((2, nroots))
            for s in range(2):
                ovlp4[s] = np.einsum('kia,ia->k', Yn[s], vec1[s])
            ovlp_0n += ovlp4

        ovlp_0n = np.sum(ovlp_0n, axis=0)*ovlp_00
        if not has_m:
            return np.array([ovlp_00, ovlp_0n])

    # excited-excited
    if has_m and has_n:
        ovlp_mn = np.zeros((nroots, nroots))
        for s in range(2):
            # e-g * g-e and first contribution of e-e * g-g
            ovlp_mn += 2.*np.einsum('m,n->mn', ovlp1[s], ovlp3[s])
            if has_y:
                ovlp_mn -= 2.*np.einsum('m,n->mn', ovlp2[s], ovlp4[s])

            # e-e * g-g
            ovlp_mn += np.einsum('mia,njb,ji,ab->mn', Xm[s], Xn[s], vec0[s], vec3[s])
            if has_y:
                ovlp_mn -= np.einsum('mia,njb,ij,ba->mn', Ym[s], Yn[s], vec0[s], vec3[s])

        ovlp_mn *= ovlp_00
        return np.block([[ovlp_00, ovlp_0n.reshape(1,-1)], [ovlp_m0.reshape(-1,1), ovlp_mn]])


def cal_wf_overlap_sf(Xm, Ym, Xn, Yn, Cm, Cn, S, extype=0):
    r"""
    calculate wavefunction overlap between excited states at different nuclear configurations
    for ro spin-flip case
    """
    # follow the spin-flip dft implemented at https://github.com/Haskiy/pyscf
    # extype=0: spin flip up, based on ms=-1 triplet ground-state
    # extype=1: spin flip down, based on ms=+1 triplet ground-state
    has_y = ((Ym is not None) and (Yn is not None))

    #if extype == 0:
    #    nroots, no_b, nv_a = Xm.shape
    #    if has_y:
    #        _,  no_a, nv_b = Ym.shape
    #elif extype == 1:
    #    nroots, no_a, nv_b = Xm.shape
    #    no_b, nv_a = no_a-2, nv_b-2
    #    if has_y:
    #        _,  no_b, nv_a = Ym.shape
    #nocc, nvir = [no_a, no_b], [nv_a, nv_b]
    # use extype==1 case here, and swap orbitals for extype==0 later
    nroots, no_a, nv_b = Xm.shape
    no_b, nv_a = no_a-2, nv_b-2
    nocc, nvir = [no_a, no_b], [nv_a, nv_b]

    smo = np.einsum('...mp,mn,...nq->...pq', Cm, S, Cn)
    assert smo.ndim == 3
    # swap alpha and beta orbital overlaps
    if extype == 0: smo = smo[::-1]

    smo_oo = [None]*2
    for s in range(2): # loop over spins
        smo_oo[s] = np.copy(smo[s,:nocc[s],:nocc[s]])
    dot_0 = [np.linalg.det(x) for x in smo_oo] # loop over the first-index
    ovlp_00 = dot_0[0]*dot_0[1]


    def kernel(no_a, no_b, nv_b, soo_a, soo_b, smo_b):
        ovlp1 = np.zeros((no_a, no_a))
        for i in range(no_a):
            for j in range(no_a):
                ts = np.delete(soo_a, i, axis=0) # i-row
                ts = np.delete(ts, j, axis=1) # j-column
                ovlp1[i,j] = np.linalg.det(ts)

        ovlp2 = np.zeros((nv_b, nv_b))
        for a, b in itertools.product(range(nv_b), range(nv_b)):
            ts = np.vstack((soo_b, smo_b[a+no_b,:no_b])) # add a-row
            # hstack is samilar to column_stack but needs newaxis for 1d
            ts = np.column_stack((ts, smo_b[:no_b+1,b+no_b])) # add b-column
            ts[no_b,no_b] = smo_b[a+no_b,b+no_b]
            ovlp2[a,b] = np.linalg.det(ts)

        return ovlp1, ovlp2

    # excited-excited
    # e-g * g-e
    ovlp1, ovlp2 = kernel(no_a, no_b, nv_b, smo_oo[0], smo_oo[1], smo[1])
    #ovlp_mn = np.einsum('mia,njb,ij,ab->mn', Xm, Xn, ovlp1, ovlp2)
    ovlp_mn = np.einsum('njb,ij->nib', Xn, ovlp1)
    ovlp_mn = np.einsum('nib,ab->nia', ovlp_mn, ovlp2)
    ovlp_mn = np.einsum('mia,nia->mn', Xm, ovlp_mn)

    if has_y: # swap alpha and beta
        ovlp3, ovlp4 = kernel(no_b, no_a, nv_a, smo_oo[1], smo_oo[0], smo[0])
        # use transport
        #ovlp_mn -= np.einsum('mia,njb,ji,ba->mn', Ym, Yn, ovlp3, ovlp4)
        tmp = np.einsum('njb,ji->nib', Yn, ovlp3)
        tmp = np.einsum('nib,ba->nia', tmp, ovlp4)
        ovlp_mn -= np.einsum('mia,nia->mn', Ym, tmp)

    #return ovlp_00, ovlp_mn
    return ovlp_mn


def cal_wf_overlap(Xm, Ym, Xn, Yn, Cm, Cn, S, itype='r'):
    r"""
    calculate wavefunction overlap between excited states at different nuclear configurations
    """
    if itype == 'r':
        return cal_wf_overlap_r(Xm, Ym, Xn, Yn, Cm, Cn, S)
    elif itype == 'u':
        return cal_wf_overlap_u(Xm, Ym, Xn, Yn, Cm, Cn, S)
    elif 'sf' in itype:
        return cal_wf_overlap_sf(Xm, Ym, Xn, Yn, Cm, Cn, S, int(itype[-1]))


def _overlap_gg(Cm, Cn, S, nocc):
    # determinant overlap between ground states
    # Cm and Cn are the mo_coeffs at different nuclear configurations
    # S is the off-diagonal overlap between them
    smo = np.einsum('...mp,mn,...nq->...pq')

    if smo.ndim == 2: # single-spin orbitals
        return np.linalg.det(smo[:nocc,:nocc]), smo
    else:
        dot = []
        for i, no in enumerate(nocc):
            dot.append(np.linalg.det(smo[i,:no,:no]))
        return dot, smo


def _overlap_eg(Xm, Yn, Cm=None, Cn=None, S=None, smo=None):
    # determinant overlap between excited and ground states
    has_y = True if isinstance(Yn, np.ndarray) else False
    _, no, nv = Xm.shape

    Xm = Xm.conj()
    if has_y:
        Yn = Yn.conj()

    if not isinstance(smo, np.ndarray):
        _, smo = _overlap_gg(Cm, Cn, S, no)
    smo_oo = np.copy(smo[:no,:no])

    # e-g of Xm, g-e of Yn
    ovlp1, ovlp4 = [], []
    for a, i in itertools.product(range(nv), range(no)):
        ts = np.copy(smo_oo)
        ts[i,:] = smo[no+a,:no]
        dot = np.linalg.det(ts)

        ovlp1.append(Xm[:,i,a] * dot)
        if has_y: ovlp4.append(Yn[:,i,a] * dot)

    return ovlp1, ovlp4


def _overlap_ge(Xn, Ym, Cm=None, Cn=None, S=None, smo=None):
    r"""
    determinant overlap between excited and ground states
    slow version using expicit loops
    """
    # determinant overlap between excited and ground states
    has_y = True if isinstance(Ym, np.ndarray) else False
    _, no, nv = Xn.shape

    if not isinstance(smo, np.ndarray):
        _, smo = _overlap_gg(Cm, Cn, S, no)
    smo_oo = np.copy(smo[:no,:no])

    # g-e of Xm, e-g of Yn
    ovlp2, ovlp3 = [], []
    for a, i in itertools.product(range(nv), range(no)):
        ts = np.copy(smo_oo)
        ts[:,i] = smo[:no,no+a]
        dot = np.linalg.det(ts)

        ovlp3.append(Xn[:,i,a] * dot)
        if has_y: ovlp2.append(Ym[:,i,a] * dot)

    return ovlp2, ovlp3


def _overlap_ee(Xm, Ym, Xn, Yn, Cm=None, Cn=None, S=None, smo=None):
    r"""
    determinant overlap between excited states
    slow version using expicit loops
    """
    # determinant overlap between excited states
    has_y = True if (isinstance(Ym, np.ndarray) and isinstance(Yn, np.ndarray)) else False
    _, no, nv = Xn.shape

    Xm = Xm.conj()
    if has_y:
        Ym = Ym.conj()

    if not isinstance(smo, np.ndarray):
        _, smo = _overlap_gg(Cm, Cn, S, no)
    smo_oo = np.copy(smo[:no,:no])

    # g-e of Xm, e-g of Yn
    ovlp = 0.
    for a, i in itertools.product(range(nv), range(no)):
        for b, j in itertools.product(range(nv), range(no)):
            ts0 = np.copy(smo_oo)
            ts0[i,:] = smo[no+a,:no]
            ts0[:,j] = smo[:no,no+b]
            ts0[i,j] = smo[no+a,no+b]
            dot = np.linalg.det(ts0)

            ovlp += np.einsum('m,n->mn', Xm[:,i,a], Xn[:,j,b]) * dot
            if has_y: ovlp -= np.einsum('m,n->mn', Ym[:,j,b], Yn[:,i,a]) * dot

    return ovlp


def cal_wf_overlap_r0(Xm, Ym, Xn, Yn, Cm, Cn, S):
    r"""
    slow version of wavefunction overlap of restricted case for checking
    """
    has_m = True if isinstance(Xm, np.ndarray) else False
    has_n = True if isinstance(Xn, np.ndarray) else False
    has_y = True if (isinstance(Ym, np.ndarray) and isinstance(Yn, np.ndarray)) else False

    _, no, nv = Xm.shape

    smo = np.einsum('mp,mn,nq->pq', Cm, S, Cn)
    smo_oo = np.copy(smo[:no,:no])
    dot_0 = np.linalg.det(smo_oo)

    if not (has_m or has_n):
        return dot_0**2

    if has_m:
        Xm = Xm.conj()
        if has_y:
            Ym = Ym.conj()

    if has_m or has_y:
        ovlp1, ovlp4 = _overlap_eg(Xm, Yn, smo=smo)

    if has_n or has_y:
        ovlp2, ovlp3 = _overlap_ge(Xn, Ym, smo=smo)

    # excited-ground
    if has_m:
        ovlp_m0 = np.sum(ovlp1, axis=0)*dot_0 # e-g
        if has_y:
            ovlp_m0 += np.sum(ovlp2, axis=0)*dot_0 # e-g from Y

        if not has_n:
            return np.array([dot_0**2, 2.*ovlp_m0])

    # ground-excited
    if has_n:
        ovlp_0n = np.sum(ovlp3, axis=0)*dot_0 # g-e
        if has_y:
            ovlp_0n += np.sum(ovlp4, axis=0)*dot_0 # g-e from Y

        if not has_m:
            return np.array([dot_0**2, 2.*ovlp_0n])

    # excited-excited
    if has_m and has_n:
        # e-e * g-g
        ovlp_mn = _overlap_ee(Xm, Ym, Xn, Yn, smo=smo)*dot_0

        # e-g * g-e
        ovlp_mn += np.einsum('im,jn->mn', ovlp1, ovlp3)
        if has_y:
            ovlp_mn -= np.einsum('jm,in->mn', ovlp2, ovlp4)

        return 2.*np.block([[dot_0**2/2., ovlp_0n.reshape(1,-1)], [ovlp_m0.reshape(-1,1), ovlp_mn]])


def change_phase(x0, y0, x1, y1, mo0, mo1, ovlp):
    r"""
    change the phase of excited states at geometry 1 to match that at geometry 0
    """
    nroots, no, nv = x1.shape
    ovlp = np.einsum('mp,mn,nq->pq', mo0, ovlp, mo1)
    idx = np.argmax(np.abs(ovlp), axis=0) # large index for each column
    #print('idx:', idx)

    for i, j in enumerate(idx):
        if ovlp[i,j] < 0.:
            #print(i, j)
            mo1[:,j] *= -1
            if j < no:
                x1[:,j,:] *= -1.
            else:
                x1[:,:,j-no] *= -1.
            if isinstance(y1, np.ndarray):
                if j < no:
                    y1[:,j,:] *= -1.
                else:
                    y1[:,:,j-no] *= -1.

    return x1, y1, mo1


def sign_fixing(mat):
    r"""
    fix the sign indeterminacy of wavefunction overlap matrix
    refer Zhou, Subotnik JCTC 2020, 10.1021/acs.jctc.9b00952
    """
    #U, s, Vt = np.linalg.svd(mat)
    #mat = np.einsum('ij,jk->ik', U, Vt)

    if np.linalg.det(mat) < 0.:
        mat[:,0] *= -1.

    nroots = mat.shape[0]

    # Jacobi sweeps
    converged = False
    while not converged:
        converged = True

        for i, j in itertools.product(range(nroots), range(nroots)):
            dot  = 3.* (mat[i,i]**2 + mat[j,j]**2)
            dot += 6.* (mat[i,j] * mat[j,i])
            dot += 8.* (mat[i,i] + mat[j,j])
            dot -= 3.* (np.dot(mat[i,:], mat[:,i]) + np.dot(mat[j,:], mat[:,j]))

            if dot < 0.:
                mat[:,i] *= -1.
                mat[:,j] *= -1.
                converged = False

    return mat



if __name__ == '__main__':
    h2o = """
            H    1.6090032   -0.0510674    0.4424329
            O    0.8596350   -0.0510674   -0.1653507
            H    0.1102668   -0.0510674    0.4424329
    """

    functional = 'hf'
    basis = '6-311+g*'
    spin = 0 # Nalpha - Nbeta
    charge = 0
    verbose = 0

    rpa = 1
    nroots = 5

    itype = 'r' #'r', 'u', 'sf-0', 'sf-1'
    #print('itype:', itype)
    if 'sf' in itype:
        if itype[-1] == '0': spin = -2
        elif itype[-1] == '1': spin = 2


    mol0 = gto.M(
        atom    = h2o,
        basis   = basis,
        spin    = spin,
        charge  = charge,
        verbose = verbose
    )

    coords = mol0.atom_coords() # bohr
    #coords[0, 2] += .1 # bohr
    coords[1, 2] -= .1 # bohr
    mol1 = mol0.set_geom_(coords, inplace=False, unit='bohr')


    def run_td(mol, rpa=True, itype='r'):
        if itype == 'r':
            scf_model = scf.RKS
        if itype == 'u' or 'sf' in itype:
            scf_model = scf.UKS

        if 'sf' in itype:
            td_model = tdscf.TDDFT_SF if rpa else tdscf.TDA_SF
        else:
            td_model = tdscf.TDDFT if rpa else tdscf.TDA


        mf = scf_model(mol)
        mf.xc = functional
        mf.grids.prune = True
        e = mf.kernel()
        #print('ground-state energy:', e)

        td = td_model(mf)
        td.max_cycle = 600
        td.max_space = 200
        td.nroots  = nroots
        td.verbose = 0
        if 'sf' in itype:
            td.extype = int(itype[-1])
            #td.collinear_samples = 200 #200 is default
        td.kernel()
        #print_matrix('excitation energy:', td.e)
        if not td.converged.all():
            print('td is not converged:', td.converged)
            #print_matrix('norm:', np.einsum('mia,nia->mn', xs, xs) - np.einsum('mia,nia->mn', ys, ys))

        if itype == 'r':
            nocc = int((mf.mo_occ>0).sum())
        else:
            nocc = [int((mf.mo_occ[0]>0).sum()), int((mf.mo_occ[1]).sum())]
        mo = mf.mo_coeff
        xs, ys = assemble_amplitudes(td.xy, nroots, rpa, itype)

        return xs, ys, mo

    from lumeq.utils import set_performance_log
    set_performance_log(debug=True)

    ovlp = gto.intor_cross('int1e_ovlp', mol0, mol1)
    x0, y0, mo0 = run_td(mol0, rpa, itype)
    x1, y1, mo1 = run_td(mol1, rpa, itype)
    if itype == 'r':
        x1, y1, mo1 = change_phase(x0, y0, x1, y1, mo0, mo1, ovlp)
    elif itype == 'u':
        for i in range(2):
            x1[i], y1[i], mo1[i] = change_phase(x0[i], y0[i], x1[i], y1[i], mo0[i], mo1[i], ovlp)

    state_ovlp = cal_wf_overlap(x0, y0, x1, y1, mo0, mo1, ovlp, itype)
    state_ovlp = sign_fixing(state_ovlp)
    print_matrix('state_ovlp:\n', state_ovlp)
