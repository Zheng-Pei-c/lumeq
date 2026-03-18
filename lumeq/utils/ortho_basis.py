from lumeq import sys, np
from lumeq.utils import print_matrix
from lumeq.utils import fdiff
from lumeq.utils.unit_conversion import BOHR

from pyscf import gto

def get_ortho_basis(S, method='lowdin', eigen=False):
    r"""
    S is symmetric overlap matrix of atomic basis
    L = S^(1/2)
    Z = S^(-1/2)
    S = L * L^T
    S^-1 = Z^T * Z
    inv = S^(-1)
    """

    if method == 'lowdin':
        s, V = np.linalg.eigh(S)
        idx = s > 1e-15
        s, V = s[idx], V[:, idx]

        L = np.dot(V*np.sqrt(s), V.conj().T)
        Z = np.dot(V/np.sqrt(s), V.conj().T)
        inv = np.dot(V/s, V.conj().T)

        if eigen:
            return L, Z, inv, s, V
        else:
            return L, Z, inv

    elif method == 'cholesky':
        import scipy
        L = scipy.linalg.cholesky(S)
        Z = scipy.linalg.lapack.dtrtri(L, lower=True)[0]
        inv = scipy.linalg.inv(S)

        return L, Z, inv


def get_ortho_basis_deriv(S, dS, method='lowdin', l_or_z='L', solver='iter'):
    if method == 'cholesky':
        raise ValueError('no direct inverse for cholesky yet and the iter is not tested')

    L, Z, Sinv, s, V = get_ortho_basis(S, method=method, eigen=True)

    if l_or_z == 'Z':
        dS = -np.einsum('pq,xqr,rs->xps', Sinv, dS, Sinv)

    nbas = S.shape[0]

    if solver == 'inv':
        s = np.sqrt(s)
        if l_or_z == 'Z': s = 1./s
        sinv = np.repeat(s, nbas)
        sinv += sinv.reshape(nbas, -1).flatten(order='F')
        sinv = np.diag(1./sinv)
        # s is diagonal so that we can get kronecker product easily
        #s = np.diag(s)
        #sinv = np.kron(I, s) + np.kron(s, I)
        #sinv = np.diag(1./np.diag(sinv))

        Vinv = np.kron(V.conj().T, V.conj().T)
        V = np.kron(V, V)

        B = np.einsum('pq,qr,rs->ps', V, sinv, Vinv)
        d1 = np.einsum('pq,xq->xp', B, dS.reshape((dS.shape[0], -1), order='F'))
        d1 = np.reshape(d1, (-1, nbas, nbas), order='F')

    elif solver == 'iter':
        I = np.eye(nbas)
        if l_or_z == 'Z':
            A = np.kron(I, Z) + np.kron(Z.conj().T, I)
        else: # 'L'
            A = np.kron(I, L) + np.kron(L.conj().T, I)

        d1 = []
        for i in range(dS.shape[0]):
            x = np.linalg.solve(A, dS[i].flatten(order='F'))
            d1.append(x.reshape((nbas, nbas), order='F'))

    return np.array(d1)



if __name__ == '__main__':
    from lumeq.utils import parser
    from lumeq.utils.pyscf_helper import run_pyscf_final

    method = 'lowdin'
    infile = '../samples/h2o.in'
    parameters = parser(infile)
    results = run_pyscf_final(parameters)
    mol, mf = results['mol'], results['mf']

    ovlp = mf.get_ovlp()
    L, Z, Sinv = get_ortho_basis(ovlp, method=method)
    #print_matrix('ovlp:', ovlp, 7, 1)
    #print_matrix('S^-1/2:', Z, 7, 1)

    mf_grad = mf.nuc_grad_method()
    ovlp1 = mf_grad.get_ovlp(mol) # overlap derivative (xpq)

    # distribute derivatives onto atoms
    natm = mol.natm
    atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    nbas = ovlp.shape[0]

    S_d1 = np.zeros((natm, 3, nbas, nbas))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        S_d1[k,:,p0:p1] = ovlp1[:,p0:p1]
        S_d1[k,:,:,p0:p1] += ovlp1[:,p0:p1].transpose(0,2,1)
    S_d1 = np.reshape(S_d1, (-1, nbas, nbas))
    Sinv_d1 = -np.einsum('pq,xqr,rs->xps', Sinv, S_d1, Sinv)

    dL = get_ortho_basis_deriv(ovlp, S_d1, method=method, l_or_z='L', solver='iter')
    if method == 'lowdin':
        dL2  = get_ortho_basis_deriv(ovlp, S_d1, method=method, l_or_z='L', solver='inv')
        print('dL-dL2 diff:', np.linalg.norm(dL-dL2))
    dS2  = np.einsum('xpq,qr->xpr', dL, L)
    dS2 += np.einsum('pq,xqr->xpr', L, dL)
    print('dS-(dL*L+L*dL) diff:', np.linalg.norm(S_d1-dS2))

    dZ = get_ortho_basis_deriv(ovlp, S_d1, method=method, l_or_z='Z', solver='iter')
    if method == 'lowdin':
        dZ2  = get_ortho_basis_deriv(ovlp, S_d1, method=method, l_or_z='Z', solver='inv')
        print('dZ-dZ2 diff:', np.linalg.norm(dZ-dZ2))
    dS2  = np.einsum('xpq,qr->xpr', dZ, Z)
    dS2 += np.einsum('pq,xqr->xpr', Z, dZ)
    print('dSinv-(dZ*Z+Z*dZ) diff:', np.linalg.norm(Sinv_d1-dS2))

    L_d1, Z_d1 = dL, dZ


    # doing finite difference
    norder, step_size = 2, 1e-3
    coords = mol.atom_coords()*BOHR

    S_fd = np.zeros((natm, 3, nbas, nbas))
    L_fd = np.zeros((natm, 3, nbas, nbas))
    Z_fd = np.zeros((natm, 3, nbas, nbas))
    Sinv_fd = np.zeros((natm, 3, nbas, nbas))
    for n in range(natm):
        for x in range(3):
            fd = fdiff(norder, step_size)
            coords_new = fd.get_x(coords, [n, x])

            S1, Z1, L1, Sinv1 = [], [], [], []
            for d in range(len(coords_new)):
                pmol = mol.copy()
                pmol.set_geom_(coords_new[d])
                #s12 = gto.intor_cross('int1e_ovlp', pmol, mol)
                ovlp1 = mf.get_ovlp(pmol)
                #print_matrix('pmf:', s12, 7, 1)

                Lx, Zx, Sinvx = get_ortho_basis(ovlp1, method=method)

                S1.append(ovlp1)
                L1.append(Lx)
                Z1.append(Zx)
                Sinv1.append(Sinvx)

            S_fd[n, x] = fd.compute(np.array(S1), 1./BOHR)
            L_fd[n, x] = fd.compute(np.array(L1), 1./BOHR)
            Z_fd[n, x] = fd.compute(np.array(Z1), 1./BOHR)
            Sinv_fd[n, x] = fd.compute(np.array(Sinv1), 1./BOHR)

    S_fd = np.reshape(S_fd, (-1, nbas, nbas))
    L_fd = np.reshape(L_fd, (-1, nbas, nbas))
    Z_fd = np.reshape(Z_fd, (-1, nbas, nbas))
    Sinv_fd = np.reshape(Sinv_fd, (-1, nbas, nbas))

    diff = S_d1 - S_fd
    print('S_d1-S_fd diff:', np.linalg.norm(diff))
    #print_matrix('S_d1:', S_d1[0], 7, 1)
    #print_matrix('S_fd:', S_fd[0], 7, 1)

    diff = Sinv_d1 - Sinv_fd
    print('Sinv_d1-Sinv_fd diff:', np.linalg.norm(diff))
    #print_matrix('Sinv_d1:', Sinv_d1[0], 7, 1)
    #print_matrix('Sinv_fd:', Sinv_fd[0], 7, 1)

    diff = L_d1 - L_fd
    print('L_d1-L_fd diff:', np.linalg.norm(diff))
    #print_matrix('L_d1:', L_d1[0], 7, 1)
    #print_matrix('L_fd:', L_fd[0], 7, 1)

    diff = Z_d1 - Z_fd
    print('Z_d1-Z_fd diff:', np.linalg.norm(diff))
    #print_matrix('Z_d1:', Z_d1[0], 7, 1)
    #print_matrix('Z_fd:', Z_fd[0], 7, 1)
