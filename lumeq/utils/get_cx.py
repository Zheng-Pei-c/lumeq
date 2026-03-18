from lumeq import os, sys, np
from lumeq.utils.pyscf_helper import *
from lumeq.utils.ortho_basis import get_ortho_basis

if __name__ == '__main__':
    method = 'lowdin'
    infile = '../samples/h2o.in'
    parameters = parser(infile)
    results = run_pyscf_final(parameters)
    mol, mf = results['mol'], results['mf']

    mo = mf.mo_coeff
    ovlp = mf.get_ovlp()

    method = 'cholesky'
    L, Z, Sinv = get_ortho_basis(ovlp, method=method)
    print_matrix('L:', L)
    print_matrix('Z:', Z)

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


    Z = np.einsum('rs,sp->rp', ovlp, mo)
    dS = -np.einsum('rp,nrs,sq->npq', mo, S_d1, mo)

    I = np.eye(nbas)
    A = np.kron(I, Z) + np.kron(Z.conj().T, I)

    d1 = []
    for i in range(dS.shape[0]):
        x = np.linalg.solve(A, dS[i].flatten(order='F'))
        d1.append(x.reshape((nbas, nbas), order='F'))
    d1 = np.array(d1)

    Cx = d1
    #Cx = -np.einsum('nrs,sp->nrp', d1, mo)
    print_matrix('Cx', Cx)

    Sinv = np.einsum('rp,sp->rs', mo, mo)
    Cx = -.5* np.einsum('rs,nsp,pq->nrq', Sinv, S_d1, mo)
    print_matrix('Cx_2', Cx)
