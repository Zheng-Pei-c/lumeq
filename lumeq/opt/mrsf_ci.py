from lumeq import np
from lumeq.utils import ishermitian, print_matrix
from pyscf import gto, scf
from pyscf import ao2mo
from scipy.linalg import eigh, eig

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
    #functional = 'bhandhlyp'
    functional = 'hf'
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

    mo_coeff = mf.mo_coeff
    nmo = mo_coeff.shape[1]
    mo_occ = mf.mo_occ
    occidxa = mo_occ > 0
    occidxb = mo_occ == 2
    viridxa = ~occidxa
    viridxb = ~occidxb

    noa, nob = np.sum(occidxa), np.sum(occidxb)
    nva, nvb = np.sum(viridxa), np.sum(viridxb)
    no, nv = noa-1, nva+1
    s, t = nob, nob+1
    vs, vt = 0, 1
    print('noa:', noa, 'nva:', nva, 'nob:', nob, 'nvb:', nvb, 'no:', no, 'nv:', nv, 's:', s, 't:', t)
    nva = (2, nvb)
    nv = (1, nvb)

    fock = mf.get_fock()
    Fa, Fb = fock.focka, fock.fockb
    #print_matrix('Fa:', Fa)
    #print_matrix('Fb:', Fb)
    Fa = mo_coeff.T.conj() @ Fa @ mo_coeff
    Fb = mo_coeff.T.conj() @ Fb @ mo_coeff
    #print_matrix('Faij:', Fa[:noa,:noa])
    #print_matrix('Fbij:', Fa[:nob,:nob])
    #print_matrix('Faab:', Fa[noa:,noa:])
    #print_matrix('Fbab:', Fb[nob:,nob:])

    #orboa = mo_coeff[:,occidxa]
    #orbob = mo_coeff[:,occidxb]
    #orbva = mo_coeff[:,viridxa]
    #orbvb = mo_coeff[:,viridxb]

    eri_mo = ao2mo.kernel(mol, mo_coeff, compact=False)
    eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
    #print("MO ERI shape:", eri_mo.shape)
    #for p in range(nmo):
    #    for q in range(nmo):
    #        for r in range(nmo):
    #            for ell in range(nmo):
    #                print('8-fold symmetry %d %d %d %d:' %(p,q,r,ell))
    #                print(' %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f'
    #                      % (eri_mo[p,q,r,ell],eri_mo[p,q,ell,r],eri_mo[q,p,r,ell],eri_mo[q,p,ell,r],
    #                         eri_mo[r,ell,p,q],eri_mo[r,ell,q,p],eri_mo[ell,r,p,q],eri_mo[ell,r,q,p]))
    #                print(' %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f'
    #                      % (eri_mo[p,ell,r,q],eri_mo[ell,p,r,q],eri_mo[p,ell,q,r],eri_mo[ell,p,q,r],
    #                         eri_mo[r,q,p,ell],eri_mo[q,r,p,ell],eri_mo[r,q,ell,p],eri_mo[q,r,ell,p]))
    #                print(' %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f'
    #                      % (eri_mo[p,r,ell,q],eri_mo[r,p,ell,q],eri_mo[p,r,q,ell],eri_mo[r,p,q,ell],
    #                         eri_mo[ell,q,p,r],eri_mo[q,ell,p,r],eri_mo[ell,q,r,p],eri_mo[q,ell,r,p]))

    S1 = np.zeros((noa, nvb, noa, nvb)) # ia,jb
    S2 = np.zeros((noa, nvb, noa, nvb)) # ia,jb
    A1_1e = np.zeros((noa, nvb, noa, nvb)) # ia,jb
    A1_2e = np.zeros((noa, nvb, noa, nvb)) # ia,jb
    A2_1e = np.zeros((noa, nvb, noa, nvb)) # ia,jb
    A2_2e = np.zeros((noa, nvb, noa, nvb)) # ia,jb

    # diagonal
    for a in range(nvb):
        for i in range(noa):
            S1[i,a,i,a] += 1.

            for j in range(noa):
                A1_1e[i,a,j,a] -= Fa[j,i]
            for b in range(nvb):
                A1_1e[i,a,i,b] += Fb[nob+a,nob+b]

                for j in range(noa):
                    A1_2e[i,a,j,b] -= eri_mo[nob+a,nob+b,j,i]

    # crossing
    S2[s,vt,s,vt] -= 1.
    S2[t,vs,t,vs] -= 1.
    S2[t,vt,s,vs] += 1.
    S2[s,vs,t,vt] += 1.


    # asymmetric approach
    for i in range(noa):
        A2_1e[i,vt,s,vt] += Fa[s,i]
        A2_1e[i,vs,t,vt] -= Fa[s,i]
        A2_1e[i,vs,t,vs] += Fa[t,i]
        A2_1e[i,vt,s,vs] -= Fa[t,i]

        A2_1e[s,vt,i,vt] += Fa[i,s]
        A2_1e[t,vt,i,vs] -= Fa[i,s]
        A2_1e[t,vs,i,vs] += Fa[i,t]
        A2_1e[s,vs,i,vt] -= Fa[i,t]

    for a in range(nvb):
        A2_1e[s,a,t,vt] += Fb[nob+a,s]
        A2_1e[t,a,t,vs] -= Fb[nob+a,s]
        A2_1e[t,a,s,vs] += Fb[nob+a,t]
        A2_1e[s,a,s,vt] -= Fb[nob+a,t]

        A2_1e[t,vt,s,a] += Fb[s,nob+a]
        A2_1e[t,vs,t,a] -= Fb[s,nob+a]
        A2_1e[s,vs,t,a] += Fb[t,nob+a]
        A2_1e[s,vt,s,a] -= Fb[t,nob+a]

    for i in range(nob):
        A2_1e[i,vt,s,vt] += Fb[s,i]
        A2_1e[i,vs,t,vt] -= Fb[s,i]
        A2_1e[i,vs,t,vs] += Fb[t,i]
        A2_1e[i,vt,s,vs] -= Fb[t,i]

        A2_1e[s,vt,i,vt] += Fb[i,s]
        A2_1e[t,vt,i,vs] -= Fb[i,s]
        A2_1e[t,vs,i,vs] += Fb[i,t]
        A2_1e[s,vs,i,vt] -= Fb[i,t]

    for a in range(*nva):
        A2_1e[s,a,t,vt] += Fa[nob+a,s]
        A2_1e[t,a,t,vs] -= Fa[nob+a,s]
        A2_1e[t,a,s,vs] += Fa[nob+a,t]
        A2_1e[s,a,s,vt] -= Fa[nob+a,t]

        A2_1e[t,vt,s,a] += Fa[s,nob+a]
        A2_1e[t,vs,t,a] -= Fa[s,nob+a]
        A2_1e[s,vs,t,a] += Fa[t,nob+a]
        A2_1e[s,vt,s,a] -= Fa[t,nob+a]


    for i in range(noa):
        for j in range(nob):
            A2_2e[i,vs,j,vs] -= eri_mo[j,t,t,i]
            A2_2e[i,vs,j,vt] += eri_mo[j,t,s,i]
            A2_2e[i,vt,j,vs] += eri_mo[j,s,t,i]
            A2_2e[i,vt,j,vt] -= eri_mo[j,s,s,i]
    for i in range(nob):
        for j in range(noa):
            A2_2e[i,vs,j,vs] -= eri_mo[j,t,t,i]
            A2_2e[i,vs,j,vt] += eri_mo[j,t,s,i]
            A2_2e[i,vt,j,vs] += eri_mo[j,s,t,i]
            A2_2e[i,vt,j,vt] -= eri_mo[j,s,s,i]

    for a in range(nvb):
        for b in range(*nva):
            A2_2e[s,a,s,b] -= eri_mo[nob+a,t,t,nob+b]
            A2_2e[s,a,t,b] += eri_mo[nob+a,s,t,nob+b]
            A2_2e[t,a,s,b] += eri_mo[nob+a,t,s,nob+b]
            A2_2e[t,a,t,b] -= eri_mo[nob+a,s,s,nob+b]
    for a in range(*nva):
        for b in range(nvb):
            A2_2e[s,a,s,b] -= eri_mo[nob+a,t,t,nob+b]
            A2_2e[s,a,t,b] += eri_mo[nob+a,s,t,nob+b]
            A2_2e[t,a,s,b] += eri_mo[nob+a,t,s,nob+b]
            A2_2e[t,a,t,b] -= eri_mo[nob+a,s,s,nob+b]

    for a in range(nvb):
        for i in range(nob):
            A2_2e[s,a,i,vt] += eri_mo[nob+a,t,i,s] - eri_mo[nob+a,s,i,t]
            A2_2e[t,a,i,vs] += eri_mo[nob+a,s,i,t] - eri_mo[nob+a,t,i,s]
            A2_2e[i,vs,t,a] += eri_mo[s,nob+a,t,i] - eri_mo[s,i,t,nob+a]
            A2_2e[i,vt,s,a] += eri_mo[s,i,t,nob+a] - eri_mo[s,nob+a,t,i]
    for a in range(*nva):
        for i in range(noa):
            A2_2e[s,a,i,vt] += eri_mo[nob+a,t,i,s] - eri_mo[nob+a,s,i,t]
            A2_2e[t,a,i,vs] += eri_mo[nob+a,s,i,t] - eri_mo[nob+a,t,i,s]
            A2_2e[i,vs,t,a] += eri_mo[s,nob+a,t,i] - eri_mo[s,i,t,nob+a]
            A2_2e[i,vt,s,a] += eri_mo[s,i,t,nob+a] - eri_mo[s,nob+a,t,i]

    for a in range(nvb):
        for i in range(noa):
            A2_2e[i,a,s,vs] -= eri_mo[nob+a,t,t,i]
            A2_2e[i,a,t,vs] += eri_mo[nob+a,s,t,i]
            A2_2e[i,a,s,vt] += eri_mo[nob+a,t,s,i]
            A2_2e[i,a,t,vt] -= eri_mo[nob+a,s,s,i]

            A2_2e[s,vs,i,a] -= eri_mo[i,t,t,nob+a]
            A2_2e[s,vt,i,a] += eri_mo[i,s,t,nob+a]
            A2_2e[t,vs,i,a] += eri_mo[i,t,s,nob+a]
            A2_2e[t,vt,i,a] -= eri_mo[i,s,s,nob+a]
    for a in range(*nva):
        for i in range(nob):
            A2_2e[s,vs,i,a] -= eri_mo[i,t,t,nob+a]
            A2_2e[s,vt,i,a] += eri_mo[i,s,t,nob+a]
            A2_2e[t,vs,i,a] += eri_mo[i,t,s,nob+a]
            A2_2e[t,vt,i,a] -= eri_mo[i,s,s,nob+a]

            A2_2e[i,a,s,vs] -= eri_mo[nob+a,t,t,i]
            A2_2e[i,a,t,vs] += eri_mo[nob+a,s,t,i]
            A2_2e[i,a,s,vt] += eri_mo[nob+a,t,s,i]
            A2_2e[i,a,t,vt] -= eri_mo[nob+a,s,s,i]



    # symmetric approach
    S3 = np.zeros((noa, nvb, noa, nvb)) # ia,jb
    A3_1e = np.zeros((noa, nvb, noa, nvb)) # ia,jb
    A3_2e = np.zeros((noa, nvb, noa, nvb)) # ia,jb
    A4_1e = np.zeros((noa, nvb, noa, nvb)) # ia,jb
    A4_2e = np.zeros((noa, nvb, noa, nvb)) # ia,jb

    # diagonal
    for a in range(*nv):
        S3[t,a,t,a] += 1.
        for i in range(no):
            S3[i,a,i,a] += 1.
    for i in range(no):
        S3[i,vs,i,vs] += 1.
    S3[t,vs,t,vs] += 1.


    # Closed-shell |0> has the core and s doubly occupied.
    h_mo = mo_coeff.T.conj() @ mf.get_hcore() @ mo_coeff
    F0 = h_mo.copy()
    for k in range(s+1):
        F0 += 2 * eri_mo[:,:,k,k] - eri_mo[:,k,k,:]

    # i belongs to C+s, a belongs to t+V
    for a in range(*nv):
        A3_1e[t,a,t,vs] += F0[s,nob+a]
        A3_1e[t,vs,t,a] += F0[nob+a,s]
        A3_1e[t,a,t,a] -= F0[s,s]
        for b in range(*nv):
            A3_1e[t,a,t,b] += F0[nob+a,nob+b]

    for i in range(no):
        A3_1e[i,vs,t,vs] -= F0[t,i]
        A3_1e[t,vs,i,vs] -= F0[i,t]
        A3_1e[i,vs,i,vs] += F0[t,t]
        for j in range(no):
            A3_1e[i,vs,j,vs] -= F0[j,i]

    for a in range(*nv):
        for i in range(no):
            A3_1e[i,a,i,vs] += F0[s,nob+a]
            A3_1e[i,vs,i,a] += F0[nob+a,s]
            A3_1e[i,a,t,a] -= F0[t,i]
            A3_1e[t,a,i,a] -= F0[i,t]
            A3_1e[i,a,i,a] += F0[t,t] - F0[s,s]

            for j in range(no):
                A3_1e[i,a,j,a] -= F0[j,i]
            for b in range(*nv):
                A3_1e[i,a,i,b] += F0[nob+a,nob+b]


    # Accumulate the ordered X_ia X_jb coefficients, then symmetrize.
    for a in range(*nv):
        for i in range(no):
            A3_2e[i,a,t,vs] -= eri_mo[nob+a,s,t,i]
            A3_2e[t,vs,i,a] -= eri_mo[i,t,s,nob+a]

            A3_2e[i,a,i,vs] += eri_mo[nob+a,s,t,t]
            A3_2e[i,vs,i,a] += eri_mo[s,nob+a,t,t]

            A3_2e[i,a,i,a] -= eri_mo[s,s,t,t]

            A3_2e[i,a,t,a] += eri_mo[s,s,t,i]
            A3_2e[t,a,i,a] += eri_mo[i,t,s,s]

            A3_2e[t,a,i,vs] -= eri_mo[nob+a,s,i,t]
            A3_2e[i,vs,t,a] -= eri_mo[s,nob+a,t,i]

            for j in range(no):
                for b in range(*nv):
                    A3_2e[i,a,j,b] -= eri_mo[nob+a,nob+b,j,i]

                A3_2e[i,a,j,a] += (eri_mo[j,t,t,i] - eri_mo[j,i,t,t]
                                    + eri_mo[j,i,s,s])
                A3_2e[i,a,j,vs] -= eri_mo[nob+a,s,j,i]
                A3_2e[j,vs,i,a] -= eri_mo[i,j,s,nob+a]
            for b in range(*nv):
                A3_2e[i,a,i,b] += (eri_mo[nob+a,nob+b,t,t]
                                    + eri_mo[nob+a,s,s,nob+b]
                                    - eri_mo[nob+a,nob+b,s,s])
                A3_2e[i,a,t,b] -= eri_mo[nob+a,nob+b,t,i]
                A3_2e[t,b,i,a] -= eri_mo[nob+b,nob+a,i,t]

    for a in range(*nv):
        for b in range(*nv):
            A3_2e[t,a,t,b] += (eri_mo[nob+a,s,s,nob+b]
                                - eri_mo[nob+a,nob+b,s,s])

    for i in range(no):
        for j in range(no):
            A3_2e[i,vs,j,vs] += (eri_mo[j,t,t,i] - eri_mo[j,i,t,t])


    # coupling
    for a in range(*nv):
        A4_1e[t,vs,t,a] += F0[s,nob+a]
        A4_1e[t,a,t,vs] += F0[nob+a,s]
        A4_1e[s,a,s,vt] += F0[nob+a,t]
        A4_1e[s,vt,s,a] += F0[t,nob+a]
        A4_1e[t,vt,s,a] -= F0[s,nob+a]
        A4_1e[s,a,t,vt] -= F0[nob+a,s]
        A4_1e[t,a,s,vs] -= F0[nob+a,t]
        A4_1e[s,vs,t,a] -= F0[t,nob+a]

    for i in range(no):
        A4_1e[i,vs,t,vt] += F0[s,i]
        A4_1e[t,vt,i,vs] += F0[i,s]
        A4_1e[s,vs,i,vt] += F0[i,t]
        A4_1e[i,vt,s,vs] += F0[t,i]
        A4_1e[i,vt,s,vt] -= F0[s,i]
        A4_1e[s,vt,i,vt] -= F0[i,s]
        A4_1e[t,vs,i,vs] -= F0[i,t]
        A4_1e[i,vs,t,vs] -= F0[t,i]


    for i in range(no):
        for j in range(no):
            A4_2e[i,vs,j,vs] += eri_mo[j,t,t,i]
            A4_2e[i,vs,j,vt] -= eri_mo[j,t,s,i]
            A4_2e[i,vt,j,vs] -= eri_mo[j,s,t,i]
            A4_2e[i,vt,j,vt] += eri_mo[j,s,s,i]

    for a in range(*nv):
        for b in range(*nv):
            A4_2e[s,a,s,b] += eri_mo[nob+a,t,t,nob+b]
            A4_2e[s,a,t,b] -= eri_mo[nob+a,s,t,nob+b]
            A4_2e[t,a,s,b] -= eri_mo[nob+a,t,s,nob+b]
            A4_2e[t,a,t,b] += eri_mo[nob+a,s,s,nob+b]

    for a in range(*nv):
        for i in range(no):
            A4_2e[i,a,s,vs] += eri_mo[nob+a,t,t,i]
            A4_2e[i,a,t,vs] -= eri_mo[nob+a,s,t,i]
            A4_2e[i,a,s,vt] -= eri_mo[nob+a,t,s,i]
            A4_2e[i,a,t,vt] += eri_mo[nob+a,s,s,i]

            A4_2e[s,vs,i,a] += eri_mo[i,t,t,nob+a]
            A4_2e[t,vs,i,a] -= eri_mo[i,t,s,nob+a]
            A4_2e[s,vt,i,a] -= eri_mo[i,s,t,nob+a]
            A4_2e[t,vt,i,a] += eri_mo[i,s,s,nob+a]

            A4_2e[i,vs,t,a] += eri_mo[s,i,t,nob+a] - eri_mo[s,nob+a,t,i]
            A4_2e[i,vt,s,a] += eri_mo[s,nob+a,t,i] - eri_mo[s,i,t,nob+a]

            A4_2e[s,a,i,vt] += eri_mo[nob+a,s,i,t] - eri_mo[nob+a,t,i,s]
            A4_2e[t,a,i,vs] += eri_mo[nob+a,t,i,s] - eri_mo[nob+a,s,i,t]


    # Convert the crossing block from H-E0 to H-ET via its overlap -S2.
    e0_shift = Fb[s,s] - Fa[t,t] - eri_mo[s,s,t,t]


    sa, si, sb, sj = '', '', '', ''
    for i in range(noa):
        if i==0: si = 'c'
        elif i==1: si = 's'
        elif i==2: si = 't'
        for a in range(nvb):
            if a==0: sa = 's'
            elif a==1: sa = 't'
            else: sa = 'v'
            for j in range(noa):
                if j==0: sj = 'c'
                elif j==1: sj = 's'
                elif j==2: sj = 't'
                for b in range(nvb):
                    if b==0: sb = 's'
                    elif b==1: sb = 't'
                    else: sb = 'v'
                    print('%d %d %d %d: X_{%s%s} X_{%s%s} %9.6f %9.6f  %9.6f %9.6f'
                          % (a, i, b, j, sa, si, sb, sj, -.5*A2_1e[i,a,j,b], A4_1e[i,a,j,b], -.5*A2_2e[i,a,j,b], A4_2e[i,a,j,b]))



    S1 = S1.reshape(noa*nvb, noa*nvb)
    S2 = S2.reshape(noa*nvb, noa*nvb)
    S3 = S3.reshape(noa*nvb, noa*nvb)
    ishermitian('S1', S1)
    ishermitian('S2', S2)
    ishermitian('S3', S3)
    print(np.allclose(S1, S3))
    print(np.allclose(S3, np.eye(noa*nvb)))
    print_matrix('S+:', S1+S2)
    print_matrix('S-:', S1-S2)


    # compare diagonal
    A1_1e = A1_1e.reshape(noa*nvb, noa*nvb)
    A1_2e = A1_2e.reshape(noa*nvb, noa*nvb)
    A3_1e = A3_1e.reshape(noa*nvb, noa*nvb)
    A3_2e = A3_2e.reshape(noa*nvb, noa*nvb)
    A1 = A1_1e + A1_2e
    A3 = A3_1e + A3_2e + e0_shift * S3
    #print_matrix('symmetric diagonal 1e:', A3_1e)
    #print_matrix('asymmetric diagonal 1e:', A1_1e)
    print('A1=A3?', np.allclose(A1, A3))

    A2_1e = A2_1e.reshape(noa*nvb, noa*nvb) * (-.5)
    A2_2e = A2_2e.reshape(noa*nvb, noa*nvb) * (-.5)
    A4_1e = A4_1e.reshape(noa*nvb, noa*nvb)
    A4_2e = A4_2e.reshape(noa*nvb, noa*nvb)
    A2 = A2_1e + A2_2e
    A4 = A4_1e + A4_2e - e0_shift * S2 # S2 has oppisite phase to S4

    #ishermitian('A2_1e', A2_1e)
    #ishermitian('A4_1e', A4_1e)
    #print_matrix('A2_1e:', A2_1e)
    #print_matrix('A4_1e:', A4_1e)

    #ishermitian('A2_2e', A2_2e)
    #ishermitian('A4_2e', A4_2e)
    #print_matrix('A2_2e:', A2_2e)
    #print_matrix('A4_2e:', A4_2e)
    print('A2=A4?', np.allclose(A2, A4))


    def solve_gep_singular_s(A, S, tol_ratio=1e-10):
        evals, evecs = np.linalg.eigh(S)
        keep = evals > tol_ratio * evals.max()
        V = evecs[:, keep]
        Ared = V.T @ A @ V
        Sred = V.T @ S @ V
        w, v = eigh(Ared, Sred)
        X = V @ v
        return w, X

    w, vec1 = solve_gep_singular_s(A1+A2, S1-S2)
    print_matrix("Asymmetric (+) Eigenvalues (singular S):\n", w, 5)
    w, vec2 = solve_gep_singular_s(A1-A2, S1+S2)
    print_matrix("Asymmetric (-) Eigenvalues (singular S):\n", w, 5)
    w, vec3 = solve_gep_singular_s(A3+A4, S1-S2)
    print_matrix("Symmetric (+) Eigenvalues (singular S):\n", w, 5)
    w, vec4 = solve_gep_singular_s(A3-A4, S1+S2)
    print_matrix("Symmetric (-) Eigenvalues (singular S):\n", w, 5)


    vec = np.hstack((vec1, vec2)).reshape(noa, nvb, -1).transpose(2,0,1)
    nstates = vec.shape[0]

    # find transition density
    detach_a = np.zeros((nstates, nstates, noa, noa))
    attach_a = np.zeros((nstates, nstates, nvb, nvb))
    detach_s = np.zeros((nstates, nstates, noa, noa))
    attach_s = np.zeros((nstates, nstates, nvb, nvb))

    # asymmetric approach
    # diagonal
    detach_a = -np.einsum('mia,nja->mnij', vec, vec)
    attach_a = np.einsum('mia,nib->mnab', vec, vec)
    detach_a_diag = detach_a.copy()
    attach_a_diag = attach_a.copy()
    # crossing
    detach_a[:,:,s,:] += np.einsum('mi,n->mni', vec[:,:,vt], vec[:,s,vt])
    detach_a[:,:,s,:] -= np.einsum('mi,n->mni', vec[:,:,vs], vec[:,t,vt])
    detach_a[:,:,t,:] += np.einsum('mi,n->mni', vec[:,:,vs], vec[:,t,vs])
    detach_a[:,:,t,:] -= np.einsum('mi,n->mni', vec[:,:,vt], vec[:,s,vs])

    detach_a[:,:,:,s] += np.einsum('m,ni->mni', vec[:,s,vt], vec[:,:,vt])
    detach_a[:,:,:,s] -= np.einsum('m,ni->mni', vec[:,t,vt], vec[:,:,vs])
    detach_a[:,:,:,t] += np.einsum('m,ni->mni', vec[:,t,vs], vec[:,:,vs])
    detach_a[:,:,:,t] -= np.einsum('m,ni->mni', vec[:,s,vs], vec[:,:,vt])

    attach_a[:,:,:,vs] += np.einsum('ma,n->mna', vec[:,s,:], vec[:,t,vt])
    attach_a[:,:,:,vs] -= np.einsum('ma,n->mna', vec[:,t,:], vec[:,t,vs])
    attach_a[:,:,:,vt] += np.einsum('ma,n->mna', vec[:,t,:], vec[:,s,vs])
    attach_a[:,:,:,vt] -= np.einsum('ma,n->mna', vec[:,s,:], vec[:,s,vt])

    attach_a[:,:,vs,:] += np.einsum('m,na->mna', vec[:,t,vt], vec[:,s,:])
    attach_a[:,:,vs,:] -= np.einsum('m,na->mna', vec[:,t,vs], vec[:,t,:])
    attach_a[:,:,vt,:] += np.einsum('m,na->mna', vec[:,s,vs], vec[:,t,:])
    attach_a[:,:,vt,:] -= np.einsum('m,na->mna', vec[:,s,vt], vec[:,s,:])

    # The spin-image terms also run over beta core orbitals and alpha
    # external virtual orbitals in the one-electron crossing matrix.
    detach_a[:,:,s,:nob] += np.einsum('mi,n->mni', vec[:,:nob,vt], vec[:,s,vt])
    detach_a[:,:,s,:nob] -= np.einsum('mi,n->mni', vec[:,:nob,vs], vec[:,t,vt])
    detach_a[:,:,t,:nob] += np.einsum('mi,n->mni', vec[:,:nob,vs], vec[:,t,vs])
    detach_a[:,:,t,:nob] -= np.einsum('mi,n->mni', vec[:,:nob,vt], vec[:,s,vs])

    detach_a[:,:,:nob,s] += np.einsum('m,ni->mni', vec[:,s,vt], vec[:,:nob,vt])
    detach_a[:,:,:nob,s] -= np.einsum('m,ni->mni', vec[:,t,vt], vec[:,:nob,vs])
    detach_a[:,:,:nob,t] += np.einsum('m,ni->mni', vec[:,t,vs], vec[:,:nob,vs])
    detach_a[:,:,:nob,t] -= np.einsum('m,ni->mni', vec[:,s,vs], vec[:,:nob,vt])

    attach_a[:,:,2:,vs] += np.einsum('ma,n->mna', vec[:,s,2:], vec[:,t,vt])
    attach_a[:,:,2:,vs] -= np.einsum('ma,n->mna', vec[:,t,2:], vec[:,t,vs])
    attach_a[:,:,2:,vt] += np.einsum('ma,n->mna', vec[:,t,2:], vec[:,s,vs])
    attach_a[:,:,2:,vt] -= np.einsum('ma,n->mna', vec[:,s,2:], vec[:,s,vt])

    attach_a[:,:,vs,2:] += np.einsum('m,na->mna', vec[:,t,vt], vec[:,s,2:])
    attach_a[:,:,vs,2:] -= np.einsum('m,na->mna', vec[:,t,vs], vec[:,t,2:])
    attach_a[:,:,vt,2:] += np.einsum('m,na->mna', vec[:,s,vs], vec[:,t,2:])
    attach_a[:,:,vt,2:] -= np.einsum('m,na->mna', vec[:,s,vt], vec[:,s,2:])
    detach_a_cross = detach_a - detach_a_diag
    attach_a_cross = attach_a - attach_a_diag


    # symmetric approach
    # diagonal
    detach_s[:,:,:no,:no] -= np.einsum('mia,nja->mnij', vec[:,:no,:], vec[:,:no,:])
    attach_s[:,:,1:,1:] += np.einsum('mia,nib->mnab', vec[:,:,1:], vec[:,:,1:])

    detach_s[:,:,s,s] -= np.einsum('mia,nia->mn', vec[:,:,1:], vec[:,:,1:])
    attach_s[:,:,vt,vt] += np.einsum('mia,nia->mn', vec[:,:no,:], vec[:,:no,:])

    detach_s[:,:,:no,t] -= np.einsum('mia,na->mni', vec[:,:no,:], vec[:,t,:])
    detach_s[:,:,t,:no] -= np.einsum('ma,nia->mni', vec[:,t,:], vec[:,:no,:])
    attach_s[:,:,1:,vs] += np.einsum('mia,ni->mna', vec[:,:,1:], vec[:,:,vs])
    attach_s[:,:,vs,1:] += np.einsum('mi,nia->mna', vec[:,:,vs], vec[:,:,1:])
    detach_s_diag = detach_s.copy()
    attach_s_diag = attach_s.copy()

    # coupling
    detach_s[:,:,s,:no] += np.einsum('mi,n->mni', vec[:,:no,vs], vec[:,t,vt])
    detach_s[:,:,s,:no] -= np.einsum('mi,n->mni', vec[:,:no,vt], vec[:,s,vt])
    detach_s[:,:,t,:no] += np.einsum('mi,n->mni', vec[:,:no,vt], vec[:,s,vs])
    detach_s[:,:,t,:no] -= np.einsum('mi,n->mni', vec[:,:no,vs], vec[:,t,vs])

    detach_s[:,:,:no,s] += np.einsum('m,ni->mni', vec[:,t,vt], vec[:,:no,vs])
    detach_s[:,:,:no,s] -= np.einsum('m,ni->mni', vec[:,s,vt], vec[:,:no,vt])
    detach_s[:,:,:no,t] += np.einsum('m,ni->mni', vec[:,s,vs], vec[:,:no,vt])
    detach_s[:,:,:no,t] -= np.einsum('m,ni->mni', vec[:,t,vs], vec[:,:no,vs])

    attach_s[:,:,vs,1:] += np.einsum('m,na->mna', vec[:,t,vs], vec[:,t,1:])
    attach_s[:,:,vs,1:] -= np.einsum('m,na->mna', vec[:,t,vt], vec[:,s,1:])
    attach_s[:,:,vt,1:] += np.einsum('m,na->mna', vec[:,s,vt], vec[:,s,1:])
    attach_s[:,:,vt,1:] -= np.einsum('m,na->mna', vec[:,s,vs], vec[:,t,1:])

    attach_s[:,:,1:,vs] += np.einsum('ma,n->mna', vec[:,t,1:], vec[:,t,vs])
    attach_s[:,:,1:,vs] -= np.einsum('ma,n->mna', vec[:,s,1:], vec[:,t,vt])
    attach_s[:,:,1:,vt] += np.einsum('ma,n->mna', vec[:,s,1:], vec[:,s,vt])
    attach_s[:,:,1:,vt] -= np.einsum('ma,n->mna', vec[:,t,1:], vec[:,s,vs])
    detach_s_cross = detach_s - detach_s_diag
    attach_s_cross = attach_s - attach_s_diag


    nao = mo_coeff.shape[0]
    P1 = np.zeros((nstates,nstates,nao,nao))
    P2 = np.zeros_like(P1)
    triplet_ref = 2 * mo_coeff[:,:nob] @ mo_coeff[:,:nob].T.conj()
    triplet_ref += np.outer(mo_coeff[:,s],mo_coeff[:,s].conj())
    triplet_ref += np.outer(mo_coeff[:,t],mo_coeff[:,t].conj())
    closed_minus_triplet = (np.outer(mo_coeff[:,s],mo_coeff[:,s].conj())
                            - np.outer(mo_coeff[:,t],mo_coeff[:,t].conj()))
    vec_flat = vec.reshape(nstates,noa*nvb)
    nplus = vec1.shape[1]

    # The (+) and (-) eigenvectors are different spin-adapted sectors.
    # A spin-independent density has no matrix elements between sectors.
    for sign, states in ((1,slice(0,nplus)),(-1,slice(nplus,nstates))):
        da = detach_a_diag[states,states] - .5*sign*detach_a_cross[states,states]
        aa = attach_a_diag[states,states] - .5*sign*attach_a_cross[states,states]
        ds = detach_s_diag[states,states] + sign*detach_s_cross[states,states]
        ass = attach_s_diag[states,states] + sign*attach_s_cross[states,states]

        P1[states,states] = np.einsum('mnij,pi,qj->mnpq',da,mo_coeff[:,:noa],mo_coeff[:,:noa])
        P1[states,states] += np.einsum('mnab,pa,qb->mnpq',aa,mo_coeff[:,nob:],mo_coeff[:,nob:])
        P2[states,states] = np.einsum('mnij,pi,qj->mnpq',ds,mo_coeff[:,:noa],mo_coeff[:,:noa])
        P2[states,states] += np.einsum('mnab,pa,qb->mnpq',ass,mo_coeff[:,nob:],mo_coeff[:,nob:])

        overlap = vec_flat[states] @ (S1-sign*S2) @ vec_flat[states].T
        P1[states,states] += overlap[:,:,None,None] * triplet_ref
        P2[states,states] += overlap[:,:,None,None] * (triplet_ref+closed_minus_triplet)

    print('is P1=P2?', np.allclose(P1, P2))

    # find transition dipole moments
    M = mf.mol.intor_symmetric('int1e_r', comp=3)
    dipoles = np.einsum('mnpq,xpq->mnx', P1, M)
    print_matrix('Transition dipole moments (a.u.):', dipoles[0,1:])
