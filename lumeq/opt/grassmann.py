from lumeq import sys, np
from scipy.linalg import expm

from lumeq.utils import print_matrix
from lumeq.utils import get_ortho_basis

np.set_printoptions(precision=4, linewidth=200, suppress=True,)

def geodesic_exp(T, full=True, scale=1.):
    from scipy.linalg import expm

    if scale != 1.: T = scale * T # never short-handed!

    if full:
        return expm(T)
    else:
        v, o = T.shape # lower triangle
        exp = np.zeros((o+v, o+v))
        exp[o:, :o] = np.copy(T)
        exp[:o, o:] = -np.copy(T.T)
        return expm(exp)


def cs_decompose(T, full=True, scale=1.):
    U, s, Vt = np.linalg.svd(T, full_matrices=full)
    #print('singular values:', s, np.cos(s), np.sin(s))
    #print_matrix('Ut*U:', np.einsum('ji,jk->ik', U, U))
    #print_matrix('Vt*V:', np.einsum('ij,kj->ik', Vt, Vt))

    if scale != 1.: s = scale * s
    c, s = np.cos(s), np.sin(s)
    return U, c, s, Vt


def geodesic_svd(B, scale=1.):
    v, o = B.shape
    U, c, s, Vt = cs_decompose(B, scale=scale)

    z = np.zeros((v, o))
    bas = np.block([[Vt.T, z.T],
                    [z, U]])

    k, l = np.min([v,o]), np.abs(v-o)
    z = np.zeros((l, k))
    c, s = np.diag(c), np.diag(s)

    if v > o:
        cs = np.block([[c, -s, z.T],
                       [s, c, z.T],
                       [z, z, np.eye(l)]])
    else:
        cs = np.block([[c, z.T, -s],
                       [z, np.eye(l), z],
                       [s, z.T, c]])

    return np.einsum('ij,jk,lk->il', bas, cs, bas)


def geodesic_svd_compact(Y, T):
    U, c, s, Vt = cs_decompose(T, full=False)

    Y1 = np.einsum('ji,j,jk->ik', Vt, c, Vt)
    Y1 = np.einsum('ij,jk->ik', Y, Y1)
    Y1 += np.einsum('ij,j,jk->ik', U, s, Vt)
    return Y1



class Grassmann(object):
    def __init__(self, mf=None, S=None, Q=None, Y=None, P=None, update_method='steepest_descent'):
        self._mf = mf
        nocc = mf.mol.nelectron // 2

        if S is None: S = mf.get_ovlp()
        if Q is None: Q = mf.mo_coeff
        if Y is None: Y = mf.mo_coeff[:, :nocc]
        #P = mf.get_init_guess() *.5
        if P is None: P = mf.make_rdm1() *.5

        self.nocc = nocc
        self.nvir = Q.shape[1] - nocc
        print('nocc:', self.nocc, 'nvir:', self.nvir)

        self.S = S
        self._Q = Q
        self._Y = Y
        self._P = P
        self.L, self.Z, self.sinv = get_ortho_basis(S)

        self.update_method = update_method

        self.max_cycle = 50 # default
        self.debug = 1

        self.energy_tot = mf.energy_tot # object function
        self.get_fock = mf.get_fock     # function gradient


    def barzilai_borwein_step(self, G0, G1, T0):
        dG = G1 - G0
        alpha = np.einsum('ij,ij->', dG, T0) / np.einsum('ij,ij->', dG, dG)
        return -alpha*G1


    def polak_ribiere_step(self, G0, G1, T0):
        dG = G1 - G0
        beta = np.einsum('ij,ij->', dG, G1) / np.einsum('ij,ij->', G0, G0)
        return (beta*T0) - G1


    def init_guess(self):
        raise NotImplementedError


    def kernel(self, C=None, tol=8):
        tol = np.power(1, -float(tol))

        if C is None:
            C = self.init_guess()
        print_matrix('C initial:\n', C)

        #P = np.einsum('ij,jk,lk->il', C, self.I, C)
        #P = np.einsum('ji,jk,kl->il', self.Z, P, self.Z)
        P = self._P
        energy = self.energy_tot(dm=P*2)

        gradient = self.get_fock(dm=P*2) *0.5
        G = self.project_tangent_space(C, gradient)
        T = -np.copy(G)
        C, P = self.update(C, T)

        for i in range(self.max_cycle):
            C_old = np.copy(C)
            G_old = np.copy(G)
            energy_old = np.copy(energy)

            gradient = self.get_fock(dm=P*2) *0.5
            G = self.project_tangent_space(C, gradient)
            print_matrix('tangent:\n', G)

            if self.update_method == 'barzilai_borwein':
                T = self.barzilai_borwein_step(G_old, G, T)
            elif self.update_method == 'conjugate_gradient':
                T = self.conjugate_gradient_step(C, G_old, G, T)
            C, P = self.update(C, T)

            energy = self.energy_tot(dm=P*2)
            error = np.linalg.norm(C-C_old)
            print('i:', i+1, 'error:', error, 'energy:', energy, energy-energy_old)

            if error < tol:
                break



class Projection_Grassmann(Grassmann):
    def init_guess(self):
        return self._P


    def project_tangent_space(self, P, G):
        # [P, [P, G]] at point P, where P^2=P and G is a symmetric matrix
        PG = np.einsum('ij,jk->ik', P, G)
        GP = np.einsum('ij,jk->ik', G, P)
        return PG + GP - 2.*np.einsum('ij,jk->ik', P, GP)



class Quotient_Grassmann(Grassmann):
    def init_guess(self):
        Y = np.copy(self._Y)
        Y = np.einsum('ji,jk->ik', self.L, Y)
        return Y


    def project_tangent_space(self, Y, G):
        # (I - YY^T) G Y
        P = np.einsum('ij,kj->ik', Y, Y)
        G = np.einsum('ij,jk->ik', G, Y)
        return np.einsum('ij,jk->ik', (np.eye(P.shape[0]) - P), G)


    def tangent_parallel_transport(self, Y, G, T):
        U, s, Vt = np.linalg.svd(T, full_matrices=False)
        sVt = np.einsum('i,ik->ik', s, Vt)
        c, s = np.cos(s), np.sin(s)

        dT = -np.einsum('ij,kj,k->ik', Y, Vt, s)
        dT += np.einsum('ij,j->ij', U, c)
        dG = dT - U
        dT = np.einsum('ij,jk->ik', dT, sVt)
        dG = np.einsum('ij,kj,kl->il', dG, U, G)
        dG += G
        return dT, dG


    def conjugate_gradient_step(self, Y, G0, G1, T0):
        dT, dG = self.tangent_parallel_transport(Y, G0, T0)
        gamma = np.einsum('ij,ij->', G1-dG, G1) / np.einsum('ij,ij->', G0, G0)
        return (gamma*dT) - G1


    def update(self, Y, T):
        Y = geodesic_svd_compact(Y, T)

        P = np.einsum('ij,kj->ik', Y, Y)
        P = np.einsum('ji,jk,kl->il', self.Z, P, self.Z) # AO

        return Y, P



class Involution_Grassmann(Grassmann):
    def init_guess(self):
        self.I = np.diag(np.concatenate((np.ones(self.nocc), -np.ones(self.nvir))))

        C = np.copy(self._Q)
        C = np.einsum('ji,jk->ik', self.L, C)
        return C


    def project_tangent_space(self, V, G):
        G = np.einsum('ij,jk,lk->il', self.Z, G, self.Z)
        return np.einsum('ji,jk,kl->il', V, G, V)[self.nocc:, :self.nocc]


    #def armijo_linesearch(self, G, Q):


    def update(self, C, T):
        #print('C:\n', C)
        #print('I:\n', I)
        #Q = np.einsum('ij,jk,lk->il', C, self.I, C)
        #print('trQ:\n', np.trace(Q))
        #print('Q:\n', Q)
        #print('QQ:\n', np.einsum('ij,jk->ik', Q, Q))

        #V, R = np.linalg.qr(.5*(np.eye(C.shape[0]) + Q))
        #print('V:\n', V)
        ##print('R:\n', R)

        #T = T[self.nocc:, :self.nocc]
        exp = geodesic_svd(T)
        if self.debug:
            diff = exp - geodesic_exp(T, False)
            if np.linalg.norm(diff) > 1e-8:
                raise ValueError('exponential svd is wrong.')
        #print_matrix('exp:', exp)

        C = np.einsum('ij,jk->ik', C, exp) # orthogonal
        #print_matrix('C:', C)
        print_matrix('C:', np.einsum('ji,jk->ik', self.Z, C))

        #P = np.einsum('ij,jk,lk->il', C, self.I, C)
        P = np.einsum('ij,kj->ik', C[:, :self.nocc], C[:, :self.nocc])
        P = np.einsum('ji,jk,kl->il', self.Z, P, self.Z) # AO

        return C, P



if __name__ == '__main__':
    from pyscf import gto, scf
    atom = """
    O       0.   0.     0.
    H       0.  -0.757  0.587
    H       0.   0.757  0.587
    """

    mol = gto.Mole(
            atom = atom,
            charge = 0,
            basis = 'sto-3g',
            )
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 1
    mf.kernel()

    models = ['projection', 'quotient', 'involution']
    update_methods = ['barzilai_borwein', 'conjugate_gradient']

    model, update_method = models[int(sys.argv[1])], update_methods[int(sys.argv[2])]

    if model == 'projection':
        grass = Projection_Grassmann(mf=mf)
    elif model == 'quotient':
        grass = Quotient_Grassmann(mf=mf, update_method=update_method)
    elif model == 'involution':
        grass = Involution_Grassmann(mf=mf, update_method=update_method)

#    grass.max_cycle = 1
    grass.kernel()


    #mf.verbose = 5
    mf.max_cycle = 50
    mf.kernel()
    print_matrix('C reference:', mf.mo_coeff)
    #mo = mf.mo_coeff
    #gradient = mf.get_fock(dm=mf.make_rdm1()) *0.5
    #print('Fock:', np.einsum('ji,jk,kl->il', mo, gradient, mo))
