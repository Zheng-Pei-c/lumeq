from lumeq import sys, np
from lumeq.utils.pyscf_helper import *

from pyscf import scf, tdscf, gto, lib

def assemble_amplitudes(xy, nstates=None, rpa=False, itype='r'):
    r"""
    reshape the transition amplitudes from pyscf tdscf module

    Args:
        xy: ``tdscf.xy`` tuples.
        nstates (int, optional): Number of excited states.
        rpa (bool, optional): Whether to include the RPA ``Y`` amplitudes.
        itype (str): One of ``'r'``, ``'ro'``, or ``'u'``.

    Returns:
        tuple: Reshaped ``(xs, ys)`` transition amplitudes.
    """
    if nstates == 1:
        return reshape_xys([xy], None, itype)

    xs, ys = [], []
    if itype == 'u':
        xsa, xsb, ysa, ysb = [], [], [], []
        for i, _xy in enumerate(xy):
            (xia, xib), (yia, yib) = _xy
            xsa.append(xia)
            xsb.append(xib)
            ysa.append(yia)
            ysb.append(yib)
        xs, ys = [np.array(xsa), np.array(xsb)], [np.array(ysa), np.array(ysb)]

    else: # itype in {'r', 'ro'}:
        for i, _xy in enumerate(xy):
            xi, yi = _xy
            xs.append(xi)
            ys.append(yi)
        xs, ys = np.array(xs), np.array(ys)

    if not rpa:
        ys = None
    # (nroots, nocc, nvir)
    return xs, ys


def get_transition_dm(xy, coeff1, coeff2=None, scale=2.):
    r"""
    calculate transition 1-particle density matrix bewteen ground and excited states from xy amplitudes

    Args:
        xy (list): ``[xs, ys]`` amplitudes, where ``xs`` and ``ys`` are arrays
            with shape ``(nstates, nocc, nvir)``.
        coeff1 (numpy.ndarray): Ground-state molecular-orbital coefficients.
        coeff2 (numpy.ndarray, optional): Molecular-orbital coefficients for a
            different spin channel or geometry. Defaults to ``coeff1``.
        scale (float): ``2`` for restricted and ``1`` for unrestricted cases.

    Returns:
        numpy.ndarray: Transition one-particle density matrix.
    """
    if not isinstance(coeff2, np.ndarray):
        coeff2 = coeff1

    xs, ys = xy
    _, o, v = xs.shape
    rdm1 = np.einsum('pi,nia,qa->npq', coeff1[:,:o], xs, coeff2[:,-v:].conj())
    if isinstance(ys, np.ndarray): # add y contribution as transpose
        rdm1 += np.einsum('pi,nia,qa->nqp', coeff1[:,:o], ys, coeff2[:,-1:].conj())

    return scale * rdm1


def get_attach_dm(xy1, orbv1, xy2=None, orbv2=None, scale=2.):
    r"""calculate attach part of 1-particle density matrix from xy amplitudes bewteen excited states
    """
    xs1, ys1 = xy1
    if xy2 is None:
        xs2, ys2 = xs1, ys1
    else:
        xs2, ys2 = xy2
    if not isinstance(orbv2, np.ndarray):
        orbv2 = orbv1

    rdm1 = np.einsum('mia,nib->mnab', xs1, xs2)
    if isinstance(ys1, np.ndarray):
        rdm1 += np.einsum('mia,nib->mnba', ys1, ys2)
    #print('attach: %8.4f' % np.trace(rdm1))
    return scale * np.einsum('pa,mnab,qb->mnpq', orbv1, rdm1, orbv2.conj())


def get_detach_dm(xy1, orbo1, xy2=None, orbo2=None, scale=2.):
    r"""calculate detach part of 1-particle density matrix from xy amplitudes bewteen excited states"""
    xs1, ys1 = xy1
    if xy2 is None:
        xs2, ys2 = xs1, ys1
    else:
        xs2, ys2 = xy2
    if not isinstance(orbo2, np.ndarray):
        orbo2 = orbo1

    rdm1 = np.einsum('mia,nja->mnij', xs1, xs2)
    if isinstance(ys1, np.ndarray):
        rdm1 += np.einsum('mnia,mnja->mnji', ys1, ys2)
    #print('detach: %8.4f' % np.trace(rdm1))
    return -scale * np.einsum('pi,mnij,qj->mnpq', orbo1, rdm1, orbo2.conj())


def get_difference_dm(xy1, coeff1, xy2=None, coeff2=None, scale=2.):
    r"""
    calculate difference density matrix bewteen excited states

    Args:
        xy1 (list): ``[xs, ys]`` amplitudes for the first state or geometry.
        xy2 (list, optional): ``[xs, ys]`` amplitudes for the second state or
            geometry.
        coeff1 (numpy.ndarray): Molecular orbitals of the first state or
            geometry.
        coeff2 (numpy.ndarray, optional): Molecular orbitals of the second
            state, geometry, or spin channel.

    Returns:
        numpy.ndarray: Difference one-particle density matrix.
    """
    if not isinstance(coeff2, np.ndarray):
        coeff2 = coeff1

    _, o, v = xy1[0].shape # xs
    orbo1, orbv1 = coeff1[:,:o], coeff1[:,-v:]
    orbo2, orbv2 = coeff2[:,:o], coeff2[:,-v:]

    rdm1  = get_attach_dm(xy1, orbv1, xy2, orbv2, scale)
    rdm1 += get_detach_dm(xy1, orbo1, xy2, orbo2, scale)

    return rdm1


def cal_rdm1(xy1, coeff1, xy2=None, coeff2=None, scale=2., itype='trans'):
    r"""
    calculate 1-particle density matrices from xy amplitudes

    Args:
        xy1 (list): XY amplitudes for the first set of states.
        coeff1: Molecular orbitals for the first state set.
        xy2 (list, optional): XY amplitudes for the second set of states.
        coeff2 (numpy.ndarray, optional): Molecular orbitals for the second
            state set.
        scale (float): ``2`` for restricted and ``1`` for unrestricted cases.
        itype (str): Density-matrix type, e.g. ``'trans'`` or ``'diff'``.
    """
    rdm1 = []
    if 'trans' in itype:
        rdm1 = get_transition_dm(xy1, coeff1, coeff2, scale=scale)
    if 'diff' in itype:
        rdm1 = get_difference_dm(xy1, coeff1, xy2, coeff2, scale=scale)

    return rdm1


def cal_dipoles(dip_mat, rdm):
    r"""
    calculate dipole moments from dipole integrals and 1-particle density matrices

    Args:
        dip_mat (numpy.ndarray): Dipole integrals in AO basis with shape ``(3, nao, nao)``.
        rdm (numpy.ndarray): One-particle density matrices with shape ``(..., nao, nao)``.

    Returns:
        numpy.ndarray: Dipole moments with shape ``(..., 3)``.
    """
    return np.einsum('xpq,...pq->...x', dip_mat, rdm)



if __name__ == '__main__':
    infile = sys.argv[1]
    parameters = parser(infile)
    results = run_pyscf_final(parameters)
    mol, mf, td = results['mol'], results['mf'], results['td']
    if not isinstance(mol, list):
        mol, mf, td = [mol], [mf], [td]

    dipoles = []
    for n in range(len(mol)):
        #print_matrix('dipole reference:', td[n].transition_dipole())
        coeff = mf[n].mo_coeff
        xys = td[n].xy
        dip_mat = mol[n].intor('int1e_r', comp=3)
        rdm1 = cal_rdm1(xys, coeff, scale=2., itype='diff')
        dipole = cal_dipoles(dip_mat, rdm1)
        #dipoles.append(dipole)
        #print_matrix('dipole:', dipole)

        nstate = len(xys)
        dipole2 = np.zeros((nstate, nstate, 3))
        icount = 0
        for i in range(len(xys)):
            for j in range(i, len(xys)):
                dipole2[i,j] = dipole2[j,i] = dipole[icount]
                icount += 1

        print_matrix('dipole:', dipole2.reshape(nstate, -1))
    #dipoles = np.array(dipoles)
