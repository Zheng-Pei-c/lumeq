import sys
import numpy as np

from functools import reduce
from pyscf import lib
from pyscf.hessian import thermo
from pyscf.data import nist

from lumeq.utils.pyscf_helper import *
from lumeq.utils import convert_units, print_matrix
from lumeq.polariton.qed_ks import polariton_cs
from lumeq.polariton.qed_ks_grad import get_multipole_matrix_d1, finite_difference
from lumeq.plot.vibrational_spectra import get_dipole_dev, infrared

"""
diagonalize the matter and photon Hessian here
"""

LINDEP_THRESHOLD = 1e-7

def project_trans_rotation(mol, hess, exclude_trans=True, exclude_rot=True,
                           mass=None):
    '''Each column is one mode
    '''
    if mass is None:
        mass = mol.atom_mass_list(isotope_avg=True)
    atom_coords = mol.atom_coords()
    mass_center = np.einsum('z,zx->x', mass, atom_coords) / mass.sum()
    atom_coords = atom_coords - mass_center
    natm = atom_coords.shape[0]

    mass_hess = np.einsum('pqxy,p,q->pqxy', hess, mass**-.5, mass**-.5)
    h = mass_hess.transpose(0,2,1,3).reshape(natm*3,natm*3)

    TR = thermo._get_TR(mass, atom_coords)
    TRspace = []
    if exclude_trans:
        TRspace.append(TR[:3])

    if exclude_rot:
        rot_const = thermo.rotation_const(mass, atom_coords)
        rotor_type = thermo._get_rotor_type(rot_const)
        if rotor_type == 'ATOM':
            pass
        elif rotor_type == 'LINEAR':  # linear molecule
            TRspace.append(TR[3:5])
        else:
            TRspace.append(TR[3:])

    if TRspace:
        TRspace = np.vstack(TRspace)
        q, r = np.linalg.qr(TRspace.T)
        P = np.eye(natm * 3) - q.dot(q.T)
        w, v = np.linalg.eigh(P)
        bvec = v[:,w > LINDEP_THRESHOLD]
        h = reduce(np.dot, (bvec.T, h, bvec))
        force_const_au, mode = np.linalg.eigh(h)
        mode = bvec.dot(mode)
    else:
        force_const_au, mode = np.linalg.eigh(h)
        bvec = None

    return h, bvec, force_const_au, mode


def get_g1_d1(mf, frequency, hessobj):
    # frequency is the photon frequency
    if isinstance(frequency, list) or isinstance(frequency, np.ndarray):
        if len(frequency) != len(mf.c_lambda):
            raise ValueError('the number of frequencies does not match photon mode')

    mol = mf.mol
    natm = mol.natm
    atmlst = range(mol.natm)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mocc = mo_coeff[:,mo_occ>0]

    dipole = np.einsum('...pq,...->pq...', mf.dipole, frequency) # combine with frequency
    dm = mf.make_rdm1()

    dipole_d1, _ = get_multipole_matrix_d1(mol, mf.c_lambda, mf.origin)
    dipole_d1 = np.einsum('xpq...,...->xpq...', dipole_d1, frequency)

    mo1 = lib.chkfile.load(hessobj.chkfile, 'scf_mo1')
    mo1 = {int(k): mo1[k] for k in mo1}

    g1 = [None]*natm
    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        dm1 = np.einsum('xpi,qi->xpq', mo1[ia], mocc)

        g1[k] = np.einsum('pq...,xqp->x...', dipole, dm1)*2. # 2 for dm1
        g1[k] += np.einsum('xpq...,qp->x...', dipole_d1[:,p0:p1], dm[:,p0:p1])

    return 2.*np.array(g1)


def harmonic_analysis(mol, force_const_au, norm_mode, g1, frequency,
                      imaginary_freq=True, mass=None):
    if mass is None:
        mass = mol.atom_mass_list(isotope_avg=True)

    w2 = np.einsum('...,...->...', frequency, frequency)
    if isinstance(w2, float): w2 = [w2]

    # last dimension is for photon mode
    g1 = np.einsum('zr...,izr->i...', g1, norm_mode).reshape(norm_mode.shape[0], -1)

    n1, n2 = g1.shape
    #print(n1, n2)
    hess2 = np.zeros((n1+n2, n1+n2))
    hess2[:n1,:n1] += np.diag(force_const_au)
    hess2[:n1,n1:] += g1
    hess2[n1:,:n1] += g1.T
    hess2[n1:,n1:] += np.diag(w2)
    print_matrix('hess2:', hess2)

    force_const_au, mode0 = np.linalg.eigh(hess2)


    results = {}
    freq_au = np.lib.scimath.sqrt(force_const_au)
    results['freq_error'] = np.count_nonzero(freq_au.imag > 0)
    if not imaginary_freq and np.iscomplexobj(freq_au):
        # save imaginary frequency as negative frequency
        freq_au = freq_au.real - abs(freq_au.imag)

    results['freq_au'] = freq_au
    au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * np.pi)
    results['freq_wavenumber'] = freq_au * au2hz / nist.LIGHT_SPEED_SI * 1e-2

    norm_mode = np.einsum('izr,ij->jzr', norm_mode, mode0[:n1])
    results['norm_mode'] = norm_mode # (mode, atom, xyz)
    results['total_mode'] = np.concatenate((norm_mode.reshape(-1,mol.natm*3).T, mode0[n1:]), axis=0)

    reduced_mass = 1./np.einsum('izr,izr->i', norm_mode, norm_mode)
    results['reduced_mass'] = reduced_mass

    # https://en.wikipedia.org/wiki/Vibrational_temperature
    results['vib_temperature'] = freq_au * au2hz * nist.PLANCK / nist.BOLTZMANN

    # force constants
    dyne = 1e-2 * nist.HARTREE2J / nist.BOHR_SI**2
    results['force_const_au'] = force_const_au
    results['force_const_dyne'] = reduced_mass * force_const_au * dyne  #cm^-1/a0^2

    return results



if __name__ == '__main__':
    #infile = 'h2o.in'
    #parameters = parser(infile)

    #charge, spin, atom = parameters.get(section_names[0])[1:4]
    #functional, basis = get_rem_info(parameters.get(section_names[1]))[:2]
    #mol = build_molecule(atom, basis, charge, spin, verbose=0)

    hf = """
            H    0. 0. -0.459
            F    0. 0.  0.459
    """
    h2o = """
            H    1.6090032   -0.0510674    0.4424329
            O    0.8596350   -0.0510674   -0.1653507
            H    0.1102668   -0.0510674    0.4424329
    """
    co2 = """
            O    0. 0. 0.386715
            C    0. 0. 1.550000
            O    0. 0. 2.713285
    """
    feco5 = """
           Fe    0.       0.         0.002909
           C     0.       0.         1.809744
           C     1.811959 0.         0.001959
           C     0.       1.565109  -0.899762
           C     0.      -1.565109  -0.899762
           C    -1.811959 0.         0.001959
           O     0.       0.         2.955158
           O     0.       2.557516  -1.471811
           O     0.      -2.557516  -1.471811
           O     2.953719 0.         0.000929
           O    -2.953719 0.         0.000929
    """
    atom = locals()[sys.argv[1]] if len(sys.argv) > 1 else hf

    functional = 'hf'
    mol = build_molecule(atom, '6-31g')

    #frequency = 0.42978 # gs doesn't depend on frequency
    frequency = 0.45726295
    coupling = np.array([.0, .0, .03])

    mf = polariton_cs(mol) # in coherent state
    mf.xc = functional
    mf.grids.prune = True
    mf.get_multipole_matrix(coupling)

    e_tot = mf.kernel()
    hessobj = mf.Hessian()
    h = hessobj.kernel()

    dip_dev = get_dipole_dev(mf, hessobj)
    print_matrix('dip_dev:', dip_dev, 5, 1)

    results = thermo.harmonic_analysis(mol, h) # only molecular block
    force_const_au = results['force_const_au']
    norm_mode = results['norm_mode']

    print_matrix('freq_au:', results['freq_au'])
    print_matrix('freq_wavenumber:', results['freq_wavenumber'])
    #print_matrix('force_const_dyne:', results['force_const_dyne'])
    print_matrix('mode:', results['norm_mode'].reshape(len(results['freq_au']), -1).T)
    sir = infrared(dip_dev, results['norm_mode'])
    print_matrix('infrared intensity:', sir)

    if np.any(np.abs(coupling) > 1e-4):

        d1 = get_g1_d1(mf, frequency, hessobj)
        #print_matrix('d1:', d1, 5, 1)
        results = harmonic_analysis(mol, force_const_au, norm_mode, d1, frequency)
        print_matrix('freq_wavenumber:', results['freq_wavenumber'])
        #print_matrix('force_const_dyne:', results['force_const_dyne'])
        print_matrix('total mode:', results['total_mode'])

        sir = infrared(dip_dev, results['norm_mode'])
        print_matrix('infrared intensity:', sir)
