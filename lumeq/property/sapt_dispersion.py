from lumeq import sys, np
from lumeq.utils import print_matrix
from lumeq.property.rdm_analysis import assemble_amplitudes

from pyscf.scf import _vhf

def get_xys_in_ao(mo_coeff, xs, ys, has_y=False, scale=1):
    # scale is 2 for restricted
    # and 1 for unrestricted case
    no = xs.shape[1]
    orbo, orbv = mo_coeff[:,:no], mo_coeff[:,no:]
    return np.einsum('xia,ma,ni->xmn', xs, orbv, orbo.conj()*scale)


def build_trans_density(mo_coeff, amps, itype='r', has_y=False):
    xs, ys = assemble_amplitudes(amps, itype=itype, rpa=has_y)

    if itype == 'r':
        return get_xys_in_ao(mo_coeff, xs, ys, itype, has_y, scale=2)
    elif itype == 'u':
        dms = [None]*2
        for s in range(2):
            dms[s] = get_xys_in_ao(mo_coeff[s], xs[s], ys[s], has_y)
        return dms


r"""
refer: 10.1021/acs.jctc.8b01058
    calculate dispersion energy between two molecules
    by second-order perturbation theory from many-body wavefunctions
    using the excitation energies, transition amplitudes,
    and 2e-opeartor Coulomb interactions
    E = sum_{ij} (< Psi_{A0} Psi_{B0} | V | Psi_{Ai} Psi_{Bj} >)^{2}
        / (E_{A0} + E_{B0} - E_{Ai} - E_{Bj})
"""
def cal_sapt(mols, mo_coeffs, amps, energies, omega=None, itype='r'):
    from pyscf.gto.mole import conc_mol
    # get supermolecule
    mol = conc_mol(mols[0], mols[1])

    nstates = [len(e) for e in energies]
    energies = np.array(energies[0])[:,None] + np.array(energies[1])

    print('start computing crossing coulomb interaction')
    # build state interactions
    func = eval('cal_sapt_'+itype)
    v = func(mol, mo_coeffs, amps, nstates, omega)
    v = -np.einsum('kl,kl,kl->', v, v, 1./energies)
    return v


def cal_sapt_r(mol, mo_coeffs, amps, nstates, omega=None):
    r"""
    mol is the supermolecule
    """
    nbas0, nbas1 = mo_coeffs[0].shape[0], mo_coeffs[1].shape[0]
    nbas = nbas0 + nbas1

    dms = [np.zeros((nstates[0], nbas0, nbas0)),
           np.zeros((nstates[1], nbas, nbas))]
    dms[0] = build_trans_density(mo_coeffs[0], amps[0], 'r')
    dms[1][:,nbas0:,nbas0:] = build_trans_density(mo_coeffs[1], amps[1], 'r')

    hermi = 1
    with_j, with_k = True, False
    with mol.with_range_coulomb(omega):
        vj, vk = _vhf.direct(dms[1], mol._atm, mol._bas, mol._env,
                             None, hermi, mol.cart, with_j, with_k)

    v = np.einsum('knm,lmn->kl', dms[0], vj[:,:nbas0,:nbas0])
    return v


def cal_sapt_u(mol, mo_coeffs, amps, nstates, omega=None):
    r"""
    mol is the supermolecule
    """
    nbas0, nbas1 = mo_coeffs[0][0].shape[0], mo_coeffs[1][0].shape[0]
    nbas = nbas0 + nbas1

    dms = [np.zeros((2, nstates[0], nbas0, nbas0)),
           np.zeros((2, nstates[1], nbas, nbas))]
    dms[0] = build_trans_density(mo_coeffs[0], amps[0], 'u')
    dms[1][:,:,nbas0:,nbas0:] = build_trans_density(mo_coeffs[1], amps[1], 'u')

    hermi = 0
    with_j, with_k = True, False
    with mol.with_range_coulomb(omega):
        vj, vk = _vhf.direct(dms[1], mol._atm, mol._bas, mol._env,
                             None, hermi, mol.cart, with_j, with_k)

    v = np.einsum('sknm,slmn->kl', dms[0], vj[:,:,:nbas0,:nbas0])
    return v


def run_tddft(symbols, coords, charge=1, spin=1, functional='hf', basis='sto-3g'):
    from pyscf import gto, scf, tdscf
    from lumeq.utils.pyscf_helper import build_atom

    atom = build_atom(symbols, coords)
    mol = gto.M(
            atom = atom,
            basis = basis,
            charge = charge,
            spin = spin,
            )

    mf = scf.UKS(mol)
    mf.xc = functional
    mf.kernel()
    print('no, norb:', mol.nelec, mf.mo_coeff.shape[1])

    nstates = 1000

    td = tdscf.TDA(mf)
    td.kernel(nstates=nstates)
    return mf, td


if __name__ == '__main__':
    from lumeq.utils.sec_mole import read_symbols_coords

    charge = 1
    spin = 1
    functional = 'hf'

    xyzfile = sys.argv[1]
    symbols, coords = read_symbols_coords(xyzfile)
    natom = len(symbols)//2

    mf0, td0 = run_tddft(symbols[:natom], coords[:natom], charge, spin, functional)
    mf1, td1 = run_tddft(symbols[natom:], coords[natom:], charge, spin, functional)

    mols = [mf0.mol, mf1.mol]
    mo_coeffs = [mf0.mo_coeff, mf1.mo_coeff]
    amps = [td0.xy, td1.xy]
    energies = [td0.e, td1.e]
    v = cal_sapt(mols, mo_coeffs, amps, energies, itype='u')
    print('sapt energy:', v)
