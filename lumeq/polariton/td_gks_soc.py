from lumeq import np
from lumeq.utils import print_matrix, ishermitian

from pyscf.scf import _response_functions
from pyscf.scf import ghf
from pyscf.tdscf import ghf as td_ghf

def make_h01_soc1e(mol, orbo=None, orbv=None, so_eff_charge=False, qed_fac=1):
    from pyscf.data import nist
    alpha2 = nist.ALPHA ** 2

    if so_eff_charge:
        from pyscf.prop.zfs.uhf import koseki_charge
        hso1e = 0
        for ia in range(mol.natm):
            Z = koseki_charge(mol.atom_charge(ia))
            mol.set_rinv_origin(mol.atom_coord(ia))
            hso1e += -Z * mol.intor_asymmetric('int1e_prinvxp', 3)
    else:
        hso1e = mol.intor_asymmetric('int1e_pnucxp', 3)
    hso1e *= qed_fac * (alpha2/4)
    return hso1e

#TODO: compare with GHF.x2c in scf.ghf.py


def _gen_ghf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, max_memory=None):
    r'''Generate a function to compute the product of GHF response function and
    GHF density matrices. add soc components
    '''
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol

    occidx = np.where(mo_occ==1)[0]
    viridx = np.where(mo_occ==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    hso1e = make_h01_soc1e(mol, orbo, orbv)
    hso1e_o = np.einsum('xpq,pi,qj->xij', hso1e, orbo.conj(), orbo)
    hso1e_v = np.einsum('xpq,pa,qb->xab', hso1e, orbv.conj(), orbv)

    vresp0 = ghf.GHF.gen_response(hermi=0)
    def vind(dm1):
        v1ao = vresp0(dm1)

        return v1ao

    return vind

def get_ab(mf, mo_energy=None, mo_coeff=None, mo_occ=None):
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol

    occidx = np.where(mo_occ==1)[0]
    viridx = np.where(mo_occ==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]

    a_soc = np.zeros((nocc,nvir,nocc,nvir), dtype=np.complex128)
    b_soc = np.zeros_like(a_soc)

    hso1e = make_h01_soc1e(mol, orbo, orbv) * (-1j)
    hso1e = np.array([np.block([[h,h],[h,h]]) for h in hso1e])
    print(hso1e.shape, orbo.shape, orbv.shape)
    hso1e_o = np.einsum('xpq,pi,qj->xij', hso1e, orbo.conj(), orbo)
    hso1e_v = np.einsum('xpq,pa,qb->xab', hso1e, orbv.conj(), orbv)

    nocc, nvir = nocc//2, nvir//2
    hso_oz = np.array([hso1e_o[2,:nocc,:nocc], -hso1e_o[2,nocc:,nocc:]])
    hso_op = hso1e_o[0,nocc:,:nocc] + 1j*hso1e_o[1,nocc:,:nocc]
    hso_om = hso1e_o[0,:nocc,nocc:] - 1j*hso1e_o[1,:nocc,nocc:]

    hso_vz = np.array([hso1e_v[2,:nvir,:nvir], -hso1e_v[2,nvir:,nvir:]])
    hso_vp = hso1e_v[0,nvir:,:nvir] + 1j*hso1e_v[1,nvir:,:nvir]
    hso_vm = hso1e_v[0,:nvir,nvir:] - 1j*hso1e_v[1,:nvir,nvir:]

    for i in range(nocc*2):
        a_soc[i,:nvir,i,:nvir] += hso_vz[0]
        a_soc[i,nvir:,i,nvir:] += hso_vz[1]
        a_soc[i,:nvir,i,nvir:] += hso_vm
        a_soc[i,nvir:,i,:nvir] += hso_vp

        b_soc[i,:nvir,i,:nvir] += hso_vz[0].T
        b_soc[i,nvir:,i,nvir:] += hso_vz[1].T
        b_soc[i,:nvir,i,nvir:] += hso_vm.T
        b_soc[i,nvir:,i,:nvir] += hso_vp.T

    for a in range(nvir*2):
        a_soc[:nocc,a,:nocc,a] -= hso_oz[0]
        a_soc[nocc:,a,nocc:,a] -= hso_oz[1]
        a_soc[:nocc,a,nocc:,a] -= hso_om
        a_soc[nocc:,a,:nocc,a] -= hso_op

        b_soc[:nocc,a,:nocc,a] -= hso_oz[0].T
        b_soc[nocc:,a,nocc:,a] -= hso_oz[1].T
        b_soc[:nocc,a,nocc:,a] -= hso_om.T
        b_soc[nocc:,a,:nocc,a] -= hso_op.T

    nov = nocc*nvir*4
    ishermitian('a_soc', a_soc.reshape((nov,-1)))
    ishermitian('b_soc', b_soc.reshape((nov,-1)))

    a, b = td_ghf.get_ab(mf, mo_energy, mo_coeff, mo_occ)
    eigval, eigvec = np.linalg.eig(a.reshape((nov,-1)))
    print_matrix('A eigval:', np.sort(eigval))

    #eigval, eigvec = np.linalg.eig(np.block([[a.reshape((nov,-1)), b.reshape((nov,-1))], [-b.reshape((nov,-1)).conj(), -a.reshape((nov,-1)).conj()]]))
    #print_matrix('A+B eigval:', np.sort(eigval))

    a, b = a+a_soc, b+b_soc
    eigval, eigvec = np.linalg.eig(a.reshape((nov,-1)))
    print_matrix('A eigval:', np.sort(eigval).real)

    #eigval, eigvec = np.linalg.eig(np.block([[a.reshape((nov,-1)), b.reshape((nov,-1))], [-b.reshape((nov,-1)).conj(), -a.reshape((nov,-1)).conj()]]))
    #print_matrix('A+B eigval:', np.sort(eigval).real)

    return a, b


class GHF(ghf.GHF):
    gen_response = _gen_ghf_response



if __name__ == '__main__':
    from lumeq.utils.pyscf_helper import *
    infile = 'h2o.in'
    parameters = parser(infile)

    charge, spin, atom = parameters.get(section_names[0])[1:4]
    functional, basis = get_rem_info(parameters.get(section_names[1]))[:2]

    atom = """
    H 0. 0. 0.
    F 0. 0. 0.714
    """
    basis = 'sto-3g'
    mol = build_molecule(atom, basis, charge, spin, verbose=0)

    mf = ghf.GHF(mol)
    e_tot = mf.kernel()
    a, b = get_ab(mf)

    #td = mf.TDA()
    #td.kernel(nstates=40)
    #print_matrix('excitation:', td.e)

    #td = mf.TDHF()
    #td.kernel(nstates=40)
    #print_matrix('excitation:', td.e)
