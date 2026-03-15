import sys
import numpy as np

import pyscf
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft.rks import RKS

from lumeq.utils.pyscf_helper import *
from lumeq.utils import convert_units, print_matrix

def get_scaled_lambda(c_lambda, frequency, photon_coeff=1.):
    """
    return frequency-scaled coupling strength (c_lambda)
    """
    if isinstance(frequency, float):
        return np.sqrt(frequency/2.) * np.einsum('x,x->x', photon_coeff, c_lambda)
    else:
        return np.einsum('i,ix,ix->x', np.sqrt(frequency)/np.sqrt(2.), photon_coeff, c_lambda)


def get_lambda2(c_lambda): # lambda square for quadrupole contraction
    c2 = np.einsum('...x,...y->...xy', c_lambda, c_lambda)
    if c2.ndim == 3: # contract modes
        c2 = np.sum(c2, axis=0)
    return c2


def get_multipole_matrix(mol, itype='dipole', dipole=None, quadrupole=None,
                         c_lambda=None, origin=None):
    """
    c_lambda: (n_mode, 3) = coupling_strength * sqrt(2.*photon_frequency)
    """
    if origin is None:
        origin = np.zeros(3)
        #origin = get_center_of_mass(mol)
    if isinstance(c_lambda, list):
        c_lambda = np.array(c_lambda)

    if itype == 'all':
        if not isinstance(dipole, np.ndarray):
            itype += '_dipole'
        if not isinstance(quadrupole, np.ndarray):
            itype += '_quadrupole'

    with mol.with_common_orig(origin):
        if 'dipole' in itype:
            dipole = mol.intor('int1e_r', comp=3, hermi=0)

        if 'quadrupole' in itype:
            nao = mol.nao_nr()
            quadrupole = mol.intor('int1e_rr', comp=9, hermi=0).reshape(3,3,nao,nao)

    if isinstance(c_lambda, np.ndarray):
        if isinstance(dipole, np.ndarray):
            dipole = np.einsum('xpq,...x->...pq', dipole, c_lambda)

        if isinstance(quadrupole, np.ndarray):
            quadrupole = np.einsum('xypq,xy->pq', quadrupole, get_lambda2(c_lambda))

    multipoles = {'dipole': dipole, 'quadrupole': quadrupole}
    return multipoles


def get_dse_2e(dipole, dm, with_j=False, scale_k=.5): # c_lambda is included
    # scale k by 1/2 for restricted orbital case by default
    if dipole.ndim == 2:
        vk = np.einsum('pq,rs,...qr->...ps', dipole, dipole, dm)
        if with_j is False:
            return vk*scale_k
        else:
            vj = np.einsum('pq,rs,...rs->...pq', dipole, dipole, dm)
            return [vj, vk*scale_k]
    else: # contract modes
        vk = np.einsum('ipq,irs,...qr->...ps', dipole, dipole, dm)
        if with_j is False:
            return vk*scale_k
        else:
            vj = np.einsum('ipq,irs,...rs->...pq', dipole, dipole, dm)
            return [vj, vk*scale_k]


def get_dse_2e_xyz(dipole, dm, with_j=False, scale_k=.5): # xyz without coupling
    vk = np.einsum('xpq,yrs,...qr->...xyps', dipole, dipole, dm)
    if with_j is False:
        return vk*scale_k
    else:
        vj = np.einsum('xpq,yrs,...rs->...xypq', dipole, dipole, dm)
        return [vj, vk*scale_k]


def get_dse_elec_nuc(dipole, nuc_dip): # c_lambda is included
    if isinstance(nuc_dip, float):
        return -nuc_dip * dipole
    else: # numpy does not sum over ellipsis
        return -np.einsum('lpq,l->pq', dipole, nuc_dip) # l is the number of photon modes


def get_nuclear_dipoles(mol, c_lambda, origin=None):
    """
    lambda cdot nuclear_dipole
    """
    if origin is None:
        origin = np.zeros(3)

    charges = mol.atom_charges()
    # the subtraction is along the common axis, and already in bohr
    coords  = np.subtract(mol.atom_coords(), origin)
    nuc_dip = np.einsum('i,ix->x', charges, coords)
    return np.einsum('x,...x->...', nuc_dip, c_lambda)


def get_energy_nuc_dip(nuc_dip):
    energy = .5 * np.dot(nuc_dip, nuc_dip)
    return energy



class polariton(RKS):
    """
    QED-RKS ground state, independent of photon frequency
    """
    def get_multipole_matrix(self, c_lambda, dipole=None, quadrupole=None,
                             origin=None, frequency=None, trans_coeff=None):
        multipoles = get_multipole_matrix(self.mol, 'all', dipole, quadrupole, c_lambda=c_lambda, origin=origin)
        self.origin = origin
        self.c_lambda = np.array(c_lambda)
        self.dipole = multipoles['dipole']
        self.quadrupole = multipoles['quadrupole']

        # needed for bilinear terms # assume hartree unit!
        self.photon_freq = frequency
        self.photon_trans_coeff = trans_coeff
        self.freq_scaled_lambda = None
        if isinstance(self.photon_freq, float) or isinstance(self.photon_freq, list) or isinstance(self.photon_freq, np.ndarray): # scale photon_freq with frequency
            self.freq_scaled_lambda = get_scaled_lambda(self.c_lambda, self.photon_freq, trans_coeff)
        elif isinstance(trans_coeff, list) or isinstance(trans_coeff, np.ndarray):
            self.freq_scaled_lambda = trans_coeff

        # set it when needed
        #self.with_dse_response = True # dse response


    def nuc_grad_method(self): # used in Hessian evaluation
        from lumeq.polariton import qed_ks_grad
        return qed_ks_grad.Gradients(self)


class polariton_cs(polariton):
    """
    in photon coherent states
    """
    def get_hcore(self, mol=None):
        h = super().get_hcore(mol) # from RKS class
        hquad = .5* self.quadrupole # need 1/2 for dse
        h += hquad
        h = lib.tag_array(h, hquad=hquad)
        return h


    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        if dm is None: dm = self.make_rdm1()
        vxc = super().get_veff(mol, dm, *args, **kwargs)
        vdse_k = get_dse_2e(self.dipole, dm, with_j=False)
        edse_k = -.5* np.einsum('...pq,...qp->', vdse_k, dm) # 1/2 needed for dse, exchange has already scaled in the integral for another required 1/2

        # old tags are destroyed after the number operations
        ecoul, exc, vj, vk = vxc.ecoul, vxc.exc, vxc.vj, vxc.vk
        vxc -= vdse_k
        vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk, edse_k=edse_k, vdse_k=vdse_k)
        return vxc


    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vhf is None or getattr(vhf, 'edse_k', None) is None:
            vhf = self.get_veff(self.mol, dm)

        e_tot, e2 = super().energy_elec(dm, h1e, vhf)

        # the dse quadrupole energy is already in e_tot from e1
        # but the ks class didn't add dse jk energy, which is added in hf class though
        edse_k = vhf.edse_k.real
        e_tot += edse_k
        e2 += edse_k

        # keep track of the individual terms
        equad = np.einsum('pq,...qp->', h1e.hquad, dm)
        self.scf_summary['equad'] = equad.real
        self.scf_summary['edse_k'] = edse_k
        logger.debug(self, 'Quadrupole Energy = %s  DSE-K Energy = %s', equad, edse_k)
        return e_tot, e2


    def get_coupling_energy(self, dm=None, unit='ev'):
        if isinstance(dm, np.ndarray):
            self.energy_tot(dm=dm)

        e = self.scf_summary
        e = [e['equad'], e['edse_k']]
        e.append(e[0] + e[1])
        return convert_units(np.array(e), 'hartree', unit)


    def gen_response(self, *args, **kwargs): # for CPHF or excited-states
        vind0 = super().gen_response(*args, **kwargs)
        singlet = kwargs.get('singlet', None) # only used for RHF, default is None

        with_dse_response = self.with_dse_response if hasattr(self, 'with_dse_response') else True

        def vind(dm1): # 2e terms
            v1 = vind0(dm1)
            if with_dse_response:
                if singlet is None: # orbital hessian or CPHF type response function
                    vdse_k = get_dse_2e(self.dipole, dm1, with_j=False)
                    v1 -= vdse_k
            return v1
        return vind



class polariton_ns(polariton):
    """
    in photon number states, not recommended!
    """
    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        h = super().get_hcore(mol) # from RKS class
        hquad = .5* self.quadrupole # need 1/2 for dse
        self.nuc_dip = get_nuclear_dipoles(mol, self.c_lambda)
        hdipe = get_dse_elec_nuc(self.dipole, self.nuc_dip)

        hdipole = None
        if isinstance(self.freq_scaled_lambda, np.ndarray): # bilinear term
            hdipole = -get_multipole_matrix(mol, 'dipole', c_lambda=self.freq_scaled_lambda, origin=self.origin)['dipole'] # electrons are negative
            h += hdipole

        h += (hquad + hdipe)
        h = lib.tag_array(h, hquad=hquad, hdipe=hdipe, hdipole=hdipole)
        return h


    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        if dm is None: dm = self.make_rdm1()
        vxc = super().get_veff(mol, dm, *args, **kwargs)
        vdse_j, vdse_k = get_dse_2e(self.dipole, dm, with_j=True)
        edse_j = .5* np.einsum('...pq,...qp->', vdse_j, dm) # need 1/2 for dse here
        edse_k = -.5* np.einsum('...pq,...qp->', vdse_k, dm) # need 1/2 for dse here

        ecoul, exc, vj, vk = vxc.ecoul, vxc.exc, vxc.vj, vxc.vk
        vxc += (vdse_j - vdse_k)
        vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk, edse_j=edse_j, edse_k=edse_k, vdse_j=vdse_j, vdse_k=vdse_k)
        return vxc


    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vhf is None or getattr(vhf, 'edse_k', None) is None:
            vhf = self.get_veff(self.mol, dm)

        e_tot, e2 = super().energy_elec(dm, h1e, vhf)

        # the dse e1 energy is already in e_tot
        # but the ks class didn't add dse jk energy, which is added in hf class though
        edse_j, edse_k = vhf.edse_j.real, vhf.edse_k.real
        e_tot += (edse_j + edse_k)
        e2 += (edse_j + edse_k)

        equad = np.einsum('pq,...qp->', h1e.hquad, dm)
        edipe = np.einsum('pq,...qp->', h1e.hdipe, dm)
        elineare = 0.
        if isinstance(self.freq_scaled_lambda, np.ndarray): # bilinear term
            elineare = np.einsum('pq,...qp->', h1e.hdipole, dm)

        self.scf_summary['equad'] = equad.real
        self.scf_summary['edipe'] = edipe.real
        self.scf_summary['elineare'] = elineare.real
        self.scf_summary['edse_j'] = edse_j
        self.scf_summary['edse_k'] = edse_k
        logger.debug(self, 'Bilinear Electronic Energy = %s Quadrupole Energy = %s  Nuclear-Electronic Dipole Energy = %s  DSE-J Energy = %s  DSE-K Energy = %s', elineare, equad, edipe, edse_j, edse_k)
        return e_tot, e2


    def energy_nuc(self):
        enuc = super().energy_nuc()
        edipn = get_energy_nuc_dip(self.nuc_dip)

        elinearn = 0.
        if isinstance(self.freq_scaled_lambda, np.ndarray): # bilinear term
            elinearn = np.sum(get_nuclear_dipoles(self.mol, self.freq_scaled_lambda))

        self.scf_summary['edipn'] = edipn
        self.scf_summary['elinearn'] = elinearn
        logger.debug(self, 'Linear Nuclar Energy = %s Nuclear Dipole Energy = %s', elinearn, edipn)
        return (enuc+edipn+elinearn)


    def get_coupling_energy(self, dm=None, unit='ev'):
        if isinstance(dm, np.ndarray):
            self.get_hcore() # get nuc_dip for the energy
            self.energy_tot(dm=dm)

        e = self.scf_summary
        e = [e['elineare'], e['equad'], e['edipe'], e['edse_j'], e['edse_k'], e['edipn'], e['elinearn']]
        e.append(np.sum(e))
        return convert_units(np.array(e), 'hartree', unit)



def print_qed_dse_energy(coupling, e0, e1, e_tot, unit='ev'):
    # e0: dse energy with gas-phase density
    # e1: dse energy with qed-ks density
    # e_tot: gas-phase and qed-ks total energy
    print('coupling:', end=' ')
    if isinstance(coupling, float):
        print('%7.5f' % coupling, end=' ')
    else:
        for i in range(len(coupling)):
            print('%7.5f' % coupling[i], end=' ')
    for i, e in enumerate([e0, e1]):
        print(' dse'+str(i)+':', end='')
        for i in range(len(e)):
            print('%11.5f' % e[i], end=' ')
    print(' polariton: %11.5f %s' % (e_tot[1]-e_tot[0], unit))



if __name__ == '__main__':
    #infile = 'h2o.in'
    #parameters = parser(infile)

    #charge, spin, atom = parameters.get(section_names[0])[1:4]
    #functional, basis = get_rem_info(parameters.get(section_names[1]))[:2]
    #mol = build_molecule(atom, basis, charge, spin, verbose=0)

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
    functional = 'pbe0'
    mol = build_molecule(locals()[atom], '6-311++g**')
    mf = RKS(mol)

    mf.xc = functional
    mf.grids.prune = True
    e_tot0 = mf.kernel()
    nocc = mol.nelectron // 2

    dm = mf.make_rdm1()
    multipoles = get_multipole_matrix(mol, itype='dipole_quadrupole')
    dipole, quadrupole = multipoles['dipole'], multipoles['quadrupole']

    frequency = 0.42978 # gs doesn't depend on frequency

    #coherent_state = False
    coherent_state = True

    scf_method = polariton_cs if coherent_state else polariton_ns

    dse = []
    for c in np.linspace(0, 10, 21): # better to use integer here
        for x in range(2, 3):
            coupling = np.zeros(3)
            coupling[x] = c*1e-2

            mf1 = scf_method(mol) # in number (Fock) state

            #mf1.verbose = 10
            mf1.xc = functional
            mf1.grids.prune = True
            mf1.get_multipole_matrix(coupling)

            e0 = mf1.get_coupling_energy(dm=dm)
            e_tot = mf1.kernel()#(dm0=dm)
            e1 = mf1.get_coupling_energy()

            e_tot = np.array([e_tot0, e_tot])
            e_tot = convert_units(e_tot, 'hartree', 'ev')
            print_qed_dse_energy(coupling[x], e0, e1, e_tot)
