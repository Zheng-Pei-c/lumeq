import sys
import numpy as np

from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.dft.rks import RKS
from pyscf.grad import rks as rks_grad

from lumeq.utils.pyscf_helper import *
from lumeq.utils import convert_units, print_matrix, fdiff
from lumeq.polariton import polariton_cs, polariton_ns
from lumeq.polariton.qed_ks import get_lambda2
from lumeq.utils.fdiff import change_matrix_phase_c

def finite_difference(mf, norder=2, step_size=1e-4, ideriv=2, extra=False):
    scf_method = mf.__class__ # .__name__ to get the class name

    functional, prune, grid_level = mf.xc, mf.grids.prune, mf.grids.level
    coupling = mf.c_lambda if hasattr(mf, 'c_lambda') else None
    photon_freq = mf.photon_freq if hasattr(mf, 'photon_freq') else None
    trans_coeff = mf.photon_trans_coeff if hasattr(mf, 'photon_trans_coeff') else None
    origin = mf.origin if hasattr(mf, 'origin') else None

    mo = mf.mo_coeff
    mol = mf.mol
    natoms = mol.natm
    coords = mol.atom_coords()

    fd_e, fd_mo, fd_dip, fd_g = [], [], [], []
    for n in range(natoms):
        for x in range(3):
            fd = fdiff(norder, step_size)
            coords_new = fd.get_x(coords, [n, x])

            g1, mo1, gs_dipole1, transp = [], [], [], []
            for k in range(coords_new.shape[0]):
                mol_new = mol.set_geom_(coords_new[k], inplace=False, unit='bohr')
                mf1 = scf_method(mol_new) # in coherent state
                mf1.xc = functional
                mf1.grids.prune = prune
                mf1.grids.level = grid_level
                if isinstance(coupling, np.ndarray) or isinstance(coupling, list):
                    mf1.get_multipole_matrix(coupling, origin=origin,
                                             frequency=photon_freq,
                                             trans_coeff=trans_coeff)

                e_tot = mf1.kernel()

                if ideriv == 1:
                    g1.append(e_tot)
                elif ideriv == 2:
                    g = mf1.Gradients()
                    if functional != 'hf': g.grid_response = True
                    de1 = g.kernel()
                    g1.append(de1)

                if extra:
                    mo1.append(change_matrix_phase_c(mo, mf1.mo_coeff))
                    #print_matrix('mo '+str(n)+' '+str(x)+' '+str(k)+':', mo1[-1], 5, 1)
                    gs_dipole1.append(mf1.dip_moment(mol_new, unit='au', verbose=0))
                    if isinstance(extra, float) or isinstance(extra, list) or isinstance(extra, np.ndarray): # extra is the frequency
                        transp.append(np.einsum('...pq,qp,...->...', mf1.dipole, mf1.make_rdm1(), extra))

            fd_e.append( fd.compute(np.array(g1)))

            if extra:
                fd_mo.append( fd.compute(np.array(mo1)))
                fd_dip.append( fd.compute(np.array(gs_dipole1)))
                if isinstance(extra, float) or isinstance(extra, list) or isinstance(extra, np.ndarray): # extra is the frequency
                    fd_g.append( fd.compute(np.array(transp)) )

    return np.reshape(fd_e, (3*natoms, -1)), np.array(fd_mo), np.array(fd_dip), np.array(fd_g)


def cal_multipole_matrix_fd(mol, dm=None, origin=None, norder=2, step_size=1e-4, ideriv=1):
    # no coupling strength
    if origin is None:
        origin = np.zeros(3)

    coords = mol.atom_coords() # in bohr
    natoms = mol.natm
    nao = mol.nao_nr()
    #print_matrix('coords:\n', coords)

    combine = True if isinstance(dm, np.ndarray) else False

    dipole_d1, quadrupole_d1 = [], []
    for n in range(natoms):
        for x in range(3):
            fd = fdiff(norder, step_size)
            coords_new = fd.get_x(coords, [n, x])

            dipole, quadrupole = [], []
            for c in range(coords_new.shape[0]):
                mol_new = mol.set_geom_(coords_new[c], inplace=False, unit='bohr')
                #print_matrix('mol_new:', mol_new.atom_coords())
                with mol_new.with_common_orig(origin):
                    if ideriv == 1:
                        ints1 = mol_new.intor('int1e_r', comp=3, hermi=0)
                        ints2 = mol_new.intor('int1e_rr', comp=9, hermi=0)
                        if combine:
                            ints1 = np.einsum('xpq,...pq->...x', ints1, dm)
                            ints2 = np.einsum('xpq,...pq->...x', ints2, dm)
                            if dm.ndim == 2:
                                ints2 = ints2.reshape(3,3)
                            elif dm.ndim == 3:
                                ints2 = ints2.reshape(-1,3,3)
                        else:
                            ints2 = ints2.reshape(3,3,nao,nao)

                    elif ideriv == 2:
                        ints1, ints2 = cal_multipole_matrix_d1(mol_new, dm, origin)

                    dipole.append(ints1)
                    quadrupole.append(ints2)


            dipole_d1.append( fd.compute(np.array(dipole)) )
            quadrupole_d1.append( fd.compute(np.array(quadrupole)) )

    dipole_d1 = np.array(dipole_d1)
    quadrupole_d1 = np.array(quadrupole_d1)

    ## get pyscf style compact matrices
    #if not combine and ideriv == 2:
    #    dip_d1, qua_d1 = np.zeros((3,3,3,nao,nao)), np.zeros((3,3,3,3,nao,nao))
    #    dipole_d1 = dipole_d1.reshape(natoms, 3, natoms, 3, 3, nao, nao)
    #    quadrupole_d1 = quadrupole_d1.reshape(natoms, 3, natoms, 3, 3, 3, nao, nao)

    #    atmlst = range(mol.natm)
    #    aoslices = mol.aoslice_by_atom()
    #    for k, ia in enumerate(atmlst):
    #        p0, p1 = aoslices[ia,2:]

    #        tmp1 = dipole[:,:,p0:p1]
    #        tmp2 = quadrupole[:,:,p0:p1]

    return dipole_d1, quadrupole_d1


def cal_multipole_matrix_d1(mol, dm=None, origin=None): # no coupling strength
    if origin is None:
        origin = np.zeros(3)

    natoms = mol.natm
    nao = mol.nao_nr()
    with mol.with_common_orig(origin):
        # move derivative as the first index and the ket as bra at the same time
        dipole = mol.intor('int1e_irp', comp=9, hermi=0).reshape(3,3,nao,nao).transpose(1,0,3,2)
        quadrupole = mol.intor('int1e_irrp', comp=27, hermi=0).reshape(9,3,nao,nao).transpose(1,0,3,2)

    #print_matrix('dipole 0:', dipole, 5, 1)

    combine = True if isinstance(dm, np.ndarray) else False
    if combine:
        dipole_d1 = np.zeros((natoms, 3, 3))
        quadrupole_d1 = np.zeros((natoms, 3, 9))
    else:
        dipole_d1 = np.zeros((natoms, 3, 3, nao, nao))
        quadrupole_d1 = np.zeros((natoms, 3, 9, nao, nao))

    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]

        tmp1 = dipole[:,:,p0:p1]
        tmp2 = quadrupole[:,:,p0:p1]

        if combine:
            dipole_d1[k] = - 2.* np.einsum('ixpq,pq->ix', tmp1, dm[p0:p1])
            quadrupole_d1[k] = - 2.* np.einsum('ixpq,pq->ix', tmp2, dm[p0:p1])
        else:
            dipole_d1[k,:,:,p0:p1] -= tmp1
            dipole_d1[k,:,:,:,p0:p1] -= tmp1.transpose(0,1,3,2)

            quadrupole_d1[k,:,:,p0:p1] -= tmp2
            quadrupole_d1[k,:,:,:,p0:p1] -= tmp2.transpose(0,1,3,2)

    if combine:
        return dipole_d1.reshape(natoms*3,3), quadrupole_d1.reshape(natoms*3,3,3)
    else:
        return dipole_d1.reshape(natoms*3,3,nao,nao), quadrupole_d1.reshape(natoms*3,3,3,nao,nao)


def get_multipole_matrix_d1(mol, c_lambda, origin=None, itype='all'):
    if origin is None:
        origin = np.zeros(3)

    dipole, quadrupole = None, None
    if itype == 'all':
        itype += 'dipole_quadrupole'

    nao = mol.nao_nr()
    with mol.with_common_orig(origin):
        # these derivatives have extra minus
        if 'dipole' in itype:
            dipole = mol.intor('int1e_irp', comp=9, hermi=0).reshape(3,3,nao,nao)
        if 'quadrupole' in itype:
            quadrupole = mol.intor('int1e_irrp', comp=27, hermi=0).reshape(3,3,3,nao,nao)

    if isinstance(c_lambda, np.ndarray) or isinstance(c_lambda, list):
        # these derivatives don't distinguish nuclei
        if isinstance(dipole, np.ndarray):
            dipole = - np.einsum('mxpq,...m->xqp...', dipole, c_lambda) # move mode index to the last for convenience latter
        if isinstance(quadrupole, np.ndarray):
            quadrupole = - np.einsum('mnxpq,mn->xqp', quadrupole, get_lambda2(c_lambda))

    return dipole, quadrupole


def get_dse_2e(dipole, dipole_d1, dm, with_j=False, scale_k=.5): # c_lambda is included
    # scale k by 1/2 for restricted orbital case by default
    # we moved the mode index for lambda-dipole derivative to the last
    if dipole.ndim == 2:
        if dm.ndim == 2:
            vk = np.einsum('xpq,rs,qr->xps', dipole_d1, dipole, dm)
            if with_j is False:
                return vk*scale_k
            else:
                vj = np.einsum('xpq,rs,sr->xpq', dipole_d1, dipole, dm)
                return [vj, vk*scale_k]
        else: # multiple density matrices, ie uhf
            vk = np.einsum('xpq,rs,iqr->ixps', dipole_d1, dipole, dm)
            if with_j is False:
                return vk*scale_k
            else:
                vj = np.einsum('xpq,rs,isr->ixpq', dipole_d1, dipole, dm)
                return [vj, vk*scale_k]

    elif dipole.ndim == 3:
        if dm.ndim == 2:
            vk = np.einsum('xpql,lrs,qr->xps', dipole_d1, dipole, dm)
            if with_j is False:
                return vk*scale_k
            else:
                vj = np.einsum('xpql,lrs,sr->xpq', dipole_d1, dipole, dm)
                return [vj, vk*scale_k]
        else: # multiple density matrices, ie uhf
            vk = np.einsum('xpql,lrs,iqr->ixps', dipole_d1, dipole, dm)
            if with_j is False:
                return vk*scale_k
            else:
                vj = np.einsum('xpql,lrs,isr->ixpq', dipole_d1, dipole, dm)
                return [vj, vk*scale_k]


def get_dse_elec_nuc_d1(dipole_d1, nuc_dip): # c_lambda is included
    if isinstance(nuc_dip, float):
        return -nuc_dip * dipole_d1
    else: # numpy does not sum over ellipsis
        return -np.einsum('xpql,l->xpq', dipole_d1, nuc_dip) # l is the number of photon modes


def get_dse_elec_nuc_grad(dipole, nuc_dip_d1, dm): # c_lambda is included
    dipole = np.einsum('...pq,qp->...', dipole, dm)
    if isinstance(dipole, float):
        return -nuc_dip_d1 * dipole
    else: # numpy does not sum over ellipsis
        return -np.einsum('l,lnx->nx', dipole, nuc_dip_d1) # l is the number of photon modes


def get_nuclear_dipoles_d1(charges, c_lambda):
    g1 = []
    for i in range(len(charges)):
        g1.append(np.eye(3)*charges[i])

    return np.einsum('nxy,...x->...ny', g1, c_lambda)


def get_grad_nuc_dip(nuc_dip, nuc_dip_d1):
    if isinstance(nuc_dip, float):
        return nuc_dip * nuc_dip_d1
    else: # numpy does not sum over ellipsis
        grad = np.einsum('i,iny->ny', nuc_dip, nuc_dip_d1)
    return grad



class Gradients(rks_grad.Gradients):
    def get_veff(self, mol=None, dm=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        vxc = super().get_veff(mol, dm)
        exc = vxc.exc1_grid

        vxc += self.dse_fock_d1(mol, dm)

        return lib.tag_array(vxc, exc1_grid=exc)


    # add this to the gradient class so that can be called for the cphf
    def dse_fock_d1(self, mol, dm, dipole_d1=None, quadrupole_d1=None):
        mf = self.base
        # here the dipoles and quadrupoles have aleady combined with coupling
        if not isinstance(dipole_d1, np.ndarray):
            dipole_d1, quadrupole_d1 = get_multipole_matrix_d1(mol, mf.c_lambda, mf.origin)
        vdse_k = get_dse_2e(mf.dipole, dipole_d1, dm, with_j=False)
        return (.5*quadrupole_d1 - vdse_k)


polariton_cs.Gradients = lib.class_as_method(Gradients)



class Gradients2(Gradients):
    def dse_fock_d1(self, mol, dm, dipole_d1=None, quadrupole_d1=None):
        mf = self.base
        # here the dipoles and quadrupoles have aleady combined with coupling
        if not isinstance(dipole_d1, np.ndarray):
            dipole_d1, quadrupole_d1 = get_multipole_matrix_d1(mol, mf.c_lambda, mf.origin)

        hdip = get_dse_elec_nuc_d1(dipole_d1, mf.nuc_dip)
        hdip += .5 * quadrupole_d1

        if isinstance(mf.freq_scaled_lambda, np.ndarray): # bilinear term
            # dipole derivative with scaled coupling
            hdip -= get_multipole_matrix_d1(mol, mf.freq_scaled_lambda, mf.origin, 'dipole')[0]

        vdse_j, vdse_k = get_dse_2e(mf.dipole, dipole_d1, dm, with_j=True)
        return (hdip + vdse_j - vdse_k)


    def grad_nuc(self, mol=None, atmlst=None):
        if mol is None: mol = self.mol
        g1 = super().grad_nuc(mol, atmlst)

        mf = self.base
        nuc_dip_d1 = get_nuclear_dipoles_d1(mol.atom_charges(), mf.c_lambda)
        g1 += get_dse_elec_nuc_grad(mf.dipole, nuc_dip_d1, mf.make_rdm1())
        g1 += get_grad_nuc_dip(mf.nuc_dip, nuc_dip_d1)

        if isinstance(mf.freq_scaled_lambda, np.ndarray): # bilinear term
            g1 += get_nuclear_dipoles_d1(mol.atom_charges(), mf.freq_scaled_lambda)

        return g1


polariton_ns.Gradients = lib.class_as_method(Gradients2)



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
    atom = locals()[sys.argv[1]] if len(sys.argv) > 1 else hf

    functional = 'hf'
    mol = build_molecule(atom, '3-21g')

    frequency = 0.42978 # gs doesn't depend on frequency
    trans_coeff = np.ones(3)
    coupling = np.array([0, 0, .5])

    #coherent_state = False
    coherent_state = True

    scf_method = polariton_cs if coherent_state else polariton_ns

    mf = scf_method(mol)
    mf.xc = functional
    mf.grids.prune = True
    mf.get_multipole_matrix(coupling, frequency=frequency, trans_coeff=trans_coeff)

    e_tot = mf.kernel()
    print('scf energy:', e_tot)
    dm = mf.make_rdm1()
    #print_matrix('dm:', dm, 5, 1)

    g = mf.Gradients()
    g.grid_response = True
    de1 = g.kernel()
    print_matrix('de1:', de1)


    norder, step_size = 2, 1e-4

    fde = finite_difference(mf, norder, step_size, ideriv=1)[0]
    fde = fde.reshape(-1, 3)
    print_matrix('fde:', fde)
    print('gradient diff:', np.linalg.norm(fde-de1))


    # check the multipole matrix derivatives
    #dipole_fd, quadrupole_fd = cal_multipole_matrix_fd(mol, dm=dm, norder=norder, step_size=step_size)
    #dipole_d1, quadrupole_d1 = cal_multipole_matrix_d1(mol, dm=dm)

    #print_matrix('dipole_fd:', dipole_fd, 5, 1)
    #print_matrix('dipole_d1:', dipole_d1, 5, 1)

    #print_matrix('quadrupole_fd:', quadrupole_fd, 5, 1)
    #print_matrix('quadrupole_d1:', quadrupole_d1, 5, 1)

    #print('dipole_d1 diff:', np.linalg.norm(dipole_fd-dipole_d1))
    #print('quadrupole_d1 diff:', np.linalg.norm(quadrupole_fd-quadrupole_d1))
