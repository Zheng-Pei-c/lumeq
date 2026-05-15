from functools import reduce
import sys
import numpy
from pyscf import lib, scf, tdscf, gto
from pyscf.lib import logger
from pyscf.dft import rks, gen_grid
from pyscf.dft import numint
from pyscf.scf import cphf
from pyscf.grad import rks as rks_grad
from pyscf.grad import tdrhf
from pyscf.grad.rks import grids_response_cc # tddft full response
from pyscf.grad.tdrks import _contract_xc_kernel

from lumeq.utils import print_matrix


#
# Given Y = 0, TDDFT gradients (XAX+XBY+YBX+YAY)^1 turn to TDA gradients (XAX)^1
#
def grad_elec(td_grad, x_y, singlet=True, atmlst=None,
              max_memory=2000, verbose=logger.INFO):
    '''
    Electronic part of TDA, TDDFT nuclear gradients

    Args:
        td_grad : grad.tdrhf.Gradients or grad.tdrks.Gradients object.

        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = (mo_occ>0).sum()
    nvir = nmo - nocc
    x, y = x_y
    x = numpy.asarray(x)
    if numpy.isscalar(y):
        y = numpy.zeros_like(x)
    else:
        y = numpy.asarray(y)
    xpy = (x+y).reshape(nocc,nvir).T
    xmy = (x-y).reshape(nocc,nvir).T
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]

    x_vo = x.reshape(nocc,nvir).T
    y_vo = y.reshape(nocc,nvir).T
    pvv = numpy.einsum('ai,bi->ab', x_vo, x_vo) + numpy.einsum('ai,bi->ba', y_vo, y_vo)
    poo = numpy.einsum('ai,aj->ji', x_vo, x_vo) + numpy.einsum('ai,aj->ij', y_vo, y_vo)
    # add transpose to difference density but not transition density
    Ptrans = reduce(numpy.dot, (orbv, x_vo, orbo.T))
    Ptrans += reduce(numpy.dot, (orbo, y_vo.T, orbv.T))
    Pdiff = reduce(numpy.dot, (orbv, pvv, orbv.T))
    Pdiff -= reduce(numpy.dot, (orbo, poo, orbo.T))
    Pdiff += Pdiff.T


    mem_now = lib.current_memory()[0]
    max_memory = max(2000, td_grad.max_memory*.9-mem_now)

    # firstly form the rhs of z-vector equation
    has_xc = True if hasattr(mf, 'xc') else False
    grid_response = getattr(td_grad, 'grid_response', None)
    if grid_response is None:
        grid_response = getattr(mf, 'grid_response', False)
    if has_xc:
        ni = mf._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    else: # hf
        omega, alpha, hyb = 0, 0, 1.0

    if abs(hyb) > 1e-10:
        dm = (Pdiff, Ptrans)
        vj, vk = mf.get_jk(mol, dm, hermi=0)
        vk *= hyb
        if abs(omega) > 1e-10:
            vk += mf.get_k(mol, dm, hermi=0, omega=omega) * (alpha-hyb)
        veff0doo = vj[0] * 2 - vk[0]
        if singlet:
            veff = vj[1] + vj[1].T - vk[1]
        else:
            veff = -vk[1]
    else:
        dm = (Pdiff, Ptrans)
        vj = mf.get_j(mol, dm, hermi=1)
        veff0doo = vj[0] * 2
        if singlet:
            veff = vj[1] + vj[1].T
        else:
            veff = None

    if has_xc:
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
        f1vo, f1oo, vxc1, k1ao = \
                _contract_xc_kernel(td_grad, mf.xc, Ptrans,
                                    Pdiff, True, True, singlet, max_memory)

        veff0doo += f1oo[0] + k1ao[0] * 2
        veff += f1vo[0]

    # rhs of z-vector equation
    wvo = reduce(numpy.dot, (orbv.T, veff0doo, orbo))*.5
    veff_mo = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
    wvo += numpy.einsum('ac,ai->ci', veff_mo[nocc:,nocc:], x_vo)
    wvo += numpy.einsum('ca,ai->ci', veff_mo[nocc:,nocc:], y_vo)
    wvo -= numpy.einsum('ij,cj->ci', veff_mo[:nocc,:nocc], x_vo)
    wvo -= numpy.einsum('ji,cj->ci', veff_mo[:nocc,:nocc], y_vo)
    wvo *= 4
    #print_matrix('wvo:\n', wvo)

    # set singlet=None, generate function for CPHF type response kernel
    vresp = mf.gen_response(singlet=None, hermi=1)
    def fvind(x):
        dm = reduce(numpy.dot, (orbv, x.reshape(nvir,nocc)*2, orbo.T))
        v1ao = vresp(dm+dm.T)
        return reduce(numpy.dot, (orbv.T, v1ao, orbo)).ravel()
    z1 = cphf.solve(fvind, mo_energy, mo_occ, wvo,
                    max_cycle=td_grad.cphf_max_cycle,
                    tol=td_grad.cphf_conv_tol)[0]
    z1 = z1.reshape(nvir,nocc)
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    z1ao  = reduce(numpy.dot, (orbv, z1, orbo.T))
    veff_z = vresp(z1ao+z1ao.T)

    # ground and excited-state densities
    # used in the following integrals and in the end
    dm0 = reduce(numpy.dot, (orbo, orbo.T))
    dmz1doo = z1ao + Pdiff
    dmz1doo += dmz1doo.T
    dm1 = dm0 + dmz1doo/4

    # now get the energy-weighted-density im0
    fock = mf.get_fock()
    im0  = numpy.dot(fock, dm1)
    im0 += numpy.dot(veff0doo+veff_z, dm0)*0.5
    im0 += numpy.dot(veff, Ptrans.T)
    im0 += numpy.dot(veff.T, Ptrans)
    im0 *= 2

    cct = reduce(numpy.dot, (mo_coeff, mo_coeff.T))
    im0 = reduce(numpy.dot, (cct, im0))
    im0 += im0.T
    #print_matrix('im0:\n', im0)


    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    if abs(hyb) > 1e-10:
        dm = (dm0, dmz1doo, Ptrans, Ptrans.T)
        vj, vk = td_grad.get_jk(mol, dm)
        vk *= hyb
        if abs(omega) > 1e-10:
            with mol.with_range_coulomb(omega):
                vk += td_grad.get_k(mol, dm) * (alpha-hyb)
        vj = vj.reshape(-1,3,nao,nao)
        vk = vk.reshape(-1,3,nao,nao)
        if singlet:
            veff1 = vj * 2 - vk
        else:
            veff1 = numpy.vstack((vj[:2]*2-vk[:2], -vk[2:]))
    else:
        dm = (dm0, dmz1doo, Ptrans)
        vj = td_grad.get_j(mol, dm)
        vj = vj.reshape(-1,3,nao,nao)
        veff1 = numpy.zeros((4,3,nao,nao))
        if singlet:
            veff1[:3] = vj * 2
            veff1[3] = vj[2] * 2
        else:
            veff1[:2] = vj[:2] * 2

    if has_xc:
        fxcz1 = _contract_xc_kernel(td_grad, mf.xc, z1ao, None,
                                    False, False, True, max_memory)[0]

        veff1[0] += vxc1[1:]
        veff1[1] +=(f1oo[1:] + fxcz1[1:] + k1ao[1:]*2)*2 # *2 for dmz1doo+dmz1oo.T
        veff1[2] += f1vo[1:]
        veff1[3] += f1vo[1:]

        veff1_xc = numpy.zeros((4,3,nao,nao))
        veff1_xc[0] = vxc1[1:]
        veff1_xc[1] =(f1oo[1:] + fxcz1[1:] + k1ao[1:]*2)*2 # *2 for dmz1doo+dmz1oo.T
        veff1_xc[2] = f1vo[1:]
        veff1_xc[3] = f1vo[1:]


    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        # Ground state gradients
        h1ao = hcore_deriv(ia)
        h1ao[:,p0:p1]   += veff1[0,:,p0:p1]
        h1ao[:,:,p0:p1] += veff1[0,:,p0:p1].transpose(0,2,1)
        e1  = numpy.einsum('xpq,pq->x', h1ao, dm1)*2
        e1 -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])

        e1 += numpy.einsum('xij,ij->x', veff1[1,:,p0:p1], dm0[p0:p1])
        e1 += numpy.einsum('xij,ij->x', veff1[2,:,p0:p1], Ptrans[p0:p1,:]) * 4
        e1 += numpy.einsum('xij,ji->x', veff1[3,:,p0:p1], Ptrans[:,p0:p1]) * 4

        de[k] = e1


        if has_xc:
            h1ao = numpy.zeros(h1ao.shape)
            h1ao[:,p0:p1]   += veff1_xc[0,:,p0:p1]
            h1ao[:,:,p0:p1] += veff1_xc[0,:,p0:p1].transpose(0,2,1)
            e1 = 0
            e1 += numpy.einsum('xpq,pq->x', h1ao, dm1)*2
            e1 += numpy.einsum('xij,ij->x', veff1_xc[1,:,p0:p1], dm0[p0:p1])
            e1 += numpy.einsum('xij,ij->x', veff1_xc[2,:,p0:p1], Ptrans[p0:p1,:]) * 4
            e1 += numpy.einsum('xij,ji->x', veff1_xc[3,:,p0:p1], Ptrans[:,p0:p1]) * 4

    if has_xc and grid_response:
        excsum, coord_sum = _contract_xc_kernel_grad(td_grad, mf.xc, Ptrans, dmz1doo/4,
                                    False, False, True, max_memory)
        de += excsum

    log.timer('TDDFT nuclear gradients', *time0)
    return de


# dmvo, dmoo in AO-representation
# Note spin-trace is applied for fxc, kxc
def _contract_xc_kernel_grad(td_grad, xc_code, dmvo, dmoo, with_vxc=True,
                        with_kxc=True, singlet=True, max_memory=2000):
    mol = td_grad.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    ao_loc = mol.ao_loc_nr()

    nocc = (mo_occ>0).sum()
    dm0 = reduce(numpy.dot, (mo_coeff[:,:nocc], mo_coeff[:,:nocc].T))

    # dmvo ~ reduce(numpy.dot, (orbv, Xai, orbo.T))
    dmvo = (dmvo + dmvo.T) * .5 # because K_{ia,jb} == K_{ia,jb}
    dmoo = (dmoo + dmoo.T) * .5

    deriv = 3
    excsum = numpy.zeros((mol.natm,3))
    coord_sum = numpy.zeros_like(excsum)

    if not singlet:
        raise NotImplementedError(f'{xctype} triplet')
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        raise NotImplementedError(f'{xctype}')

    def make_vtmp(ao, wv, weight, mask):
        vtmp = numpy.zeros((3, nao, nao))
        if xctype == 'LDA':
            aow = numint._scale_ao(ao[0], weight * wv[0])
            rks_grad._d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
        elif xctype == 'GGA':
            wv = wv * weight
            wv[0] *= .5
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wv, mask, ao_loc)
        else:
            wv = wv * weight
            wv[0] *= .5
            wv[4] *= .5
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wv[:4], mask, ao_loc)
            rks_grad._tau_grad_dot_(vtmp, mol, ao, wv[4], mask, ao_loc, True)
        return vtmp

    ao_deriv = 1 if xctype == 'LDA' else 2
    for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                        cutoff=grids.cutoff)
        if xctype == 'LDA':
            ao0 = ao[0]
        elif xctype == 'GGA':
            ao0 = ao[:4]
        else:
            ao0 = ao[:10]
        rho = ni.eval_rho2(mol, ao0, mo_coeff, mo_occ, mask, xctype,
                           with_lapl=False)
        exc, vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv,
                                            xctype=xctype)

        rho1 = ni.eval_rho(mol, ao0, dmvo, mask, xctype, hermi=1,
                           with_lapl=False) * 2
        rho2 = ni.eval_rho(mol, ao0, dmoo, mask, xctype, hermi=1,
                           with_lapl=False) * 2
        if xctype == 'LDA':
            rho0 = rho
            rho1 = rho1[numpy.newaxis]
            rho2 = rho2[numpy.newaxis]
        else:
            rho0 = rho[0]

        # Weight response of f_xc + v_xc*rho2 + rho1*fxc*rho1.
        wv = exc * rho0
        wv += numpy.einsum('xg,xg->g', vxc, rho2)
        wv += numpy.einsum('xg,yg,xyg->g', rho1, rho1, fxc)
        excsum += numpy.einsum('g,nxg->nx', wv, weight1)

        # Coordinate response of the same scalar with respect to
        # rho0, rho2, and rho1.
        wv0 = vxc.copy()
        wv0 += numpy.einsum('yg,xyg->xg', rho2, fxc)
        wv0 += numpy.einsum('yg,zg,xyzg->xg', rho1, rho1, kxc)

        wv1 = numpy.einsum('yg,xyg->xg', rho1, fxc) * 2

        for wv_i, dm_i in ((wv0, dm0), (vxc, dmoo), (wv1, dmvo)):
            vtmp = make_vtmp(ao, wv_i, weight, mask)
            e1 = numpy.einsum('xij,ji->x', vtmp, dm_i) * 4
            excsum[atm_id] += e1
            coord_sum[atm_id] += e1

    return excsum, coord_sum


def tddft_grad_new(mol, mf, td, state):
    atmlst = range(mol.natm)
    td_grad = td.nuc_grad_method()
    mf_grad = td_grad.base._scf.nuc_grad_method()
    #mf_grad.grid_response = True

    xy = td.xy[state-1]
    de = grad_elec(td_grad, xy)

    de += mf_grad.grad_nuc(atmlst=atmlst)
    return de



if __name__ == "__main__":
    water = '''
    H    0.5397736    0.7493682    0.0000000
    O   -0.0185014    0.0000000    0.0000000
    H    0.5397736   -0.7493682    0.0000000
    '''
    atom = water

    functional = 'pbe0'
    basis = '3-21g'
    td_method = 'tddft'
    state = 1
    nstates = 5
    grid_response = True

    mol = gto.M(
        atom=atom,
        basis=basis,
        spin=0,
        charge=0,
        verbose=0,
    )

    mf = scf.RKS(mol)
    mf.xc = functional
    mf.grids.prune = True
    mf.grid_response = grid_response
    mf.kernel()

    if td_method == 'tda':
        td_model = tdscf.TDA
    else:
        td_model = tdscf.TDDFT
    td = td_model(mf)
    td.kernel(nstates=nstates)
    print('energy:\n', td.e*27.2107)

    g1 = tddft_grad_new(mol, mf, td, state)
    print_matrix(td_method+' gradient:', g1)
