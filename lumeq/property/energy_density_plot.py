from lumeq import sys, np
np.set_printoptions(precision=6)

from pyscf import scf, gto, dft, df, lib
from pyscf import tdscf
from pyscf.dft import numint
from pyscf.tools import cubegen

try:
    from memory_profiler import profile
    from ttictoc import TicToc
except ModuleNotFoundError:
    pass

class EnergyDensity():
    r"""
    ground-state energy density: j_only(), jk(), xc()
    excited-state energy density: coulomb_only(), coulomb_exchange(), functional()
    """

    def __init__(self, atom, functional, basis, ecp=None, charge=0,
            max_memory=4000, efield=None, debug=False):
        self.debug = debug
        if len(atom) == 2:
            self.frag_atom = atom
            self.frag1 = int(len(atom[0].split())//4)
            atom = atom[0] + atom[1]
            self.frag_charge = charge
            charge = charge[0] + charge[1]
        else:
            self.frag_atom = None

        self.mol = gto.M(
                verbose = 1,
                atom = atom,
                basis = basis,
                ecp = ecp,
                charge = charge,
                max_memory = max_memory
        )

        # ground-state
        self.mf = scf.RKS(self.mol)
        self.mf.xc = functional
        print('coordinates:\n', self.mol.atom)
        print('functional: ', self.mf.xc)
        print('basis: ', self.mol.basis)
        print('charge: ', self.mol.charge)

        if efield != None:
            self.mol.set_common_orig([0, 0, 0])
            self.efield = efield
            h = (self.mol.intor('cint1e_kin_sph') + self.mol.intor('cint1e_nuc_sph')
                + np.einsum('x,xij->ij', self.efield, self.mol.intor('cint1e_r_sph', comp=3)))
            self.mf.get_hcore = lambda *args: h

        if self.mf.xc != 'wb97xd':
            self.mf._numint.libxc = dft.xcfun
        self.mf.kernel()
        print('nuclear: ', self.mol.energy_nuc())
        #self.mf.pop()

        self.scf_dm = self.mf.make_rdm1()
        #print('scf_dm dimension: ', self.scf_dm.shape)

        print('nbas: %d nao_nr: %d' % (self.mol.nbas, self.mol.nao_nr()))
        #print('mol cart: ', self.mol.cart)

        self.coeffs = self.mf.mo_coeff
        #print('mo dimension: ', self.coeffs.shape)
        #print('mo coeffs:\n', self.coeffs)
        mo_nocc = self.mf.mo_occ
        #print('mo_nocc: ', mo_nocc)
        self.orbo = self.coeffs[:, mo_nocc==2]
        self.orbv = self.coeffs[:, mo_nocc==0]

        self.norb = mo_nocc.shape[0]
        self.nocc = self.orbo.shape[1]
        self.nvir = self.norb - self.nocc
        print('nocc: ', self.nocc, ' norb: ', self.norb)
        print('orbital energy:\n', self.mf.mo_energy)


    #@profile
    def cal_excited_state(self, nstates):
        # excited-state
        self.td = tdscf.TDDFT(self.mf)
        self.td.max_cycle = 600
        self.td.max_space = 200
        self.td.kernel(nstates=nstates)
        print('TDDFT converged: ', self.td.converged)
        self.td.analyze(verbose=5)


    #@profile
    def cal_excitation_lambda_factor(self, estate):
        #self.creat_mesh_grids(1, [101, 101, 101, 0.3])

        occ_mo_grids = np.einsum('mi,pm->ip', self.orbo, self.ao_value[0])
        vir_mo_grids = np.einsum('ma,pm->pa', self.orbv, self.ao_value[0])
        smoov = np.einsum('ip,pa,p->ia', np.absolute(occ_mo_grids), np.absolute(vir_mo_grids), self.weights)

        #self.cal_excited_state(nstates)
        #if estate == -1:
        #    estate = np.arange(nstates)
        #print(estate)

        es_lambda = []
        for ie in estate:
            #self.tdxy = self.td.xy[ie]
            td_kappa = self.td.xy[ie][0] + self.td.xy[ie][1]
            kappa_sq1 = np.einsum('ia,ia,ia->', td_kappa, td_kappa, smoov)
            kappa_sq2 = np.einsum('ia,ia->', td_kappa, td_kappa)
            es_lambda.append(kappa_sq1/kappa_sq2)

        es_lambda = np.array(es_lambda)
        #print('Tozer Lambda Factor:\n', es_lambda)
        np.savetxt(sys.stdout.buffer, es_lambda.reshape(1, es_lambda.shape[0]), fmt='%13.7f', header='Tozer Lambda Factor:')


    #@profile
    def cal_overlap_detach_attach_density(self, estate):
        overlap_detach_attach = []

        for ie in estate:
            detach_dm = np.zeros((self.norb, self.norb))
            attach_dm = np.zeros((self.norb, self.norb))
            for x in range(2):
                detach_dm[:self.nocc, :self.nocc] -= np.einsum('ia,ja->ij', self.td.xy[ie][x], self.td.xy[ie][x])
                attach_dm[self.nocc:, self.nocc:] += np.einsum('ia,ib->ab', self.td.xy[ie][x], self.td.xy[ie][x])
            attach_dm = 2.0 * np.einsum('ij,pi,qj->pq', attach_dm, self.coeffs, self.coeffs.conj())
            detach_dm = 2.0 * np.einsum('ij,pi,qj->pq', detach_dm, self.coeffs, self.coeffs.conj())

            attach_rho_value = self.density_on_grids(attach_dm, xctype='GGA')[0]
            detach_rho_value = self.density_on_grids(detach_dm, xctype='GGA')[0]

            attach_rho_value = np.sqrt(np.absolute(attach_rho_value))
            detach_rho_value = np.sqrt(np.absolute(detach_rho_value))
            overlap_detach_attach.append(np.einsum('p,p,p->', attach_rho_value, detach_rho_value, self.weights))

        overlap_detach_attach = np.array(overlap_detach_attach)
        #print('Overlap of detachment and attachment density:\n', overlap_detach_attach)
        np.savetxt(sys.stdout.buffer, overlap_detach_attach.reshape(1, overlap_detach_attach.shape[0]), fmt='%13.7f', header='Overlap of detachment and attachment density:')


    #@profile
    def cal_density_matrices(self, ie):
        self.tdxy = self.td.xy[ie]
        print('transition identity check: ',
                np.einsum('ia,ia', (self.tdxy[0]+self.tdxy[1]), (self.tdxy[0]-self.tdxy[1])))

        self.trans_dm = np.einsum('ia,pi,qa->pq', self.tdxy[0], self.orbo.conj(), self.orbv)
        self.trans_dm += np.einsum('ia,pi,qa->qp', self.tdxy[1], self.orbo, self.orbv.conj())
        self.trans_dm_sym = self.trans_dm + self.trans_dm.T
        #print('trans_dm dimension: ', self.trans_dm.shape)

        self.diff_dm = np.zeros((self.norb, self.norb))
        for x in range(2):
            self.diff_dm[:self.nocc, :self.nocc] -= np.einsum('ia,ja->ij', self.tdxy[x], self.tdxy[x])
            self.diff_dm[self.nocc:, self.nocc:] += np.einsum('ia,ib->ab', self.tdxy[x], self.tdxy[x])
        self.diff_dm = np.einsum('ij,pi,qj->pq', self.diff_dm, self.coeffs, self.coeffs.conj())
        self.diff_dm *= 2.0
        #print('diff_dm dimension: ', self.diff_dm.shape)


    #@profile
    def check_transtion_density(self):
        # check transition density matrix
        self.mol.set_common_origin([0,0,0])
        dipm = mol.intor('int1e_r')
        #print('dipm dimension: ', dipm.shape)
        transdip = np.einsum('xij,ji->x', dipm, self.trans_dm_sym)
        print('transdip:\n', transdip)
        print('reference transition dipole:\n', self.td.transition_dipole())


    #@profile
    def creat_mesh_grids(self, grid_type, nxyz):
        if grid_type == 1:
            # default mesh grids and weights
            self.coords = self.mf.grids.coords
            self.weights = self.mf.grids.weights
            self.ngrids = self.weights.shape[0]
        elif grid_type == 2:
            self.cc = cubegen.Cube(self.mol, nx=nxyz[0], ny=nxyz[1], nz=nxyz[2], resolution=nxyz[3])
            self.coords = self.cc.get_coords()
            self.ngrids = self.cc.get_ngrids()
            self.weights = ( np.ones(self.ngrids)
                    * (self.cc.xs[1]-self.cc.xs[0])
                    * (self.cc.ys[1]-self.cc.ys[0])
                    * (self.cc.zs[1]-self.cc.zs[0])
                    )
            print('nx: ', self.cc.nx, ' ny: ', self.cc.ny, ' nz: ', self.cc.nz)

        print('ngrids: ', self.ngrids)
        # ao integral and its derivatives
        #self.ao_value = self.mol.eval_gto('GTOval_sph_deriv1', self.coords)
        # if we need to seperate batch
        self.ao_value = np.zeros((4, self.ngrids, self.mol.nao_nr()))
        blksize = min(8000, self.ngrids)
        for ip0, ip1 in lib.prange(0, self.ngrids, blksize):
            self.ao_value[:,ip0:ip1,:] = self.mol.eval_gto('GTOval_sph_deriv1', self.coords[ip0:ip1])
        print('ao_value dimension: ', self.ao_value.shape)


    #@profile
    def density_on_grids(self, dm, xctype='GGA'):
        # rho and its derivatives
        #rho_value = numint.eval_rho(self.mol, self.ao_value, dm, xctype=xctype)
        rho_value = np.zeros((4, self.ngrids))
        blksize = min(8000, self.ngrids)
        for ip0, ip1 in lib.prange(0, self.ngrids, blksize):
            rho_value[:,ip0:ip1] = numint.eval_rho(self.mol, self.ao_value[:,ip0:ip1,:], dm, xctype=xctype)
        return rho_value


    #@profile
    def grab_gs_density_on_grids(self, xctype='GGA'):
        # ground-state rho and its derivatives
        self.g_rho_value = self.density_on_grids(self.scf_dm, xctype)
        #print('g_rho_value dimension: ', self.g_rho_value.shape)


    #@profile
    def grab_es_density_on_grids(self, xctype='GGA'):
        # difference rho and its derivatives
        self.d_rho_value = self.density_on_grids(self.diff_dm, xctype)
        #print('d_rho_value dimension: ', self.d_rho_value.shape)
        # transition rho and its derivatives
        self.t_rho_value = self.density_on_grids(self.trans_dm_sym, xctype)
        #print('t_rho_value dimension: ', self.t_rho_value.shape)


    #@profile
    def energy_density_core(self, dm, rho_value):
        # nuclear attraction energy density
        engrho = 0
        for i in range(self.mol.natm):
            r = self.mol.atom_coord(i)
            Z = self.mol.atom_charge(i)
            rp = r - self.coords
            engrho -= Z / np.einsum('xi,xi->x', rp, rp)**.5
        #print('engrho dimension: ', engrho.shape)
        #engrho = engrho * rho_value[0]

        if self.cal_another_attraction:
            engrho *= 0.5

        self.energy_density.append(engrho * rho_value[0])
        self.ed_components.append('attraction')

        if self.cal_another_attraction:
            for i in range(self.mol.natm):
                r = self.mol.atom_coord(i)
                Z = self.mol.atom_charge(i)
                rp = r - self.coords
                engrho = -Z / np.einsum('xi,xi->x', rp, rp)**.5
                self.nuclear_attraction2.append(0.5*engrho * rho_value[0])


        if len(self.mol._ecpbas) > 0:
            engrho = 0
            hecp = self.mol.intor_symmetric('ECPscalar')
            hecp = np.einsum('ij,jm,ni->mn', hecp, dm, dm)

            #engrho += .5 * self.density_on_grids(hecp)[0]
            engrho = .5 * self.density_on_grids(hecp)[0]
            self.energy_density.append(engrho)
            self.ed_components.append('ecpotential')

        #self.energy_density.append(engrho)

        # kinetic energy density
        engrho = 0
        for x in range(1, 4):
            engrho += np.einsum('pi,ij,pj->p', self.ao_value[x], dm, self.ao_value[x].conj())
        self.energy_density.append(.5 * engrho)
        self.ed_components.append('kinetic')


    #@profile
    def check_energy_core(self, dm):
        index1 = self.ed_components.index('attraction')
        ener_attrac = np.einsum('i,i', self.energy_density[index1], self.weights)
        if self.cal_another_attraction:
            ener_attrac += np.einsum('fi,i->', self.nuclear_attraction2, self.weights)
        print('nuclear attraction energy: ', ener_attrac)
        h_nuc = self.mol.intor('int1e_nuc')
        if len(self.mol._ecpbas) > 0:
            h_nuc += self.mol.intor_symmetric('ECPscalar')
        print('ref nuclear attraction energy: ', np.einsum('ij,ji', h_nuc, dm))

        index2 = self.ed_components.index('kinetic')
        print('kinetic energy: ', np.einsum('i,i', self.energy_density[index2], self.weights))
        h_kin = self.mol.intor('int1e_kin')
        print('ref kinetic energy: ', np.einsum('ij,ji', h_kin, dm))

        print('core energy (eV): ', np.einsum('i,i', (self.energy_density[index1]+self.energy_density[index2]), self.weights)*27.211 + ener_attrac*27.211)
        if len(self.mol._ecpbas) > 0:
            print('ecp energy(eV): ', np.einsum('i,i', self.energy_density[self.ed_components.index('ecpotential')], self.weights*27.211))
        h_core = self.mf.get_hcore()
        print('ref core energy (eV): ', np.einsum('ij,ji', h_core, dm)*27.211)


    #@profile
    def use_auxilary_basis(self, auxbasis):
        # auxilary basis for resolution of identity
        self.auxmol = df.addons.make_auxmol(self.mol, auxbasis=auxbasis)
        print('aux nbas: %d aux nao_nr: %d' % (self.auxmol.nbas, self.auxmol.nao_nr()))
        self.auxmf = scf.RKS(self.auxmol)
        #auxmf.xc = functional
        print('functional: ', self.auxmf.xc)
        self.auxmf.kernel()

        self.eri2c = self.auxmol.intor('int2c2e_sph')
        #print('eri2c dimension: ', self.eri2c.shape)
        pmol = self.mol + self.auxmol
        self.eri3c = pmol.intor('int3c2e_sph',
                shls_slice=(0,self.mol.nbas,0,self.mol.nbas,self.mol.nbas,self.mol.nbas+self.auxmol.nbas))
        self.eri3c = self.eri3c.reshape(self.mol.nao_nr(), self.mol.nao_nr(), -1)
        #print('eri3c dimension: ', self.eri3c.shape)


    #@profile
    def energy_density_j_only(self, dm, rho_value):
        r"""
        ground-state dm and rho_value
        """
        nbas = self.mol.nbas

        # coulomb energy density: (difference density, transition density)
        vele = np.zeros(self.ngrids)
        for ip0, ip1 in lib.prange(0, self.ngrids, 600):
            fakemol = gto.fakemol_for_charges(self.coords[ip0:ip1])

            pmol = self.mol + fakemol
            ints = pmol.intor('int3c2e_sph', shls_slice=(0,nbas,0,nbas,nbas,nbas+fakemol.nbas))
            vele[ip0:ip1] = np.einsum('ijp,ij->p', ints, dm)

            if ip0 == 0:
                print('fake mol basis shell: ', fakemol.nbas)
                print('ints dimension: ', ints.shape)
                print('vele dimension: ', vele[ip0:ip1].shape)

        self.energy_density.append(.5 * vele * rho_value[0])
        self.ed_components.append('coulomb1')


    #@profile
    def energy_density_jk(self, dm, rho_value, hf_type):
        r"""
        ground-state dm and rho_value
        """
        nbas = self.mol.nbas

        # coulomb energy density
        vele = np.zeros(self.ngrids)
        # exchange energy density
        vexc = np.zeros(self.ngrids)
        for ip0, ip1 in lib.prange(0, self.ngrids, 600):
            fakemol = gto.fakemol_for_charges(self.coords[ip0:ip1])

            pmol = self.mol + fakemol
            ints = pmol.intor('int3c2e_sph', shls_slice=(0,nbas,0,nbas,nbas,nbas+fakemol.nbas))

            vele[ip0:ip1] = np.einsum('ijp,ij->p', ints, dm)

            esp_ao = gto.intor_cross('int1e_ovlp', self.mol, fakemol)
            esp = np.einsum('mp,mu->up', esp_ao, dm)

            vexc[ip0:ip1] = - np.einsum('mp,up,mup->p', esp, esp, ints)

        self.energy_density.append(.5 * vele * rho_value[0])
        self.ed_components.append('coulomb1')


        hyb, alpha, omega = hf_type
        if abs(omega) < 1e-10:
            self.energy_density.append(.25 * hyb * vexc)
        else:  # For range separated Coulomb operator
            vklr = self.energy_density_k_rsh(dm, omega, alpha, hyb)
            self.energy_density.append(.25 * (hyb * vexc + vklr))
        self.ed_components.append('exchange1')


    #@profile
    def energy_density_k_rsh(self, dm, omega, alpha, hyb):
        nbas = self.mol.nbas
        # rsh exchange energy density
        vklr = np.zeros(self.ngrids)

        with self.mol.with_range_coulomb(omega):
            for ip0, ip1 in lib.prange(0, self.ngrids, 600):
                fakemol = gto.fakemol_for_charges(self.coords[ip0:ip1])

                pmol = self.mol + fakemol
                ints = pmol.intor('int3c2e_sph', shls_slice=(0,nbas,0,nbas,nbas,nbas+fakemol.nbas))

                esp_ao = gto.intor_cross('int1e_ovlp', self.mol, fakemol)
                esp = np.einsum('mp,mu->up', esp_ao, dm)

                vklr[ip0:ip1] = - np.einsum('mp,up,mup->p', esp, esp, ints)

        vklr *= (alpha - hyb)
        return vklr


    #@profile
    def check_energy_jk(self, dm, hf_type):
        eri_j1, eri_k1 = self.mf.get_jk()

        index1 = self.ed_components.index('coulomb1')
        print('coulomb energy: ', np.einsum('i,i', self.energy_density[index1], self.weights))
        print('ref coulomb energy: ', .5 * np.einsum('ij,ji', eri_j1, dm))

        hyb, alpha, omega = hf_type
        if abs(hyb) > 0:
            index1 = self.ed_components.index('exchange1')
            print('exchange energy: ', np.einsum('i,i', self.energy_density[index1], self.weights))
            if abs(omega) < 1e-10:
                print('ref exchange energy: ', -.25 * hyb * np.einsum('ij,ji', eri_k1, dm))
            else:
                eri_klr = self.mf.get_k(self.mol, dm, hermi=1, omega=omega)
                eri_k1 = hyb * eri_k1 + (alpha - hyb) * eri_klr
                print('ref exchange energy: ', -.25 * np.einsum('ij,ji', eri_k1, dm))


    #@profile
    def energy_density_xc(self):
        evf_xc = self.mf._numint.libxc.eval_xc(self.mf.xc, self.g_rho_value, deriv=2)[:3]

        self.energy_density.append(evf_xc[0] * self.g_rho_value[0])
        self.ed_components.append('functional1')


    #@profile
    def check_energy_xc(self):
        index1 = self.ed_components.index('functional1')
        print('xc energy: ', np.einsum('i,i', self.energy_density[index1], self.weights))
        print('ref xc energy (with exchange): ', self.mf.get_veff(self.mol, self.scf_dm).exc)


    #@profile
    def energy_density_coulomb_only(self, dm1, dm2, rho_value1, rho_value2):
        r"""
        dm1, dm2: ground-state, transition_symmetric dm
        rho_value1, rho_value2: difference, transition rho_value
        """
        dms = np.array((dm1, dm2))
        print('dms dimenstion: ', dms.shape)
        nbas = self.mol.nbas

        # coulomb energy density: (difference density, transition density)
        vele = np.zeros((2, self.ngrids))
        for ip0, ip1 in lib.prange(0, self.ngrids, 600):
            fakemol = gto.fakemol_for_charges(self.coords[ip0:ip1])

            pmol = self.mol + fakemol
            ints = pmol.intor('int3c2e_sph', shls_slice=(0,nbas,0,nbas,nbas,nbas+fakemol.nbas))
            vele[:, ip0:ip1] = np.einsum('ijp,dij->dp', ints, dms)

        self.energy_density.append(vele[0,:] * rho_value1[0])
        self.energy_density.append(vele[1,:] * rho_value2[0])
        self.ed_components.append('coulomb1')
        self.ed_components.append('coulomb2')


    #@profile
    def energy_density_coulomb_exchange(self, dm1, dm2, dm3, dm4,
            rho_value1, rho_value2, hf_type):
        r"""
        dm1, dm2: ground-state, difference, transition, transition_symmetric dm
        rho_value1, rho_value2: difference, transition rho_value
        """
        nbas = self.mol.nbas

        # coulomb energy density
        vele = np.zeros((2, self.ngrids))
        # exchange energy density
        vexc = np.zeros((2, self.ngrids))
        for ip0, ip1 in lib.prange(0, self.ngrids, 600):
            fakemol = gto.fakemol_for_charges(self.coords[ip0:ip1])

            pmol = self.mol + fakemol
            # V_{lambda sigma} (r)
            ints = pmol.intor('int3c2e_sph', shls_slice=(0,nbas,0,nbas,nbas,nbas+fakemol.nbas))

            #Coulomb 1-e and 2-e
            vele[0, ip0:ip1] = np.einsum('ijp,ij->p', ints, dm1)   # V(rho_gs(r))
            vele[1, ip0:ip1] = np.einsum('ijp,ij->p', ints, dm4)   # V(rho_trans(r)

            #Exchange 1-e and 2-e
            # V_lambda(r)
            esp_ao = gto.intor_cross('int1e_ovlp', self.mol, fakemol)
            esp1 = np.einsum('mp,mu->up', esp_ao, dm1)   # phi_lambda(r) P_{lambda, sigma}
            esp2 = np.einsum('mp,mu->up', esp_ao, dm2)   # phi_lambda(r) Pdiff_{lambda, sigma}
            esp3 = np.einsum('mp,mu->up', esp_ao, dm3)   # phi_lambda(r) Ptrans_{lambda, sigma}

            vexc[0, ip0:ip1] = - np.einsum('mp,up,mup->p', esp1, esp2, ints)   #P_{mu nu} phi_mu(r) P_diff{lambda sigma} phi_{lambda} (r)  V_{nu sigma} (r)
            vexc[1, ip0:ip1] = - np.einsum('mp,up,mup->p', esp3, esp3, ints)   #Ptrans_{mu nu} phi_mu(r) Ptrans{lambda sigma} phi_{lambda} (r)  V_{nu sigma} (r)
            esp3 = np.einsum('mp,mu->up', esp_ao, dm3.T)   # phi_lambda(r) Ptrans_{lambda, sigma}
            vexc[1, ip0:ip1] -= np.einsum('mp,up,mup->p', esp3, esp3, ints)
            vexc[1, ip0:ip1] *= 0.5

        self.energy_density.append(vele[0,:] * rho_value1[0])   # V(rho_gs(r)) * rho_diff(r)
        self.energy_density.append(vele[1,:] * rho_value2[0])   # V(rho_trans(r)) * trans(r)
        self.ed_components.append('coulomb1')
        self.ed_components.append('coulomb2')


        hyb, alpha, omega = hf_type
        if abs(omega) < 1e-10:
            self.energy_density.append(.5 * hyb * vexc[0,:])
            self.energy_density.append(2.0 * hyb * vexc[1,:])
        else:
            vklr = self.energy_density_exchange_rsh(dm1, dm2, dm3, omega, alpha, hyb)
            self.energy_density.append(.5 * (hyb * vexc[0,:] + vklr[0,:]))
            self.energy_density.append(2.0 * (hyb * vexc[1,:] + vklr[1,:]))

        self.ed_components.append('exchange1')
        self.ed_components.append('exchange2')


    #@profile
    def energy_density_exchange_rsh(self, dm1, dm2, dm3,
            omega, alpha, hyb):
        r"""
        dm1, dm2, dm3: ground-state, difference, transition dm
        """
        nbas = self.mol.nbas
        # rsh exchange energy density
        vklr = np.zeros((2, self.ngrids))

        with self.mol.with_range_coulomb(omega):
            for ip0, ip1 in lib.prange(0, self.ngrids, 600):
                fakemol = gto.fakemol_for_charges(self.coords[ip0:ip1])

                pmol = self.mol + fakemol
                ints = pmol.intor('int3c2e_sph', shls_slice=(0,nbas,0,nbas,nbas,nbas+fakemol.nbas))

                esp_ao = gto.intor_cross('int1e_ovlp', self.mol, fakemol)
                esp1 = np.einsum('mp,mu->up', esp_ao, dm1)
                esp2 = np.einsum('mp,mu->up', esp_ao, dm2)
                esp3 = np.einsum('mp,mu->up', esp_ao, dm3)

                vklr[0, ip0:ip1] = - np.einsum('mp,up,mup->p', esp1, esp2, ints)
                vklr[1, ip0:ip1] = - np.einsum('mp,up,mup->p', esp3, esp3, ints)
                esp3 = np.einsum('mp,mu->up', esp_ao, dm3.T)   # phi_lambda(r) Ptrans_{lambda, sigma}
                vklr[1, ip0:ip1] -= np.einsum('mp,up,mup->p', esp3, esp3, ints)
                vklr[1, ip0:ip1] *= 0.5

        vklr *= (alpha - hyb)
        return vklr


    #@profile
    def check_energy_coulomb_exchange(self, dm, hf_type):
        eri_j1, eri_k1 = self.mf.get_jk()

        index1, index2 = self.ed_components.index('coulomb1'), self.ed_components.index('coulomb2')
        print('coulomb energy 1 (eV): ', np.einsum('i,i', self.energy_density[index1], self.weights)*27.211)
        print('ref coulomb energy 1 (eV): ', np.einsum('ij,ji', eri_j1, dm)*27.211)
        print('coulomb energy 2 (eV): ', np.einsum('i,i', self.energy_density[index2], self.weights)*27.211)

        hyb, alpha, omega = hf_type
        if abs(hyb) > 0:
            index1, index2 = self.ed_components.index('exchange1'), self.ed_components.index('exchange2')
            print('exchange energy 1 (eV): ', np.einsum('i,i', self.energy_density[index1], self.weights)*27.211)
            if abs(omega) < 1e-10:
                print('ref exchange energy 1 (eV): ', -.5 * hyb * np.einsum('ij,ji', eri_k1, dm)*27.211)
            else:
                eri_klr = self.mf.get_k(self.mol, self.scf_dm, hermi=1, omega=omega)
                eri_k1 = hyb * eri_k1 + (alpha - hyb) * eri_klr
                print('ref exchange energy: ', -.5 * np.einsum('ij,ji', eri_k1, dm))
            print('exchange energy 2 (eV): ', np.einsum('i,i', self.energy_density[index2], self.weights)*27.211)


    #@profile
    def energy_density_functional(self):
        evf_xc = self.mf._numint.libxc.eval_xc(self.mf.xc, self.g_rho_value, deriv=2)[:3]
        vrho, vgamma = evf_xc[1][:2]
        frho, frhogamma, fgg = evf_xc[2][:3]
        #print('vgamma dimension: ', len(vgamma))
        #print('frho dimension: ', len(frho))

        # exchange-correlation: difference density part
        wv = np.zeros((4, self.ngrids))
        wv[0] = vrho
        wv[1:4] = 2.0 * vgamma * self.g_rho_value[1:4]
        self.energy_density.append(np.einsum('np,np->p', wv, self.d_rho_value[:4]))
        self.ed_components.append('functional1')

        # exchange-correlation: transition density part
        wv = np.zeros((4, self.ngrids))
        sigma1 = np.einsum('xr,xr->r', self.g_rho_value[1:4], self.t_rho_value[1:4])
        wv[0] = frho * self.t_rho_value[0] + 2.0 * frhogamma * sigma1
        f_ov = 2.0 * frhogamma * self.t_rho_value[0] + 4.0 * fgg * sigma1
        wv[1:4] = np.einsum('r,xr->xr', f_ov, self.g_rho_value[1:4])
        wv[1:4] += np.einsum('r,xr->xr', 2*vgamma, self.t_rho_value[1:4])
        self.energy_density.append(np.einsum('xr,xr->r', wv, self.t_rho_value))
        self.ed_components.append('functional2')

    #@profile
    def check_energy_functional(self):
        index1, index2 = self.ed_components.index('functional1'), self.ed_components.index('functional2')
        print('xc energy 1 (eV): ', np.einsum('i,i', self.energy_density[index1], self.weights)*27.211)
        print('xc energy 2 (eV): ', np.einsum('i,i', self.energy_density[index2], self.weights)*27.211)


    def sum_energy_density_component(self):
        total_energy_density = np.einsum('xp->p', self.energy_density)
        if self.cal_another_attraction:
            total_energy_density += np.einsum('fp->p', self.nuclear_attraction2)
        self.energy_density.append(total_energy_density)
        self.ed_components.append('total')


    #@profile
    def decompose_energy_density_gs(self, hf_type, has_xc):
        self.energy_density = []
        self.ed_components = []
        self.nuclear_attraction2 = [] # temporarily save nuclear partition
        """
        ['attraction', 'kinetic', 'coulomb1', 'coulomb2',
            'exchange1', 'exchange2', 'functional1', 'functional2']
        """

        self.energy_density_core(self.scf_dm, self.g_rho_value)
        if abs(hf_type[0]) > 0:
            #with TicToc('coulomb and exchange energy'):
            self.energy_density_jk(self.scf_dm, self.g_rho_value, hf_type)
        else:
            #with TicToc('coulomb energy'):
            self.energy_density_j_only(self.scf_dm, self.g_rho_value)
        if has_xc:
            #with TicToc('xc energy'):
            self.energy_density_xc()
        self.sum_energy_density_component()

        if self.debug:
            #with TicToc('check grid energy'):
            self.check_energy_core(self.scf_dm)
            self.check_energy_jk(self.scf_dm, hf_type)
            if has_xc:
                self.check_energy_xc()
        #print('total electron energy: ', np.einsum('p,p->', self.energy_density[-1], self.weights))
        energy_density = np.einsum('fp,p->f', self.energy_density, self.weights)
        np.savetxt(sys.stdout.buffer, energy_density.reshape(1, energy_density.shape[0]), fmt='%13.7f', header='total electron energy:')
        print('ref total electron energy: ', self.mf.energy_elec()[0])


    #@profile
    def decompose_energy_density_es(self, directory, hf_type, has_xc,
            ie, grid_type=1, dohirshfeld=True, dobecke=True):
        self.energy_density = []
        self.ed_components = []
        self.nuclear_attraction2 = [] # temporarily save nuclear partition

        #with TicToc('es density on grids'):
        self.cal_density_matrices(ie)
        self.grab_es_density_on_grids()
        #with TicToc('core energy'):
        self.energy_density_core(self.diff_dm, self.d_rho_value)

        if abs(hf_type[0]) > 0:
            #with TicToc('coulomb and exchange energy'):
            self.energy_density_coulomb_exchange(self.scf_dm, self.diff_dm, self.trans_dm, self.trans_dm_sym,
                    self.d_rho_value, self.t_rho_value, hf_type)
        else:
            #with TicToc('coulomb energy'):
            self.energy_density_coulomb_only(self.scf_dm, self.trans_dm_sym,
                    self.d_rho_value, self.t_rho_value)
        if has_xc:
            #with TicToc('xc energy'):
            self.energy_density_functional()
        self.sum_energy_density_component()

        if self.debug:
            #with TicToc('check grid energy'):
            self.check_energy_core(self.diff_dm)
            self.check_energy_coulomb_exchange(self.diff_dm, hf_type)
            if has_xc:
                self.check_energy_functional()
        self.es_energy.append(np.einsum('fp,p->f', self.energy_density, self.weights)*27.211)


    def plot_gs_energy_density(self, directory, plotnum):
        orb_on_grid = np.einsum('pm,mi->pi', self.ao_value[0], self.coeffs)
        for i in range(self.nocc-plotnum, self.nocc+plotnum):
            self.cc.write(orb_on_grid[:,i].reshape(self.cc.nx, self.cc.ny, self.cc.nz),
                    directory+'gs'+str(i+1)+'_orbital.cube')

        self.cc.write(self.g_rho_value[0].reshape(self.cc.nx, self.cc.ny, self.cc.nz),
                directory+'gs_density.cube')

        for n in range(len(self.ed_components)):
            self.cc.write(self.energy_density[n].reshape(self.cc.nx, self.cc.ny, self.cc.nz),
                    directory+'gs_energy_density_'+self.ed_components[n]+'.cube')


    def plot_es_energy_density(self, directory, ie):
        for n in range(len(self.ed_components)):
            self.cc.write(self.energy_density[n].reshape(self.cc.nx, self.cc.ny, self.cc.nz),
                    directory+'es'+str(ie+1)+'_energy_density_'+self.ed_components[n]+'.cube')

        self.cc.write(self.t_rho_value[0].reshape(self.cc.nx, self.cc.ny, self.cc.nz),
                directory+'es'+str(ie+1)+'_transition_density.cube')
        self.cc.write(self.d_rho_value[0].reshape(self.cc.nx, self.cc.ny, self.cc.nz),
                directory+'es'+str(ie+1)+'_difference_density.cube')


    #@profile
    def decompose_energy_density(self, directory, nstates=1, estate=0,
            plotnum=3, grid_type=1, nxyz=0.1, dohirshfeld=True, dobecke=True,
            withcharge=False, decompose_es=True, cal_another_attraction=False):
        auxbasis = '6-31g'
        #auxbasis = 'weigend'

        self.cal_another_attraction = cal_another_attraction

        dft_type = 0
        if self.mf.xc != 'wb97xd':
            dft_type = dft.xcfun.parse_xc(self.mf.xc)
        else:
            dft_type = dft.libxc.parse_xc(self.mf.xc)
            omega, alpha, hyb = self.mf._numint.rsh_and_hybrid_coeff(self.mf.xc)
            dft_type[0][:] = [hyb, alpha, omega]

        has_xc = True if dft_type[1]!=[] else False
        hyb, alpha, omega = dft_type[0]
        print('parse_xc: ', dft_type, ' has_xc: ', has_xc)
        print('hyb: ', hyb, ' alpha: ', alpha, ' omega: ', omega)
        self.dft_type = dft_type

        #with TicToc('creat mesh grids'):
        self.creat_mesh_grids(grid_type, nxyz)
        #with TicToc('gs density on grids'):
        self.grab_gs_density_on_grids()

        self.decompose_energy_density_gs(dft_type[0], has_xc)

        if grid_type == 1:
            self.partition_energy_or_charge(-1, dohirshfeld, dobecke)
        elif grid_type == 2:
            self.plot_gs_energy_density(directory, plotnum)


        if nstates >= 1:
            #with TicToc('TDDFT'):
            self.cal_excited_state(nstates)
            if estate == -1:
                estate = np.arange(nstates)
            print(estate)

            if decompose_es:
                self.es_energy = []
                for ie in estate:
                    self.decompose_energy_density_es(directory, dft_type[0], has_xc,
                            ie, grid_type, dohirshfeld, dobecke)

                    if grid_type == 1:
                        self.partition_energy_or_charge(ie, dohirshfeld, dobecke, withcharge)
                    elif grid_type == 2:
                        self.plot_es_energy_density(directory, ie)

                np.savetxt(sys.stdout.buffer, np.array(self.es_energy), fmt='%13.7f', header='excited-state total excitation energy (eV):')

            if grid_type == 1:
                self.cal_excitation_lambda_factor(estate)
                self.cal_overlap_detach_attach_density(estate)


        if grid_type == 1:
            if self.frag_atom and dohirshfeld:
                hirshfeld_energy = np.array(self.hirshfeld_energy[1:])*27.211
                hirshfeld_charge = np.array(self.hirshfeld_charge)
                #print(hirshfeld_energy)
                print('\nhirshfeld fragment energy (eV):')
                for n in range(hirshfeld_energy.shape[0]):
                    for p in range(2):
                        for i in range(len(self.ed_components)):
                            if i < len(self.ed_components)-1:
                                print('%13.7f ' % hirshfeld_energy[n,p,i], end=' ')
                            else:
                                print('%13.7f ' % hirshfeld_energy[n,p,i])
                print('\nhirshfeld fragment charge:')
                for n in range(hirshfeld_charge.shape[0]):
                    print('%10.5f %10.5f' % (hirshfeld_charge[n,0], hirshfeld_charge[n,1]))

            if self.frag_atom and dobecke:
                becke_energy = np.array(self.becke_energy[1:])*27.211
                becke_charge = np.array(self.becke_charge)
                print('\nbecke fragment energy (eV):')
                for n in range(becke_energy.shape[0]):
                    for p in range(2):
                        for i in range(len(self.ed_components)):
                            if i < len(self.ed_components)-1:
                                print('%13.7f ' % becke_energy[n,p,i], end=' ')
                            else:
                                print('%13.7f ' % becke_energy[n,p,i])
                print('\nbecke fragment charge:')
                for n in range(becke_charge.shape[0]):
                    print('%10.5f %10.5f' % (becke_charge[n,0], becke_charge[n,1]))


    #@profile
    def partition_energy_or_charge(self, ie, dohirshfeld=True, dobecke=True, withcharge=False):
        if ie < 0:
            # this is first call
            if self.frag_atom and dohirshfeld:
                self.group_hirshfeld_partition()
            if dobecke:
                self.becke_partition()

        # default give ground state density value
        relaxed_rho_value = self.g_rho_value[0]

        if ie>=0 and withcharge:
            self.relaxed_z_kernel(ie)
            relaxed_dm = self.diff_dm + self.z1_dm
            relaxed_rho_value = self.density_on_grids(relaxed_dm, xctype='GGA')[0]

        index1 = self.ed_components.index('attraction')

        if self.frag_atom and dohirshfeld:
            hirshfeld_energy = np.einsum('fp,ip,p->fi', self.hirshfeld_partion, self.energy_density, self.weights)
            if self.cal_another_attraction:
                hirshfeld_atom_attraction = np.einsum('dp,p->d', self.nuclear_attraction2, self.weights)
                hirshfeld_atom_attraction = np.array([np.einsum('n->', hirshfeld_atom_attraction[:self.frag1]), np.einsum('n->', hirshfeld_atom_attraction[self.frag1:])])
                print('hirshfeld nuclear attraction two: ', hirshfeld_energy[:,index1].T*27.211, ' ', hirshfeld_atom_attraction*27.211)
                hirshfeld_energy[:,index1] += hirshfeld_atom_attraction
            self.hirshfeld_energy.append(hirshfeld_energy)
            print('hirshfeld fragment energy: ', self.hirshfeld_energy[-1][:,-1])

            # we dont add nuclear charge to ground state population
            hirshfeld_charge = - np.einsum('fp,p,p->f', self.hirshfeld_partion, relaxed_rho_value, self.weights)
            self.hirshfeld_charge.append(hirshfeld_charge)
            print('hirshfeld fragment charge: ', self.hirshfeld_charge[-1])

        if dobecke:
            becke_energy = np.zeros((len(self.ed_components), self.mol.natm))
            becke_charge = np.zeros(self.mol.natm)
            for ia in range(self.mol.natm):
                ip0, ip1 = self.atomic_ngrids[ia], self.atomic_ngrids[ia+1]
                #print('ip0, ', ip0, ' ip1 ', ip1)
                for i in range(len(self.ed_components)):
                    becke_energy[i,ia] = np.einsum('p,p->', self.energy_density[i][ip0:ip1], self.weights[ip0:ip1])
                becke_charge[ia] = - np.einsum('p,p->', relaxed_rho_value[ip0:ip1], self.weights[ip0:ip1])

            if self.cal_another_attraction:
                becke_attraction2 = np.zeros(self.mol.natm)
                for ia in range(self.mol.natm):
                    becke_attraction2[ia] = np.einsum('p,p->', self.nuclear_attraction2[ia], self.weights)
                if self.frag_atom:
                    print('becke nuclear attraction two: ', np.array([np.einsum('n->', becke_energy[index1,:self.frag1]), np.einsum('n->', becke_energy[index1,self.frag1:])])*27.211, ' ', np.array([np.einsum('n->', becke_attraction2[:self.frag1]), np.einsum('n->', becke_attraction2[self.frag1:])])*27.211)
                for ia in range(self.mol.natm):
                    becke_energy[index1,ia] += becke_attraction2[ia]

            if ie < 0:
                for ia in range(self.mol.natm):
                    becke_charge[ia] += self.mol.atom_charge(ia)

            print('becke atomic charge: ', becke_charge)

            if self.frag_atom:
                tbecke_energy = np.zeros((2, len(self.ed_components)))
                tbecke_energy[0] = np.einsum('in->i', becke_energy[:,:self.frag1]).T
                tbecke_energy[1] = np.einsum('in->i', becke_energy[:,self.frag1:]).T
                self.becke_energy.append(tbecke_energy)
                print('becke fragment energy: ', self.becke_energy[-1][:,-1])
                self.becke_charge.append([np.einsum('n->', becke_charge[:self.frag1]), np.einsum('n->', becke_charge[self.frag1:])])
            else:
                self.becke_energy.append(becke_energy)
                print('becke atomic energy: ', self.becke_energy[-1][:,-1])
                self.becke_charge.append(becke_charge)


    #@profile
    def group_hirshfeld_partition(self):
        r"""
        density ratio of fragments on grid points
        """

        f_rho_value = np.zeros((3, self.ngrids))
        for f in range(2):
            fmol = gto.M(
                    verbose = 1,
                    atom = self.frag_atom[f],
                    basis = self.mol.basis,
                    ecp = self.mol.ecp,
                    charge = self.frag_charge[f],
            )

            fmf = scf.RKS(fmol)
            fmf.xc = self.mf.xc
            fmf.kernel()
            fdm = fmf.make_rdm1()
            print('fragment %d gs energy: %12.6f' %(f+1, fmf.energy_elec()[0]))
            print('fmf nuclear: ', fmol.energy_nuc())
            #fmf.analyze(10)

            blksize = min(8000, self.ngrids)
            for ip0, ip1 in lib.prange(0, self.ngrids, blksize):
                tmp_ao_value = numint.eval_ao(fmol, self.coords[ip0:ip1], deriv=0)
                f_rho_value[f, ip0:ip1] = numint.eval_rho(fmol, tmp_ao_value, fdm, xctype="LDA")
                #f_rho_value[f, ip0:ip1] = numint.eval_rho(fmol, self.ao_value[0,ip0:ip1,:], fdm, xctype="LDA")

        f_rho_value[2] = (f_rho_value[0] + f_rho_value[1])

        self.hirshfeld_partion = np.zeros((2, self.ngrids))
        for f in range(2):
            #self.hirshfeld_partion[f,:] = np.true_divide(f_rho_value[f], f_rho_value[2])
            self.hirshfeld_partion[f] = f_rho_value[f] / f_rho_value[2]
        print('fragment electron: ', np.einsum('fp,p->f', f_rho_value, self.weights))
        self.hirshfeld_energy = []
        self.hirshfeld_charge = []


    #@profile
    def becke_partition(self):
        def get_atomic_ngrid_index():
            atomic_ngrids = [0]
            wi = 0

            atom_grids_tab = self.mf.grids.gen_atomic_grids(self.mol)
            for ia in range(self.mol.natm):
                coords, _ = atom_grids_tab[self.mol.atom_symbol(ia)]
                coords = coords + self.mol.atom_coord(ia)
                num_ngrids = coords.shape[0]
                #print('atomic grids: ', num_ngrids)
                wj = 0
                while wj < num_ngrids and wi < self.ngrids:
                    if np.sqrt(np.sum((coords[wj]-self.coords[wi]) ** 2)) <= 1e-24:
                        wi += 1
                        wj += 1
                    else:
                        wj += 1
                #print('wi: ', wi, ' wj: ', wj)
                atomic_ngrids.append(wi)
            return atomic_ngrids

        self.atomic_ngrids = get_atomic_ngrid_index()
        print('pruned atomic_ngrids: ', self.atomic_ngrids)
        self.becke_energy = []
        self.becke_charge = []


    #@profile
    def mayer_bond_order(self, bondatoms, nstates, estate):
        [[s1, e1], [s2, e2]] = self.mol.aoslice_by_atom()[bondatoms, 2:4]
        #aorange = self.mol.aoslice_by_atom()[bondatoms, 2:4]
        print('aorange: ', s1, ' ', e1, ' ', s2, ' ', e2)
        ovlp = self.mf.get_ovlp()

        den = [self.scf_dm]
        if nstates > 0:
            self.cal_excited_state(nstates)

            if estate == -1:
                estate = np.arange(nstates)
            print(estate)
            for ie in estate:
                self.relaxed_z_kernel(ie)
                den.append(self.diff_dm + self.z1_dm)
        den = np.array(den)

        mayerb1 = np.einsum('emu,un->emn', den[:,s1:e1,:], ovlp[:,s2:e2])
        mayerb2 = np.einsum('emu,un->emn', den[:,s2:e2,:], ovlp[:,s1:e1])
        mayer_bo = np.einsum('emu,eum->e', mayerb1, mayerb2)
        print('mayer bond order: ', mayer_bo)


    def relaxed_z_kernel(self, ie, singlet=True, atmlst=None,
            max_memory=2000):
        # now we just copy the first part of grad/tdrks kernel
        from functools import reduce
        from pyscf.grad.tdrks import _contract_xc_kernel
        td_grad = self.td.Gradients()

        mo_occ = self.mf.mo_occ
        mo_coeff = self.mf.mo_coeff
        nao, nmo = mo_coeff.shape

        xpy = (self.tdxy[0]+self.tdxy[1]).reshape(self.nocc,self.nvir).T
        xmy = (self.tdxy[0]-self.tdxy[1]).reshape(self.nocc,self.nvir).T

        dvv = np.einsum('ai,bi->ab', xpy, xpy) + np.einsum('ai,bi->ab', xmy, xmy)
        doo =-np.einsum('ai,aj->ij', xpy, xpy) - np.einsum('ai,aj->ij', xmy, xmy)
        dmzvop = reduce(np.dot, (self.orbv, xpy, self.orbo.T))
        dmzvom = reduce(np.dot, (self.orbv, xmy, self.orbo.T))
        dmzoo = reduce(np.dot, (self.orbo, doo, self.orbo.T))
        dmzoo+= reduce(np.dot, (self.orbv, dvv, self.orbv.T))


        dft_type = self.dft_type
        has_xc = True if dft_type[1]!=[] else False

#        self.mf._numint.libxc = dft.xcfun
        if has_xc:
            #mem_now = lib.current_memory()[0]
            ni = self.mf._numint
#            ni.libxc.test_deriv_order(self.mf.xc, 3, raise_error=True)

            # dm0 = mf.make_rdm1(mo_coeff, mo_occ), but it is not used when computing
            # fxc since rho0 is passed to fxc function.
            dm0 = None
            rho0, vxc, fxc = ni.cache_xc_kernel(self.mf.mol, self.mf.grids, self.mf.xc,
                                                [mo_coeff]*2, [mo_occ*.5]*2, spin=1)
            f1vo, f1oo, vxc1, k1ao = _contract_xc_kernel(td_grad, self.mf.xc, dmzvop,
                                        dmzoo, True, True, singlet, max_memory)

            hyb, alpha, omega = dft_type[0]

            if abs(hyb) > 1e-10:
                dm = (dmzoo, dmzvop+dmzvop.T, dmzvom-dmzvom.T)
                vj, vk = self.mf.get_jk(self.mol, dm, hermi=0)
                vk *= hyb
                if abs(omega) > 1e-10:
                    vk += self.mf.get_k(self.mol, dm, hermi=0, omega=omega) * (alpha-hyb)
                veff0doo = vj[0] * 2 - vk[0] + f1oo[0] + k1ao[0] * 2
                wvo = reduce(np.dot, (self.orbv.T, veff0doo, self.orbo)) * 2
                if singlet:
                    veff = vj[1] * 2 - vk[1] + f1vo[0] * 2
                else:
                    veff = -vk[1] + f1vo[0] * 2
                veff0mop = reduce(np.dot, (mo_coeff.T, veff, mo_coeff))
                wvo -= np.einsum('ki,ai->ak', veff0mop[:self.nocc,:self.nocc], xpy) * 2
                wvo += np.einsum('ac,ai->ci', veff0mop[self.nocc:,self.nocc:], xpy) * 2
                veff = -vk[2]
                veff0mom = reduce(np.dot, (mo_coeff.T, veff, mo_coeff))
                wvo -= np.einsum('ki,ai->ak', veff0mom[:self.nocc,:self.nocc], xmy) * 2
                wvo += np.einsum('ac,ai->ci', veff0mom[self.nocc:,self.nocc:], xmy) * 2
            else:
                vj = self.mf.get_j(self.mol, (dmzoo, dmzvop+dmzvop.T), hermi=1)
                veff0doo = vj[0] * 2 + f1oo[0] + k1ao[0] * 2
                wvo = reduce(np.dot, (self.orbv.T, veff0doo, self.orbo)) * 2
                if singlet:
                    veff = vj[1] * 2 + f1vo[0] * 2
                else:
                    veff = f1vo[0] * 2
                veff0mop = reduce(np.dot, (mo_coeff.T, veff, mo_coeff))
                wvo -= np.einsum('ki,ai->ak', veff0mop[:self.nocc,:self.nocc], xpy) * 2
                wvo += np.einsum('ac,ai->ci', veff0mop[self.nocc:,self.nocc:], xpy) * 2
                veff0mom = np.zeros((nmo,nmo))
            def fvind(x):
                # Cannot make call to .base.get_vind because first order orbitals are solved
                # through closed shell ground state CPHF.
                dm = reduce(np.dot, (self.orbv, x.reshape(self.nvir,self.nocc), self.orbo.T))
                dm = dm + dm.T
                # Call singlet XC kernel contraction, for closed shell ground state
                vindxc = numint.nr_rks_fxc_st(ni, self.mol, self.mf.grids, self.mf.xc,
                        dm0, dm, 0, singlet, rho0, vxc, fxc, max_memory)
                if abs(hyb) > 1e-10:
                    vj, vk = self.mf.get_jk(self.mol, dm)
                    veff = vj * 2 - hyb * vk + vindxc
                    if abs(omega) > 0e-10:
                        veff -= self.mf.get_k(self.mol, dm, hermi=1, omega=omega) * (alpha-hyb)
                else:
                    vj = self.mf.get_j(self.mol, dm)
                    veff = vj * 2 + vindxc
                return reduce(np.dot, (self.orbv.T, veff, self.orbo)).ravel()
        else:
            vj, vk = self.mf.get_jk(self.mol, (dmzoo, dmzvop+dmzvop.T, dmzvom-dmzvom.T), hermi=0)
            veff0doo = vj[0] * 2 - vk[0]
            wvo = reduce(np.dot, (self.orbv.T, veff0doo, self.orbo)) * 2
            if singlet:
                veff = vj[1] * 2 - vk[1]
            else:
                veff = -vk[1]
            veff0mop = reduce(np.dot, (mo_coeff.T, veff, mo_coeff))
            wvo -= np.einsum('ki,ai->ak', veff0mop[:self.nocc,:self.nocc], xpy) * 2
            wvo += np.einsum('ac,ai->ci', veff0mop[self.nocc:,self.nocc:], xpy) * 2
            veff = -vk[2]
            veff0mom = reduce(np.dot, (mo_coeff.T, veff, mo_coeff))
            wvo -= np.einsum('ki,ai->ak', veff0mom[:self.nocc,:self.nocc], xmy) * 2
            wvo += np.einsum('ac,ai->ci', veff0mom[self.nocc:,self.nocc:], xmy) * 2
            def fvind(x):  # For singlet, closed shell ground state
                dm = reduce(np.dot, (self.orbv, x.reshape(self.nvir,self.nocc), self.orbo.T))
                vj, vk = self.mf.get_jk(self.mol, (dm+dm.T))
                return reduce(np.dot, (self.orbv.T, vj*2-vk, self.orbo)).ravel()

        z1 = scf.cphf.solve(fvind, self.mf.mo_energy, mo_occ, wvo,
                        max_cycle=td_grad.cphf_max_cycle, tol=td_grad.cphf_conv_tol)[0]
        z1 = z1.reshape(self.nvir,self.nocc)
        #self.z1_dm = np.einsum('ai,pa,qi->pq', z1, self.orbv, self.orbo.conj())
        self.z1_dm  = reduce(np.dot, (self.orbv, z1, self.orbo.T))



if __name__ == '__main__':

    atom = """
      H   -0.0000000    0.4981795    0.7677845
      O   -0.0000000   -0.0157599    0.0000000
      H    0.0000000    0.4981795   -0.7677845
      """
    #functionals = ['hf', 'pbe', 'b3lyp' ,'camb3lyp', 'wb97xd']
    functionals = ['camb3lyp', 'wb97xd']
    basis = '6-31g*'
    nstates = 2
    estate = -1
    plotnum = 3

    directory = str(sys.argv[1])+'/' if len(sys.argv)>1 else ''
    grid_type = 1
    nxyz = [101, 101, 101, 0.1]

    dobecke = True
    dohirshfeld = True

    debug = True

    for functional in functionals:
        ed = EnergyDensity(atom, functional, basis, debug=debug)
        ed.decompose_energy_density(directory, nstates, estate, plotnum, grid_type, nxyz, dohirshfeld, dobecke)
        ed.mayer_bond_order([0,1], nstates, estate)

        print('\n')

