from lumeq import sys, np
from lumeq.utils import print_matrix, convert_units
from lumeq.dynamics import (
        ElectronicStep,
        GrassmannStep,
        CurvyStep,
        ExtendedLagStep,
        PhotonStep,
        PhotonStep2,
        NuclearStep,
        )
from lumeq.plot import plt

kT_AU_to_Kelvin = 0.25 * 9.1093826e-31 * (1.60217653e-19**2 / 8.854187817e-12 / 6.6260693e-34)**2 / 1.3806505e-23


class MolecularDynamics():
    def __init__(self, key, ed_key={}, ph_key={}, **kwargs):
        self.ed_method  = key.pop('ed_method', 'normal')
        self.ph_method  = key.pop('ph_method', None)

        self.total_time = key.pop('total_time', 4000) # au
        self.dt         = key.get('dt', 10) # au

        self.nsteps = int(self.total_time/self.dt) + 1
        print('running molecular dynamics in %5d steps and total time %6.3f fs\n' % (self.nsteps, convert_units(self.total_time, 'au', 'fs')))

        self.ndstep = self.set_nuclear_step(key)
        self.edstep = self.set_electronic_step(ed_key, atmsym=key['atmsym'], **kwargs)
        self.phstep = self.set_photon_step(ph_key, dt=self.dt)

        self.md_time_total_energy = np.zeros(self.nsteps)
        self.md_time_coordinate = np.zeros((self.nsteps, self.ndstep.natoms, 3))
        self.md_time_velocity = np.zeros((self.nsteps, self.ndstep.natoms, 3))
        self.md_time_dipole = np.zeros((self.nsteps, 3))
        self.md_time_mo_energy = [None]*self.nsteps


    def set_nuclear_step(self, key):
        return NuclearStep(key)


    def set_electronic_step(self, key, **kwargs):
        if self.ed_method == 'extended_lag':
            return ExtendedLagStep(key, **kwargs)
        elif self.ed_method == 'curvy':
            return CurvyStep(key, **kwargs)
        elif self.ed_method == 'grassmann':
            return GrassmannStep(key, **kwargs)
        elif self.ed_method == 'normal':
            return ElectronicStep(key, **kwargs)


    def set_photon_step(self, key, **kwargs):
        if self.ph_method == 'quantum':
            return PhotonStep(key, **kwargs)
        if self.ph_method == 'quantum2':
            return PhotonStep2(key, **kwargs)
        else:
            return None


    def run_dynamics(self):
        # assign local variables for class attributes
        ndstep, edstep, phstep = self.ndstep, self.edstep, self.phstep

        coords = ndstep.coordinate # equal sign used here as a pointer!
        # coords will change when self.ndstep.coordinate changes!

        kwargs = {}
        photon_energy = 0.
        if phstep:
            print('photon frequency (au) is:', phstep.frequency)
            print('photon-molecule coupling strength:', phstep.c_lambda)
            kwargs['c_lambda'] = phstep.c_lambda
            photon_energy = phstep.energy

        if 'restart' in ndstep.init_method:
            # get dynamical variables from nd class
            energy = ndstep.__dict__.pop('energy')
            dipole = ndstep.__dict__.pop('dipole')
            init_time = ndstep.__dict__.pop('init_time') # in au!
            force = ndstep.force

            etot, et, ndstep.kinetic = energy[:3]
            if phstep:
                photon_energy = energy[3]

        else: # default md initial step
            init_time = 0.
            et, force = edstep.init_electronic_density_static(coords, **kwargs)
            force = ndstep.project_force(force)

            dipole = edstep.mf.dip_moment(unit='au', verbose=0)

            if phstep and phstep.init_method == 'minimum':
                phstep.get_minimim_displacement(dipole)
                print('photon initial coordinate:\n', phstep.coordinate)

            etot = et + ndstep.kinetic + photon_energy

        print('current time: %7.3f fs' % convert_units(init_time, 'au', 'fs'), end='  ')
        print('temperature: %4.2f K' % float(ndstep.temperature * kT_AU_to_Kelvin))
        print('total energy (au): %15.10f  potential: %15.10f  kinetic: %15.10f' % (etot, et, ndstep.kinetic), end='  ')
        if phstep: print('photon: %15.10f' % photon_energy)
        else: print('')
        print_matrix('nuclear force:', ndstep.force)
        print_matrix('nuclear velocity:', ndstep.velocity)
        print_matrix('nuclear coordinate (AA):', convert_units(coords, 'bohr', 'angstrom'))
        print_matrix('molecular dipole:', dipole)

        self.md_time_total_energy[0] = etot
        self.md_time_coordinate[0] = coords
        self.md_time_velocity[0] = ndstep.velocity
        self.md_time_dipole[0] = dipole
        self.md_time_mo_energy[0] = edstep.mf.mo_energy


        # loop times
        for ti in range(1, self.nsteps):
            ndstep.update_coordinate_velocity(force, 1,
                        force_func=edstep.update_electronic_density_static, **kwargs)
            #coords = self.ndstep.coordinate # dont need to reassign!

            if phstep:
                # get bilinear coefficient and photon energy
                kwargs.update(phstep.update_density(dipole, ndstep.dt, 1))

            et, force = edstep.update_electronic_density_static(coords, **kwargs)

            if self.ed_method == 'curvy':
                et, force = edstep.update_electronic_density_static2(coords, **kwargs)

            dipole = edstep.mf.dip_moment(unit='au', verbose=0)


            # velocity_verlet is cumbersome
            force = ndstep.update_coordinate_velocity(force, 2)

            if phstep:
                # get photon energy
                phstep.update_density(dipole, ndstep.dt, 2)
                photon_energy = phstep.energy

            etot = et + ndstep.kinetic + photon_energy

            print('current time: %7.3f fs' % convert_units(init_time+ti*self.dt, 'au', 'fs'), end='  ')
            print('temperature: %4.2f K' % float(ndstep.temperature * kT_AU_to_Kelvin))
            print('total energy (au): %15.10f  potential: %15.10f  kinetic: %15.10f' % (etot, et, ndstep.kinetic), end='  ')
            if phstep: print('photon: %15.10f' % photon_energy)
            else: print('')
            print_matrix('nuclear force:', ndstep.force)
            print_matrix('nuclear velocity:', ndstep.velocity)
            print_matrix('nuclear coordinate (AA):', convert_units(coords, 'bohr', 'angstrom'))
            print_matrix('molecular dipole:', dipole)

            self.md_time_total_energy[ti] = etot
            self.md_time_coordinate[ti] = coords
            self.md_time_velocity[ti] = ndstep.velocity
            self.md_time_dipole[ti] = dipole
            self.md_time_mo_energy[ti] = edstep.mf.mo_energy

    kernel = run_dynamics


    def plot_time_variables(self, fig_name=None):
        time_line = np.linspace(0, convert_units(self.total_time, 'au', 'fs'), self.nsteps)
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11,6), sharex=True)

        coords = convert_units(self.md_time_coordinate, 'bohr', 'angstrom')
        energy = convert_units(self.md_time_total_energy, 'hartree', 'kcal')

        ax[0].plot(time_line, np.linalg.norm(coords[:,1]-coords[:,0], axis=1))
        ax[0].set_ylabel('He--H$^+$ ($\\AA$)')
        ax[1].plot(time_line, energy-energy[0])
        ax[1].set_xlabel('Time (fs)')
        ax[1].set_ylabel('$\\Delta$E (kcal/mol)')

        plt.tight_layout()
        if fig_name:
            plt.savefig(fig_name)
        else:
            plt.show()


def plot_time_variables(total_time, nsteps, dists, energies):
    time_line = np.linspace(0, convert_units(total_time, 'au', 'fs'), nsteps)
    method = ['BO', 'XL-3', 'XL-6', 'XL-9', 'Curvy']

    dists = convert_units(np.array(dists), 'bohr', 'angstrom')
    energies = np.array(energies)
    energies -= energies[0,0]

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11,6), sharex=True)
    for i in range(dists.shape[0]):
        ax[0].plot(time_line, dists[i], label=method[i])
        ax[0].set_ylabel('He--H$^+$ Length ($\\AA$)')
        ax[0].legend()

    for i in range(energies.shape[0]):
        ax[1].plot(time_line, energies[i], label=method[i])
        ax[1].set_xlabel('Time (fs)')
        ax[1].set_ylabel('Energy (a.u.)')
        ax[1].legend()

    plt.tight_layout()
    plt.savefig('dynamics')



if __name__ == '__main__':
    mdtype = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    key = {}
    key['atmsym'] = ['H', 'He']
    key['coordinate'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.929352]]
    key['velocity'] = [[0.0, 0.0, 0.0008], [0.0, 0.0, -0.0008]]

    key['dt'] = 10
    #key['total_time'] = 120
    key['total_time'] = 4000
    key['update_method'] = 'velocity_verlet'

    ed_key = {}
    ed_key['functional'] = 'hf'
    #ed_key['basis'] = '3-21g'
    #ed_key['charge'] = 0
    ed_key['basis'] = 'sto-3g'
    ed_key['charge'] = 1
    ed_key['ortho_method'] = 'lowdin' # 'cholesky'

    if mdtype == 0:
        md = MolecularDynamics(key, ed_key)
        md.run_dynamics()
        md.plot_time_variables(fig_name='normal_time_coords')

    elif mdtype == 1:
        print('run extended_lag')
        key['ed_method'] = 'extended_lag'
        md = MolecularDynamics(key)
        md.run_dynamics()
        md.plot_time_variables(fig_name='extended_lag_time_coords')

    elif mdtype == 2:
        print('run curvy')
        key['ed_method'] = 'curvy'
        key['ortho_method'] = 'cholesky'
        md = MolecularDynamics(key)
        md.run_dynamics()
        md.plot_time_variables(fig_name='curvy_time_coords')

    elif mdtype == 3:
        key['ed_method'] = 'grassmann'
        md = MolecularDynamics(key)
        md.run_dynamics()
        md.plot_time_variables(fig_name='grassmann_time_coords')

    elif mdtype == 4:
        key['update_method'] = 'recursive'
        key['recursive_numbers'] = 4
        md = MolecularDynamics(key, ed_key)
        md.run_dynamics()
        md.plot_time_variables(fig_name='normal_time_coords_recursive')

    elif mdtype == 5:
        dists = []
        energies = []

        md = MolecularDynamics(key)
        md.run_dynamics()

        dists.append( md.md_time_coordinate[:,1,-1] - md.md_time_coordinate[:,0,-1])
        energies.append( md.md_time_total_energy)

        key['ed_method'] = 'extended_lag'
        for xl_nk in [3, 6, 9]:

            key['xl_nk'] = xl_nk
            md = MolecularDynamics(key)
            md.run_dynamics()

            dists.append( md.md_time_coordinate[:,1,-1] - md.md_time_coordinate[:,0,-1])
            energies.append( md.md_time_total_energy)
            energies.append( md.md_time_total_energy2)


#        key['ed_method'] = 'curvy'
#        key['ortho_method'] = 'cholesky'
#        md = MolecularDynamics(key)
#        md.run_dynamics()
#        dists.append( md.md_time_coordinate[:,1,-1] - md.md_time_coordinate[:,0,-1])
#        energies.append( md.md_time_total_energy)

        key['ed_method'] = 'grassmann'
        key['ortho_method'] = 'lowdin'
        md = MolecularDynamics(key)
        md.run_dynamics()
        dists.append( md.md_time_coordinate[:,1,-1] - md.md_time_coordinate[:,0,-1])
        energies.append( md.md_time_total_energy)


        np.savetxt('bond.txt', np.array(dists))
        np.savetxt('energy.txt', np.array(energies))
        plot_time_variables(md.total_time, md.nsteps, dists, energies)

    elif mdtype == 5:
        dists = np.loadtxt('bond.txt')
        energies = np.loadtxt('energy.txt')
        total_time, dt = key['total_time'], key['dt']
        plot_time_variables(total_time, int(total_time/dt) + 1, dists, energies)


