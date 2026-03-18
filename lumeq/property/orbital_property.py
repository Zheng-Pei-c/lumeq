from lumeq import os, sys, np
from lumeq.utils import convert_units
from lumeq.utils.pyscf_helper import *
from lumeq.utils.unit_conversion import EV2J, E_MASS

from pyscf import scf, tdscf, gto, lib

def cal_orbital_property(mol, mf):
    orbital_property = {}
    units = {}

    nbas, nocc, nvir, nov = get_basis_info(mol)
    coeff = mf.mo_coeff[:,:nocc]
    energy = mf.mo_energy[:nocc]
    #print_matrix('mo energy occ:', energy[:nocc])
    #print_matrix('mo energy:', energy)

    # these two approaches give same values
    #time = 2.*np.pi / (-energy)
    #time = convert_units(time, 'au', 'fs')
    #print('time:', time)
    time = -convert_units(energy, 'hartree', 'fs') # it is period
    print_matrix('time:', time)
    orbital_property['time'] = time
    units['time'] = 'fs'

    length = mol.intor('int1e_r', comp=3)
    length = np.einsum('mi,xmn,ni->ix', coeff, length, coeff)
    print_matrix('length:', length)
    length = convert_units(length, 'bohr', 'aa')
    length = np.linalg.norm(length, axis=1)
    orbital_property['length'] = length
    units['length'] = 'aa'

    length2 = mol.intor('int1e_rr')
    length2 = np.einsum('mi,xmn,ni->ix', coeff, length2, coeff)
    length2 = convert_units(length2, 'bohr', 'aa')
    orbital_property['length2'] = length2
    units['length2'] = 'aa'

    velocity = 2.*convert_units(length, units['length'], 'nm') / convert_units(time, units['time'], 'ns')
    orbital_property['velocity'] = velocity
    units['velocity'] = 'm/s'
    print_matrix('velocity:', velocity)

    kinetic = mol.intor('int1e_kin')
    kinetic = np.einsum('mi,mn,ni->i', coeff, kinetic, coeff)
    kinetic = convert_units(kinetic, 'hartree', 'ev')
    orbital_property['kinetic'] = kinetic
    units['kinetic'] = 'ev'

    velocity = np.sqrt(2.*kinetic/EV2J/E_MASS)
    orbital_property['velocity2'] = velocity
    units['velocity2'] = 'm/s'

    momentum = -mol.intor('int1e_ipovlp', comp=3)
    momentum = np.einsum('mi,xmn,ni->ix', coeff, momentum, coeff)
    #print('momentum:', momentum)
    orbital_property['velocity3'] = np.linalg.norm(momentum/convert_units(E_MASS, 'kg', 'dalton'), axis=1)
    units['velocity3'] = 'm/s'

    energy = convert_units(energy, 'hartree', 'ev')
    orbital_property['energy'] = energy
    units['energy'] = 'eV'

    orbital_property['units'] = units
    return orbital_property


def cal_property_from_scratch(infile):
    parameters = parser(infile)
    mol, mf = run_pyscf_final(parameters)
    return cal_orbital_property(mol, mf)


if __name__ == '__main__':
    infile = sys.argv[1]
    orbital_property = cal_property_from_scratch(infile)
