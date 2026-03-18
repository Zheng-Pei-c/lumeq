from lumeq import os, sys, np

"""
websites to check the conversion:
    https://halas.rice.edu/unit-conversions
    https://sherwingroup.itst.ucsb.edu/internal/unit-conversion/
"""

PI2    = 2.*np.pi
FS     = 41.341374575751       # fs to atomic unit time
BOHR   = 0.529177249           # bohr to angstrom AA
H2EV   = 27.21140795           # hartree to ev
EV2J   = 1.602176634*1e-19     # ev=C*V to J= kg (m/s)^2
AVOGADRO = 6.022140857e23      # https://physics.nist.gov/cgi-bin/cuu/Value?na
CAL2J  = 4.184                 # cal to J
E_MASS  = 9.1093837*1e-31      # kg
EV2KJM = EV2J*AVOGADRO*1e-3    # ev to kcal/mol
ATOMIC_MASS = 1e-3/AVOGADRO

C         = 299792458               # speed of light m/s
BOLTZMANN = 1.380649*1e-23          # J/K
PLANCK    = 6.62607015*1e-34        # J/Hz
#PlanckBar = Planck/PI2             # Js
HBAR = PLANCK/(2*3.141592653589793) # https://physics.nist.gov/cgi-bin/cuu/Value?hbar

WN    = EV2J/C/PLANCK*1e-2    # ev to wavenumber cm^-1
EV2NS = PLANCK/EV2J*1e9       # ev to ns ### wikipedia use planckbar!
EV2NM = EV2NS*C               # ev to ns
EV2K  = EV2J/BOLTZMANN        # ev to temperature K
D2KG  = 1.4924180856045*1e-10 # Dalton to kg

BOHR_SI = BOHR * 1e-10
HARTREE2J = HBAR**2/(E_MASS*BOHR_SI**2)
AU2HZ = (HARTREE2J / (ATOMIC_MASS * BOHR_SI**2))**.5 / (2 * np.pi)

# units of these qualities
# milli(m), micro(u), nano(n), pico(p), femto(f), atto(a)
# deci(d), centi(c), angstrom(aa)
units_long = {
        'time':        ['day', 'hour', 'minute', 'second', 'millisecond',
                        'microsecond', 'nanosecond', 'picosecond',
                        'femtosecond', 'atomicunit', 'attosecond'],
        'length':      ['meter', 'decimeter', 'centimeter', 'millimeter',
                        'micrometer', 'nanometer', 'angstrom', 'bohr',
                        'picometer', 'femtometer', 'attometer'],
        'energy':      ['hartree', 'electronvolt', 'milliev', 'kcal/mol', 'kj/mol'],
        'frequency':   ['terahertz', 'cm^-1', 'gigahertz', 'megahertz', 'kilohertz', 'hertz', 'freq_au'],
        'temperature': ['kelvin'],
        'mass':        ['kilogram', 'gram', 'dalton'],
        }

units_short = {
        'time':        ['d', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'au', 'as'],
        'length':      ['m', 'dm', 'cm', 'mm', 'um', 'nm', 'aa', 'b', 'pm', 'fm', 'am'],
        'energy':      ['eh', 'ev', 'mev', 'kcal', 'kj'],
        'frequency':   ['thz', 'cm-1', 'ghz', 'mhz', 'khz', 'hz', 's-1'],
        'temperature': ['k'],
        'mass':        ['kg', 'g', 'u'],
        }
properties = list(units_long.keys())
# indices of atomic units for converting different properties
Idx_name = ['ns', 'nm', 'ev', 'cm-1', 'k', 'kg']
Idx = [0]*len(Idx_name)
for i, s in enumerate(properties):
    Idx[i] = units_short[s].index(Idx_name[i])
#print(Idx)

def default_unit_index(prop):
    return Idx[properties.index(prop)]


units_conversion = {
        'time':        np.array([8.64*1e13, 3.6*1e12, 6.*1e10, 1e9, 1e6, 1e3, 1., 1e-3, 1e-6, 1e-6/FS, 1e-9]),
        'length':      np.array([1e9, 1e8, 1e7, 1e6, 1e3, 1., .1, BOHR/10, 1e-3, 1e-6, 1e-9]),
        'energy':      np.array([H2EV, 1., 1e-3, CAL2J/EV2KJM, 1./EV2KJM]),
        'frequency':   np.array([1e12, C*1e2, 1e9, 1e6, 1e3, 1., AU2HZ]),
        'temperature': np.array([1.]),
        'mass':        np.array([1e3, 1., D2KG]),
        'energy_to_time':           EV2NS,      # ev to ns
        'energy_to_length':         EV2NM,      # ev to nm
        'energy_to_frequency':      WN,         # ev to cm-1
        'energy_to_temperature':    EV2K,       # ev to K
        'energy_to_mass':           EV2J/C**2,  # ev to kg
        }


def find_properties(unit0='au', unit1='fs'):
    unit0, unit1 = unit0.lower(), unit1.lower()

    prop, index = [], []
    for u in [unit0, unit1]:
        for s in properties:
            for name in [units_short, units_long]:
                if u in name[s]:
                    prop.append( s)
                    index.append( name[s].index(u))
    if len(prop) > 2:
        print(prop, index)
        raise ValueError('please specify units with full names:')
    return prop, index


def convert_same_units(value, prop, index):
    conv = units_conversion[prop]
    i, j = index
    return value * conv[i] / conv[j]


def convert_property_to_energy(value, prop, index):
    c1 = convert_same_units(value, prop[0], [index[0], default_unit_index(prop[0])])
    c3 = convert_same_units(1., prop[1], [index[1], default_unit_index(prop[1])])

    c2 = 0
    if 'energy' in prop[0]:
        c2 = units_conversion[prop[0]+'_to_'+prop[1]]
    elif 'time' in prop or 'length' in prop:
        c2 = units_conversion[prop[1]+'_to_'+prop[0]]
    else:
        c2 = 1./units_conversion[prop[1]+'_to_'+prop[0]]

    if 'time' in prop or 'length' in prop:
        return c2/c3/c1
    else:
        return c1*c2/c3


def convert_different_units(value, prop, index):
    if 'energy' in prop:
        return convert_property_to_energy(value, prop, index)
    else:
        k = default_unit_index('energy')
        value = convert_property_to_energy(value, [prop[0], 'energy'], [index[0], k])
        return convert_property_to_energy(value, ['energy', prop[1]], [k, index[1]])


def convert_units(value, unit0='au', unit1='fs'):
    prop, index = find_properties(unit0, unit1)
    if prop[0] == prop[1]:
        return convert_same_units(value, prop[0], index)
    else:
        return convert_different_units(value, prop, index)


def convert_other_property(value, unit0='au'):
    prop, index = find_properties(unit0, None)
    prop, index = prop[0], index[0]
    i = properties.index(prop)

    value1 = np.zeros(len(properties))
    for j, p in enumerate(properties):
        k = Idx[j]
        if j == i:
            value1[j] = convert_same_units(value, prop, [index, k])
        elif j != i:
            value1[j] = convert_different_units(value, [prop, p], [index, k])

        print(value1[j], Idx_name[j], end='  ')
    print('')
    return value1



if __name__ == '__main__':
    value0 = float(sys.argv[1])
    unit0, unit1 = sys.argv[2].lower(), sys.argv[3].lower()
    value1 = convert_units(value0, unit0, unit1)
    print(str(value0)+' '+unit0+' = '+str(value1)+' '+unit1)

    #convert_other_property(value0, unit0)
