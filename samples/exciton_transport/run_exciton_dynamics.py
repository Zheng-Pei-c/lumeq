import os, sys
import numpy as np

from lumeq.utils.read_files import read_number, read_matrix
from lumeq.dynamics.exciton_dynamics import *

if __name__ == '__main__':
    total_time = 400
    key = {}

    """
    three dimensional exciton dynamics
    """
    para_file = 'H2OBPc.exciton.txt'
    n_mode = 6
    key['n_mode'] = n_mode
    key['nuclear_mass']  = [6., 6., 754., 754., 754., 754.] # amu
    key['nuclear_omega'] = [144., 148., 5., 5., 5., 5.] # meV

    n_site   = np.array([11, 1, 1]) # how many unit cells
    n_mol    = 2 # how many molecules in a unit cell
    coords   = read_matrix(para_file, 'center of mass', 1, 3)
    distance = read_number(para_file, '_cell_length', -1, dtype=float)
    angle    = read_number(para_file, '_cell_angle', -1, dtype=float)
    nstate   = 2
    key['n_site']   = n_site
    key['n_mol']    = n_mol
    key['coords']   = coords
    key['distance'] = distance
    key['angle']    = angle
    key['nstate']   = nstate

    key['energy'] = [0., 10.] # meV

    coupling_g = np.zeros((n_mode, nstate))
    coupling_g[0,0] = 1821. # meV/AA
    coupling_g[1,1] = 2231. # meV/AA
    key['coupling_g'] = coupling_g

    dimer_label = read_number(para_file, 'dimer_label', -1, dtype=str)
    key['dimer_label'] = dimer_label

    # x, y, z axis.
    # (1,1,1) is center O, (0,1,1) and (2,1,1) is the left and right points on x-axis
    # coupling_j in (npairs, nstate, nstate) # meV
    coupling_j = read_number(para_file, 'coupling_j', -1, dtype=float)
    key['coupling_j'] = coupling_j.reshape(-1, nstate, nstate)

    # coupling_a in (npairs, n_mode, nstate, nstate) # meV/AA
    coupling_a = read_number(para_file, 'coupling_a', -1, dtype=float)
    key['coupling_a'] = coupling_a.reshape(-1, n_mode, nstate, nstate)

    obj = Dynamics(key, total_time=total_time)
    obj.kernel()
    obj.plot_time_variables()
