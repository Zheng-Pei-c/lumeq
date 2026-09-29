import os, sys
import numpy as np
from numpy import cos, sin, arccos

ELECTRON_MASS_IN_AMU = 5.4857990945e-04

# based on https://www.ucl.ac.uk/~rmhajc0/frorth.pdf
def read_unit_cell_info(ciff):
    if ciff[-4:] != '.cif':
        raise ValueError('read only cif file right now.')

    RAD = 180.0/np.pi

    elements = []
    scales = []
    with open(ciff, 'r') as cif_file:
        for line in cif_file:
            cols = line.split()
            ncols = len(cols)
            if ncols == 2 and cols[0] == '_cell_length_a':
                a = float(cols[1].split('(')[0])
            elif ncols == 2 and cols[0] == '_cell_length_b':
                b = float(cols[1].split('(')[0])
            elif ncols == 2 and cols[0] == '_cell_length_c':
                c = float(cols[1].split('(')[0])
            if ncols == 2 and cols[0] == '_cell_angle_alpha':
                alpha = float(cols[1].split('(')[0])/RAD
            elif ncols == 2 and cols[0] == '_cell_angle_beta':
                beta = float(cols[1].split('(')[0])/RAD
            elif ncols == 2 and cols[0] == '_cell_angle_gamma':
                gamma = float(cols[1].split('(')[0])/RAD
            elif ncols == 2 and cols[0] == '_cell_formula_units_Z':
                n_mol = int(cols[1])
            if ncols > 10 and cols[6][0:1] == 'U':
                #print("ncols:", ncols, "line:", line)
                elements.append(cols[1])
                scales.append(float(cols[2].split('(')[0]))
                scales.append(float(cols[3].split('(')[0]))
                scales.append(float(cols[4].split('(')[0]))

    scales = np.array(scales).reshape((-1,3))

    #Eq. 7, alpha_star is the angle between y* and z* in the reciprocal space
    alpha_star = arccos((cos(beta)*cos(gamma)-cos(alpha))/sin(beta)/sin(gamma))

    return [a,b,c], [alpha,beta,gamma,alpha_star], elements, scales, n_mol


def add_molecule(ix, iy, iz, inverse, abc, angles, elements, scales):
    #Eq. 17
    a, b, c = abc[0], abc[1], abc[2]
    alpha, beta, gamma, alpha_star = angles[0], angles[1], angles[2], angles[3]

    x, y, z = scales.T
    if inverse == 1:
        x, y, z = 1.-x, 1.-y, 1.-z

    coordinates = np.zeros_like(scales)
    coordinates[:,0] = (ix+x) * a + (iy+y) * b * cos(gamma) + (iz+z) * c * cos(beta)
    coordinates[:,1] =              (iy+y) * b * sin(gamma) - (iz+z) * c * sin(beta) * cos(alpha_star)
    coordinates[:,2] =                                        (iz+z) * c * sin(beta) * sin(alpha_star)

    return coordinates


def translate_molecule(ix, iy, iz, abc, angles, elements, coords0):
    #Eq. 17
    a, b, c = abc[0], abc[1], abc[2]
    alpha, beta, gamma, alpha_star = angles[0], angles[1], angles[2], angles[3]

    coordinates = np.copy(coords0)
    coordinates[:,0] += ix * a + iy * b * cos(gamma) + iz * c * cos(beta)
    coordinates[:,1] +=          iy * b * sin(gamma) - iz * c * sin(beta) * cos(alpha_star)
    coordinates[:,2] +=                                iz * c * sin(beta) * sin(alpha_star)

    return coordinates


def add_molecules_cell(n_images, abc, angles, elements, scales):
    ix_min = -int((n_images[0]-1)/2)
    iy_min = -int((n_images[1]-1)/2)
    iz_min = -int((n_images[2]-1)/2)
    n_total = 2 * np.prod(n_images)

    natoms = len(elements)

    elements_all = elements * n_total
    coordinates = np.zeros((n_total*natoms, 3))
    centers_all = np.zeros((n_total, 3))

    site_label = {}
    icount = 0
    for ix in range(ix_min, ix_min+n_images[0]):
        for iy in range(iy_min, iy_min+n_images[1]):
            for iz in range(iz_min, iz_min+n_images[2]):
                for inverse in range(2): # number of molecules in a unit cell

                    coordinates[icount*natoms:(icount+1)*natoms] = add_molecule(ix, iy, iz, inverse, abc, angles, elements, scales)
                    centers_all[icount] = .5 * (coordinates[icount*natoms+4] + coordinates[icount*natoms+8])
                    site_label[icount] = str(ix)+','+str(iy)+','+str(iz)+','+str(inverse)
                    icount += 1
    #print('site_label:', site_label)
    return elements_all, coordinates, centers_all, site_label


def write_xyz_files(elements, coordinates, mol):
    natoms = len(elements)

    with open(mol+'.xyz', 'w') as xyzf:
        xyzf.write('%3d \n\n' %(natoms))
        for i in range(natoms):
            xyzf.write('%1s %12.7f %12.7f %12.7f\n' %(elements[i], coordinates[i,0], coordinates[i,1], coordinates[i,2]))


def write_eda_files(elements, coordinates, mol):
    natoms = len(elements)

    with open(mol+'-dc.inp', 'w') as xyzf:
        xyzf.write('$molecule\n0 1\n--\n0 1\n')
        for i in range(natoms):
            xyzf.write('%1s %12.7f %12.7f %12.7f\n' %(elements[i], coordinates[i,0], coordinates[i,1], coordinates[i,2]))
            if i == natoms/2-1: xyzf.write('--\n0 1\n')
        xyzf.write('$end')


def dipole_dipole_interaction(d1, d2, r1, r2):
    r"""
    d1, d2: dipoles (x,y,z)
    r1, r2: dipole center coordinates (x, y, z)
    """
    dr = r2 - r1
    d1dr, d2dr = np.dot(d1, dr), np.dot(d2, dr)

    dr = np.linalg.norm(dr)
    r3, r5 = dr**3, dr**5

    return np.dot(d1, d2) / r3 - 3. * d1dr * d2dr / r5


def cal_dipole_interactions(dipoles, coords):
    ndip = dipoles.shape[0]

    dd_coupling = np.zeros((ndip, ndip))
    for i in range(ndip):
        for j in range(i):
            dd_coupling[i,j] = dd_coupling[j,i] = dipole_dipole_interaction(dipoles[i], dipoles[j], coords[i], coords[j])

    return dd_coupling


def get_center_of_mass(atmsym, coords):
    natoms = len(atmsym)
    mass = np.zeros(natoms)
    # use pyscf's
    from pyscf.data import elements
    for i in range(natoms):
        mass[i] = elements.MASSES[elements.charge(atmsym[i])] / ELECTRON_MASS_IN_AMU
    mass_center = np.einsum('z,zx->x', mass, coords) / mass.sum()
    return mass_center



class Crystal:
    def __init__(self, cif_file):
        self.mol = cif_file.replace('.cif', '')

        abc, angles, elements, scales, n_mol = read_unit_cell_info(cif_file)
        self.abc = abc
        self.angles = angles
        self.elements = elements
        self.scales = scales
        self.n_mol = n_mol
        if n_mol != 2:
            raise ValueError('Only support 2 molecules in a unit cell for now.')


    def add_molecules_cell(self, n_images=[5, 5, 5]):
        self.n_images = n_images
        self.n_total = self.n_mol * np.prod(n_images)

        self.elements_all, self.coordinates, self.centers_all, self.site_label = add_molecules_cell(
            self.n_images, self.abc, self.angles, self.elements, self.scales
        )
        return self.elements_all, self.coordinates, self.centers_all, self.site_label


    def get_distance_order(self, center=0, npairs=50, debug=0):
        centers_all = self.centers_all
        site_label = self.site_label
        n_mol = self.n_mol
        n_cell = self.n_images

        distances = []
        n_total = n_mol * np.prod(n_cell)
        i = int(n_total//n_mol) # center site (B)
        i -= center # shift if needed (A if center=1)
        self.center = i
        if debug:
            print('center site:', i, site_label[i])
        for j in range(n_total):
            distances.append(np.linalg.norm(centers_all[i]-centers_all[j]))
        distances = np.array(distances)
        order = distances.argsort()
        if npairs is None: npairs = len(order) - 1
        order = order[:npairs+1]

        self.distances = distances
        self.order = order
        return order, distances


    def get_neighbor_index(self, debug=False):
        order = self.order
        distances = self.distances
        site_label = self.site_label

        # given neighboring pairs
        index = [list(map(int, site_label[order[k]].split(','))) for k in range(len(order))]
        sort = [sort_index(d[0], d[1], d[2], d[3], n_cell[0], n_cell[1], n_cell[2], 2) for d in index]
        if debug:
            print('index:')
            for k, idx in enumerate(index):
                print('%3d ' % (order[k]+1), end='')
                for d in idx:
                    print('%2d ' %d, end='')
                print(' %10.5f' % distances[order[k]])

        # get neighbor index list
        neighbor_index = []
        for j, idx in enumerate(index[1:], start=1):
            k = index[0][3] # B molecule as center
            if sort[j] < sort[0]: # A molecule as center
                k = abs(1-k)
                a, b, c, d = idx
                # reverse the coupling direction
                # checked by index from using center (i -= 1)
                idx = [-a, -b, -c, abs(1-d)]

            neighbor_index.append([k, idx])

        self.neighbor_index = neighbor_index
        return neighbor_index


    def get_hamil_parameters(self, outfile_dir='./', mol=None):
        if mol is None: mol = self.mol
        order = self.order
        distances = self.distances

        # read energy, dipoles, and couplings from output files
        energy, coupling, trans_dipole = [], [], []
        for k in order[1:]:
            outfile = outfile_dir+mol+'-'+str(i+1)+'-'+str(k+1)+'-dimer'+'_%4.2f-dc.out' % distances[k]
            if Path(outfile).is_file():
                e, c, d = read_energy_coupling(outfile)
                #energy.append(e)
                coupling.append(c)
                energy = e[0]
                trans_dipole = d[0]

        energy, coupling = np.array(energy), np.array(coupling)
        return energy, coupling, trans_dipole


    def set_model(self, model='AB', n_cell=[10,1,1], r_cutoff=10, debug=0):
        neighbor_index = self.neighbor_index
        distances = self.distances

        return set_model(neighbor_index, distances, model, n_cell, r_cutoff,
                         debug)



if __name__ == '__main__':
    mol = 'H2OBPc'
    crystal = Crystal(mol+'.cif')
    print('abc:', crystal.abc, 'angles:', crystal.angles)
    natoms = len(crystal.elements)
    print('natoms:', natoms)

    npairs = 151
    n_images = [5, 5, 5]
    elements_all, coordinates, centers_all, site_label = crystal.add_molecules_cell(n_images)
    print('n_total:', crystal.n_total)

    order, distances = crystal.get_distance_order(center=0, npairs=npairs)
    print('site_label:')
    for k in range(npairs+1):
        print('%3d: %10s %12.5f' % (order[k]+1, site_label[order[k]], distances[order[k]]))

    i = crystal.center
    print('center of mass:', get_center_of_mass(elements_all[:natoms], coordinates[i*natoms:(i+1)*natoms]))

    fname = mol+'-'+str(n_images[0])+'-'+str(n_images[1])+'-'+str(n_images[2])
    write_xyz_files(elements_all, coordinates, fname)

    for k in order[1:npairs+1]:
        fname = mol+'-'+str(i+1)+'-'+str(k+1)+'-dimer'
        coordinates_dimer = np.zeros((2*natoms, 3))
        coordinates_dimer[:natoms]         = np.copy(coordinates[i*natoms:(i+1)*natoms])
        coordinates_dimer[natoms:2*natoms] = np.copy(coordinates[k*natoms:(k+1)*natoms])
        write_xyz_files(elements_all[:2*natoms], coordinates_dimer, fname)
        write_eda_files(elements_all[:2*natoms], coordinates_dimer, fname)
