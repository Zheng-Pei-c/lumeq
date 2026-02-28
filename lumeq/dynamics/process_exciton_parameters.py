from pathlib import Path

from lumeq import np, itertools
from lumeq.utils import read_number, read_array, print_matrix
from lumeq.dynamics.dimers_in_crystal import read_unit_cell_info, add_molecules_cell, add_molecule

def read_energy_coupling(outfile, nstate=2):
    r"""
    Read exciton energy and Coulomb coupling from the output file.

    Parameters
        outfile : str
            The q-chem output file name
        nstate : int
            The number of exciton states per site

    Returns
        energy : (nstate,) array
            Exciton energy of each site in eV
        coupling : (nstate, nstate) array
            Exciton-exciton coupling between two sites in
        trans_dipole : (2, nstate, 3) array
            Transition dipole moments of two sites in au
    """
    energy = read_number(outfile, 'excitation energy', n=1, o=7, dtype=float)
    energy = np.array(energy).reshape(2,-1)[:, :nstate]

    trans_dipole = []
    for x in range(3):
        transx = read_number(outfile, 'Trans. Mom.:', n=1, o=2*(x+1), dtype=float)
        trans_dipole.append(transx)
    trans_dipole = np.array(trans_dipole).reshape(3,2,-1).transpose(1,2,0)[:,:nstate,:]

    # state numbers to be coupled
    sts_trans = read_number(outfile, 'sts_trans', n=1, o=1, dtype=str)
    sts_trans = [int(s.split('-')[-1]) for s in sts_trans]
    coupling = read_array(outfile, 'Coulomb (eV)    Exchange (eV)  Interaction (eV)',
                          nline=np.prod(sts_trans), ncol=6, nrange=[3,4], same=False)
    coupling = np.array(coupling).reshape(sts_trans)[:nstate,:nstate]

    for i, di in enumerate(trans_dipole[0]):
        signi = np.sign(di[np.argmax(np.abs(di))])
        trans_dipole[0][i] *= signi
        for j, dj in enumerate(trans_dipole[1]):
            signj = np.sign(dj[np.argmax(np.abs(dj))])
            trans_dipole[1][j] *= signj
            coupling[i,j] *= signi*signj
    #print_matrix('coupling:', coupling)

    return energy, coupling, trans_dipole


def sort_index(i, j, k, l, nx, ny, nz, ns):
    return i*ny*nz*ns + j*nz*ns + k*ns + l


def process_parameters(cif_file, n_cell, outfile_dir='./', nstate=2, npairs=50,
                       center=0, debug=0):
    r"""
    Process the input parameters and get real parameters:
    on-site energies, transition dipoles, state couplings for dimers.

    Parameters
        cif_file : str
            The cif file name
        n_cell : list of int
            The number of unit cells in each direction [nx, ny, nz]
        outfile_dir : str
            The output file directory
        nstate : int
            The number of exciton states per site
        npairs : int
            The number of neighboring pairs to be included
        center : int
            The center site index needed to shift (0 for B, 1 for A)
        debug : int
            The debug level

    Returns
        unit_cell : dict
            The unit cell information
        energy : (nstate,) array
            Exciton energy of each site in eV
        coupling_j : (ndimer, nstate, nstate) array
            Exciton-exciton couplings between dimers in eV
        dipole : (2, nstate, 3) array
            Transition dipole moments of two sites in au
        neighbor_index : list of [int, list of int]
            The neighbor index list for each dimer
        distances : (ndimer,) array
            The distances between dimers in Angstrom
        center_coords : (2, 3) array
            The center of mass coordinates of the sites in Angstrom
    """
    cif_file = outfile_dir + cif_file
    mol = cif_file.replace('.cif', '')

    # find out the unit cell information
    abc, angles, elements, scales, n_mol = read_unit_cell_info(cif_file)
    unit_cell = {'abc': abc, 'angles': angles, 'elements': elements,
                 'scales': scales, 'n_mol': n_mol}

    elements_all, coordinates, centers_all, site_label = add_molecules_cell(n_cell, abc, angles, elements, scales)

    distances = []
    n_total = n_mol * np.prod(n_cell)
    i = int(n_total//2) # center site (B)
    i -= center # shift if needed (A if center=1)
    if debug:
        print('i:', i, site_label[i])
    for j in range(n_total):
        distances.append(np.linalg.norm(centers_all[i]-centers_all[j]))
    distances = np.array(distances)
    order = distances.argsort()
    if npairs is None: npairs = len(order) - 1
    order = order[:npairs+1]

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


    # read energy, dipoles, and couplings from output files
    energy, coupling, trans_dipole = [], [], []
    for k in order[1:npairs+1]:
        outfile = outfile_dir+mol+'-'+str(i+1)+'-'+str(k+1)+'-dimer'+'_%4.2f-dc.out' % distances[k]
        if Path(outfile).is_file():
            e, c, d = read_energy_coupling(outfile)
            #energy.append(e)
            coupling.append(c)
            energy = e[0]
            trans_dipole = d[0]

    energy, coupling = np.array(energy), np.array(coupling)

    i = int(n_total//2) - center
    keys = {'unit_cell': unit_cell,
            'energy': energy,
            'coupling_j': coupling,
            'dipole': trans_dipole,
            'neighbor_index': neighbor_index,
            'distances': distances[order[1:npairs+1]],
            'center_coords': centers_all[i:i+2]
            }
    return keys


def set_model(neighbor_index, distances, model='AB', n_cell=[10,1,1],
              r_cutoff=10, debug=0):
    r"""
    Pick the model parameters based on the model type.

    Parameters
        neighbor_index : list of [int, list of int]
            The neighbor index list for each dimer
        distances : (ndimer,) array
            The distances between dimers in Angstrom
        model : str
            The model type, e.g., 'AB', 'BC', 'ABAC', 'AB-C', 'any'
        n_cell : list of int
            The number of unit cells in each direction [nx, ny, nz]
        r_cutoff : float
            The cutoff distance in Angstrom (10, 17, 26)
        debug : int
            The debug level

    Returns
        cells : (ncell, 3) array
            The cell indices based on the model
        neighbor_index : list of [int, list of int]
            Refined neighbobr index list based on the model.
    """
    # apply distance cutoff
    neighbor_index = neighbor_index[:np.sum(distances <= r_cutoff)]
    #print('neighbor count after cutoff:', len(neighbor_index))
    #print(neighbor_index)


    # figure out dimensions
    if isinstance(n_cell, int): n_cell = [n_cell]
    for i in range(3-len(n_cell)):
        n_cell.append(1)
    #print('n_cell:', n_cell)
    ndim = 3 - np.sum(np.array(n_cell) == 1) # need to change to np array
    if ndim == 0:
        raise ValueError('At least one dimension should be greater than 1.')
    #print('ndim:', ndim)

    # chain length, number of chains, number of layers
    nx, ny, nz = np.sort(n_cell)[::-1]
    #print('nx, ny, nz:', nx, ny, nz)

    if ndim == 1: ny = 2 # quasi-1D system has two columns
    if ndim >= 2 and ny == 2: ny += 1 # add more columns for 2D
    if ndim >= 3 and nz == 1: nz += 1 # add more layers for 3D


    # now get sepecific cells for different models
    if model == 'ABC':
        # 1D ABC is already a plane whose dimension is given by nx and ny
        # nz adds more layers
        if ndim == 1: ny = nx # get a square by default
        cells = list(itertools.product(range(nx), range(ny), range(nz+1)))
        return np.array(cells), neighbor_index

    # interchangable model types
    if model == 'BACA':
        model = 'ABAC'
    if model not in {'AB', 'BC', 'ABAC', 'AB-C', 'BC-A', 'any'}:
        model = 'any'
        print('Unknown model type %s changed to any.' % model)


    # the first three neighboring dimers labeled as A, B, and C
    vectors = [np.array(neighbor_index[i][1][:3], dtype=int) for i in range(3)]

    # get the perpendicular vector for 3D expansion
    if model == 'AB':
        vector_perp = np.cross(vectors[0], vectors[1])
    elif model == 'BC':
        vector_perp = np.cross(vectors[1], vectors[2])
    elif model in {'ABAC', 'AB-C', 'BC-A', 'any'}:
        vector_perp = np.cross(vectors[1], vectors[2])
        vector_perp = np.cross(vector_perp, vectors[0])
    vector_perp = np.abs(vector_perp)
    #print('vector_perp:', vector_perp)


    # get effective 1D chain cells
    c = 1 if model in {'BC', 'BC-A'} else 0
    vec = vectors[c] # A or B as the starting of second column
    cells = [[[0,0,0]], [vec]] # O as the starting of first column

    nmax = nx - 1
    if ndim == 1: nmax = nmax // 2
    if model in {'AB', 'BC', 'AB-C', 'BC-A'}:
        for i in range(nmax):
            cells[1].append(cells[0][-1] + vectors[c+1])
            cells[0].append(cells[1][-1] - vectors[c])
    elif model in {'ABAC', 'any'}:
        for i in range(nmax//2):
            cells[1].append(cells[0][-1] + vectors[1])
            cells[0].append(cells[1][-1] - vectors[0])
            cells[1].append(cells[0][-1] + vectors[2])
            cells[0].append(cells[1][-1] - vectors[0])
    cells = np.array(cells)

    # shift to positive indices
    #cells = cells.reshape(-1, 3).T
    #for i, _cells in enumerate(cells):
    #    min_id = np.min(_cells)
    #    if min_id < 0:
    #        cells[i] -= min_id
    #cells = cells.T

    if model == 'any': # include unconnected cells
        cells = cells.reshape(-1, 3)
        ci, cj = np.min(cells, axis=0), np.max(cells, axis=0)
        rx, ry, rz = range(ci[0], cj[0]+1), range(ci[1], cj[1]+1), range(ci[2], cj[2]+1)
        cells = list(itertools.product(rx, ry, rz))

    cells = np.reshape(cells, (2, -1, 3))


    # expand to higher dimensions
    if ndim >= 2 and model in {'AB-C', 'BC-A'}:
        if model == 'AB-C':
            vec = vectors[2] - vectors[1] # BC vector
        elif model == 'BC-A':
            vecc = vectors[1] - vectors[0] # BA vector
        cells_t = np.copy(cells)
        for i in range(ny-1):
            cells_t += vec[None, None, :]
            cells = np.append(cells, cells_t, axis=0)

    elif ndim >= 2:
        cells_t = np.copy(cells[1])
        for i in range(ny-2):
            cells_t += vec[None, :]
            cells = np.append(cells, [cells_t], axis=0)

    if ndim >= 3:
        cells_t = np.copy(cells)
        for i in range(nz-1):
            cells_t += vector_perp[None, None, :]
            cells = np.append(cells, cells_t, axis=0)

    cells = cells.reshape(-1, 3)
    #print('number of cells:', len(cells))
    #print(cells)

    return cells, neighbor_index
