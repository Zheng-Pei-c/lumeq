from pathlib import Path

from lumeq import np, itertools
from lumeq.utils import read_number, read_array, print_matrix
from lumeq.dynamics.dimers_in_crystal import read_unit_cell_info, add_molecules_cell, add_molecule

def read_energy_coupling(outfile, nstate=2):
    r"""
    Read exciton energy and Coulomb coupling from the output file.

    Args:
        outfile (str): The q-chem output file name
        nstate (int): The number of exciton states per site

    Returns:
        tuple: Exciton energies with shape ``(nstate,)``, exciton-exciton
        coupling matrix with shape ``(nstate, nstate)``, and transition dipoles
        with shape ``(2, nstate, 3)``.
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

    Args:
        cif_file (str): The cif file name
        n_cell (list): The number of unit cells in each direction ``[nx, ny, nz]``.
        outfile_dir (str): The output file directory
        nstate (int): The number of exciton states per site
        npairs (int): The number of neighboring pairs to be included
        center (int): The center site index needed to shift (0 for B, 1 for A)
        debug (int): The debug level

    Returns:
        unit_cell (dict): The unit cell information
        energy (numpy.ndarray): Exciton energies of each site in eV.
        coupling_j (numpy.ndarray): Exciton-exciton couplings between dimers in eV.
        dipole (numpy.ndarray): Transition dipole moments of two sites in au.
        neighbor_index (list): Neighbor-index list for each dimer.
        distances (numpy.ndarray): Distances between dimers in Angstrom.
        center_coords (numpy.ndarray): Center-of-mass coordinates of the sites in Angstrom.
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
              r_cutoff=10, periodic=False, debug=0):
    r"""Select finite model cells or a cutoff-reachable periodic cell envelope.

    Args:
        neighbor_index: Distance-sorted (source molecule, target cell/site) rows.
        distances: Dimer distances in Angstrom, in the same order.
        model: AB, BC, ABAC/BACA, AB-C, BC-A, ABC, or any.
        n_cell: Finite dimensions. In periodic mode, entries greater than one
            select extended directions; their magnitude does not expand cells.
        r_cutoff: Coupling distance cutoff in Angstrom.
        periodic: Construct a motif and only its reachable images instead of
            expanding the entire finite chain, plane, or volume.
        debug: Debug level.

    Returns:
        cells, neighbor_index: Selected crystal cells and cutoff-filtered rows.
    """
    neighbor_index = neighbor_index[:np.sum(distances <= r_cutoff)]

    n_cell = [n_cell] if isinstance(n_cell, int) else list(n_cell)
    n_cell += [1] * (3-len(n_cell))
    ndim = 3 - np.count_nonzero(np.array(n_cell) == 1)
    if ndim == 0:
        raise ValueError('At least one dimension should be greater than 1.')

    if model == 'BACA':
        model = 'ABAC'
    if model not in {'AB', 'BC', 'ABAC', 'AB-C', 'BC-A', 'ABC', 'any'}:
        print('Unknown model type %s changed to any.' % model)
        model = 'any'

    origin = np.zeros(3, dtype=int)
    if model == 'ABC':
        # ABC already denotes an xy plane for a nominal 1D request.
        dimension = max(2, ndim)
        translations = np.eye(3, dtype=int)[:dimension]
        seeds = [origin] if dimension == 3 else [origin, np.array([0, 0, 1])]
    else:
        if len(neighbor_index) < 3:
            raise ValueError('Model geometry requires the A, B and C neighbor definitions.')

        # Integer crystal-cell displacements of the nearest A, B and C bonds.
        va, vb, vc = [np.array(entry[1][:3], dtype=int) for entry in neighbor_index[:3]]
        # Offset between the two initial columns of the molecular path.
        start = vb if model in {'BC', 'BC-A'} else va
        steps = [vb, vc] if model in {'ABAC', 'any'} else [vc if model in {'BC', 'BC-A'} else vb]
        # Net displacement of one repeat: AB -> vb-va, BC -> vc-vb,
        # ABAC -> vb+vc-2*va. Each forward step is followed by -start.
        chain_translation = sum((step-start for step in steps), origin.copy())

        # Candidate crystal cells within one repeat, in both columns.
        seeds = [origin]
        for step in steps[:-1]:
            seeds.append(seeds[-1]+step-start)
        seeds += [cell+start for cell in seeds[:]]

        # Second periodic direction, translating from one chain to the next:
        # AB-C uses BC, BC-A uses BA; other paths use the column offset.
        transverse = vc-vb if model == 'AB-C' else vb-va if model == 'BC-A' else start

        # Preserve the finite model's existing layer expansion direction.
        if model == 'AB':
            finite_normal = np.abs(np.cross(va, vb))
        elif model == 'BC':
            finite_normal = np.abs(np.cross(vb, vc))
        else:
            finite_normal = np.abs(np.cross(np.cross(vb, vc), va))

        if model == 'any' and periodic:
            # Infinite completion of the Cartesian bounding box of the path.
            growing = [chain_translation]
            if ndim >= 2:
                growing.append(start)
            if ndim == 3:
                growing.append(finite_normal)
            axes = np.flatnonzero(np.any(np.array(growing) != 0, axis=0))
            if not len(axes):
                raise ValueError('The periodic bounding box has no growing directions.')
            translations = np.eye(3, dtype=int)[axes]
            bounds = np.array(seeds)
            ranges = [range(1) if i in axes else range(int(bounds[:, i].min()), int(bounds[:, i].max())+1)
                      for i in range(3)]
            seeds = [np.array(cell) for cell in itertools.product(*ranges)]
        else:
            # Rows are repeat translations in the original crystal basis.
            translations = [chain_translation]
            if ndim >= 2:
                translations.append(transverse)
            if ndim == 3 and periodic:
                # Third direction normal to the periodic plane. Reduce to
                # an integer primitive vector while retaining its signs.
                normal = np.cross(translations[0], translations[1])
                divisor = np.gcd.reduce(np.abs(normal))
                if divisor:
                    normal //= divisor
                translations.append(normal)
            translations = np.array(translations, dtype=int)

    if periodic:
        if np.linalg.matrix_rank(translations) != len(translations):
            raise ValueError('The chosen periodic model has dependent translation generators.')

        # Convert a crystal-cell displacement into repeat coordinates.
        # Integer coefficients AND zero transverse residual imply equivalence.
        inverse = np.linalg.pinv(translations)
        # Retain one representative of each translation-equivalent seed.
        motif = []
        for seed in seeds:
            for other in motif:
                coefficients = (seed-other) @ inverse
                shift = np.rint(coefficients).astype(int)
                if (np.allclose(coefficients, shift, atol=1e-8, rtol=0)
                        and np.allclose(shift @ translations, seed-other, atol=1e-8, rtol=0)):
                    break
            else:
                motif.append(seed)
        motif = np.array(motif, dtype=int)

        # Only the image cells touched by retained bonds are needed.
        images = set()
        for cell in motif:
            for _, entry in neighbor_index:
                target = cell + entry[:3]
                for other in motif:
                    coefficients = (target-other) @ inverse
                    shift = np.rint(coefficients).astype(int)
                    if not (np.allclose(coefficients, shift, atol=1e-8, rtol=0)
                            and np.allclose(shift @ translations, target-other, atol=1e-8, rtol=0)):
                        continue
                    # Include both the forward link and its Hermitian reverse.
                    images.add(tuple(target))
                    images.add(tuple(cell-shift @ translations))
                    break  # later cell indices cannot provide another representative

        if not images:
            raise ValueError('No active bonds remain in the periodic model at this cutoff.')

        central = [tuple(c) for c in motif]
        cells = np.array(central + sorted(images-set(central)), dtype=int)
        return cells, neighbor_index  # never expand a long finite model

    nx, ny, nz = np.sort(n_cell)[::-1]
    if ndim == 1:
        ny = 2
    if ndim >= 2 and ny == 2:
        ny += 1
    if model == 'ABC':
        if ndim == 1:
            ny = nx
        return np.array(list(itertools.product(range(nx), range(ny), range(nz+1)))), neighbor_index

    cells = [[origin], [start]]
    # In 1D the requested chain length is shared by the two columns.
    nmax = (nx-1)//2 if ndim == 1 else nx-1
    for _ in range(nmax//len(steps)):
        for step in steps:
            cells[1].append(cells[0][-1]+step)
            cells[0].append(cells[1][-1]-start)

    cells = np.array(cells)

    if model == 'any':
        flat = cells.reshape(-1, 3)
        low, high = flat.min(axis=0), flat.max(axis=0)
        cells = list(itertools.product(*(range(a, b+1) for a, b in zip(low, high))))
    cells = np.reshape(cells, (2, -1, 3))

    if ndim >= 2 and model in {'AB-C', 'BC-A'}:
        # Keep legacy finite BC-A behavior; its existing branch used start.
        offset = start if model == 'BC-A' else transverse
        shifted = cells.copy()
        for _ in range(ny-1):
            shifted += offset[None, None, :]
            cells = np.append(cells, shifted, axis=0)
    elif ndim >= 2:
        shifted = cells[1].copy()
        for _ in range(ny-2):
            shifted += start[None, :]
            cells = np.append(cells, [shifted], axis=0)

    if ndim >= 3:
        shifted = cells.copy()
        for _ in range(nz-1):
            shifted += finite_normal[None, None, :]
            cells = np.append(cells, shifted, axis=0)

    return cells.reshape(-1, 3), neighbor_index
