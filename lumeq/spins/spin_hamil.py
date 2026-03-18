from lumeq import np
from lumeq.spins import qt # qutip

from lumeq.utils import print_matrix
from lumeq.plot import plt

# use ladder operators if possible
# because sigma_y is complex matrix
_spin_names = ['0', 'x', 'y', 'z', '+', '-']

def get_spin(x, j=.5):
    r"""
    spin matrices from qutip package
    """
    # x takes 0, x, y, z, +, -
    if x not in _spin_names:
        raise ValueError('spin matrix variable should be one of {0, x, y, z, +, -}!')

    if x=='0':
        return qt.qeye(int(2*j+1)) # alas to identity()
    else:
        # sigmax(/y/z) function is specific for spin-1/2
        # but gives Pauli matrices, i.e. without .5 factor
        #getattr(qt, 'sigma'+x)()
        return qt.jmat(j, x)


def get_spin_mat(x, j=.5):
    r"""
    reload get_spin() but return numpy matrix by qutip full() function
    """
    # x takes 0, x, y, z, +, -
    if x not in _spin_names:
        raise ValueError('spin matrix variable should be one of {0, x, y, z, +, -}!')

    if x=='0':
        return qt.qeye(int(2*j+1)).full() # alas to identity()
    else:
        return qt.jmat(j, x).full()


def get_spins(xs='all', j=.5, np_matrix=True):
    if type(xs) is str:
        if xs == 'all':
            xs = _spin_names
        else:
            xs = xs.strip()

    sigma = []

    if np_matrix:
        for x in xs:
            sigma.append(get_spin_mat(x, j))
    else:
        for x in xs:
            sigma.append(get_spin(x, j))

    return sigma


def get_prod_spin_list(n, xs='all', j=.5, np_matrix=True):
    r"""Build spin operators acting on each site in the full Hilbert space.

    Notes:
        This constructs the actual ``sigma_{i,x}`` operators by looping over
        the ``n`` spins, placing the requested spin matrix on site ``i`` and
        identity operators on all other sites. The returned ``n * d`` matrices
        form Hilbert-space operators or a basis in the full product space.
    """
    sigma = get_spins(xs, j, np_matrix=False) # use qutip type here

    spin_list = [None] * n
    for i in range(n):
        op_list = [qt.qeye(int(2*j+1))] * n
        _list = []
        for s in sigma:
            op_list[i] = s
            _list.append(qt.tensor(op_list))
        spin_list[i] = _list

    if np_matrix: # get numpy matrices
        # naively loopover
        for i, spins in enumerate(spin_list):
            for s, spin in enumerate(spins):
                spin_list[i][s] = spin.full()
        spin_list = np.array(spin_list)

    return spin_list


def hamil_heisenberg_1d(n, j, hz, np_matrix=True, spin_j=.5, boundary='open'):
    r"""Build the 1D Heisenberg Hamiltonian in the full Hilbert space.

    Notes:
        The Hamiltonian is

        ``H = sum_i (j_{i,x} sigma_{i,x} sigma_{i+1,x}
        + j_{i,y} sigma_{i,y} sigma_{i+1,y}
        + j_{i,z} sigma_{i,z} sigma_{i+1,z})
        + sum_i h_{z,i} sigma_{i,z}``.

        When ``j_x = j_y``, the transverse part can be rewritten as
        ``sigma_x sigma_x + sigma_y sigma_y
        = 0.5 * (sigma_p sigma_m + sigma_m sigma_p)``,
        which makes it convenient to use ladder operators and keep the matrices
        real.
    """
    si, sx, sy, sz = get_spins(xs='0xyz', j=spin_j, np_matrix=False) # use qutip
    sigma = [sx, sy, sz] # sx variable would be damaged in the later loops

    # generalize the parameters to arrays
    # built-in array is faster--so am I told
    nsize = (n-1) if boundary=='open' else n
    if isinstance(j, float): # same paremeters
        j = [[j,j,j]] * nsize
    elif np.array(j).ndim==1 and len(j) == 3:
        # given three numbers for x, y, z directions
        j = [j] * nsize
    if isinstance(hz, float):
        hz = [hz] * n

    # build the product state on-the-fly by qutip.tensor() function
    H = 0.
    for x, sx in enumerate(sigma): # sum over x,y,z directions
        for i in range(n-1):
            H += j[i][x] * qt.tensor([si]*i + [sx, sx] + [si]*(n-i-2))

    if boundary == 'periodic': # connect the two end points
        for x, sx in enumerate(sigma): # sum over x,y,z directions
            H += j[n-1][x] * qt.tensor([[sx] + [si]*(n-2) + [sx]])

    for i in range(n):
        H += hz[i] * qt.tensor([si]*i + [sz] + [si]*(n-i-1))

    if np_matrix:
        H = H.full()
    return H

hamil_xyz_1d = hamil_heisenberg_1d


def hamil_zeeman_1d(n, hz, np_matrix=True, spin_j=.5, sigma=None):
    r"""
    separate the magnetic field part for efficiency
    """
    if sigma is None:
        si, sz = get_spins(xs='0z', j=spin_j, np_matrix=False) # use qutip

    if isinstance(hz, float):
        hz = [hz] * n

    H = 0.
    for i in range(n):
        H += hz[i] * qt.tensor([si]*i + [sz] + [si]*(n-i-1))

    if np_matrix:
        H = H.full()
    return H


def hamil_x_1d(n, j, np_matrix=True, spin_j=.5, sigma=None, direction='x',
               boundary='open'):
    r"""
    separate one direction for efficiency
    """
    if sigma is None:
        si, sx = get_spins(xs='0'+direction, j=spin_j, np_matrix=False) # use qutip

    nsize = (n-1) if boundary=='open' else n
    if isinstance(j, float):
        j = [j] * nsize

    H = 0.
    for i in range(n-1):
        H += j[i] * qt.tensor([si]*i + [sx, sx] + [si]*(n-i-2))

    if boundary == 'periodic':
        H += j[n-1][x] * qt.tensor([[sx] + [si]*(n-2) + [sx]])

    if np_matrix:
        H = H.full()
    return H



def hamil_xxz_1d(n, j, delta, hz, np_matrix=True, spin_j=.5, sigma=None,
                 boundary='open'):
    r"""
    j * (xx + yy + zz * delta)
    reimplement for efficiency using ladder operator by defalut
    """
    if sigma is None:
        si, sp, sm, sz = get_spins(xs='0+-z', j=spin_j, np_matrix=False) # use qutip
        # scale parameters due to ladder operators
        j *= .5
        delta *= 2.

    nsize = (n-1) if boundary=='open' else n
    if isinstance(j, float):
        j = [j] * nsize
    if isinstance(delta, float):
        delta = [delta] * nsize
    if isinstance(hz, float):
        hz = [hz] * n
    #return hamil_xyz_1d(n, j, hz, np_matrix, spin_j, boundary)

    H = 0.
    for i in range(n-1):
        H += j[i] * (qt.tensor([si]*i + [sp, sm] + [si]*(n-i-2))
                   + qt.tensor([si]*i + [sm, sp] + [si]*(n-i-2))
                   + qt.tensor([si]*i + [sz, sz] + [si]*(n-i-2)) * delta[i]
                   )

    if boundary == 'periodic':
        H += j[n-1] * (qt.tensor([[sp] + [si]*(n-2) + [sm]])
                     + qt.tensor([[sm] + [si]*(n-2) + [sp]])
                     + qt.tensor([[sz] + [si]*(n-2) + [sz]]) * delta[i]
                     )

    for i in range(n):
        H += hz[i] * qt.tensor([si]*i + [sz] + [si]*(n-i-1))

    if np_matrix:
        H = H.full()
    return H



if __name__ == '__main__':
    n = 7
    j = -3.
    delta = .5
    hz = -2.

    si, sx, sy, sz = get_spins('0xyz', np_matrix=False)
    spin_list = get_prod_spin_list(n)

    H = hamil_xxz_1d(n, j, delta, hz, np_matrix=False)
    eigenvalues = H.eigenenergies()
    if n < 5:
        print_matrix('eigenvalues:', eigenvalues)


    # use qutip to find steady state
    # requires collapse operators, dephasing on the edges
    dephasing = np.sqrt(.1)

    # define collapse operators (e.g., dissipation at the edges)
    c_ops = []
    # add a local dephasing channel at site 0
    c_ops.append(dephasing * qt.tensor([sz] + [si]*(n-1)))
    # adding a local dephasing channel at site N-1
    c_ops.append(dephasing * qt.tensor([si]*(n-1) + [sz]))

    # find the steady state
    rho_ss = qt.steadystate(H, c_ops)

    # calculate observables in the steady state
    magnetization_z = qt.expect(qt.tensor([sz] + [si]*(n-1)), rho_ss)
    print('Steady-state magnetization at site 0:', magnetization_z)
