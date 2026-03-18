from lumeq import sys, np

from pyscf import lib
from pyscf.dft import numint
from pyscf.tools import cubegen

def creat_mesh_grids(mf, nxyz=None):
    r"""
    Create mesh grids for numerical integration.

    Args:
        mf: PySCF mean-field object.
        nxyz (list, optional): Cubic-grid parameters ``[nx, ny, nz, resolution]``.

    Returns:
        tuple: ``(cc, coords, weights, ngrids)``.
    """
    if nxyz is None:
        # default DFT mesh grids and weights
        cc = None
        coords = mf.grids.coords
        weights = mf.grids.weights
        ngrids = weights.shape[0]

    else:
        cc = cubegen.Cube(mf.mol, nx=nxyz[0], ny=nxyz[1], nz=nxyz[2], resolution=nxyz[3])
        coords = cc.get_coords()
        ngrids = cc.get_ngrids()
        weights = ( np.ones(ngrids)
                * (cc.xs[1]-cc.xs[0])
                * (cc.ys[1]-cc.ys[0])
                * (cc.zs[1]-cc.zs[0])
                )
        print('nx: ', cc.nx, ' ny: ', cc.ny, ' nz: ', cc.nz)

    #print('ngrids: ', ngrids)
    return cc, coords, weights, ngrids



class Grids:
    def __init__(self, mf, nxyz=None):
        r"""
        Initialize the Grids object.
        Args:
            mf: PySCF mean-field object.
            nxyz (list, optional): Cubic-grid parameters ``[nx, ny, nz, resolution]``.
        """
        cc, coords, weights, ngrids = creat_mesh_grids(mf, nxyz)

        self.cc = cc
        self.coords = coords
        self.weights = weights
        self.ngrids = ngrids
        self.mol = mf.mol

        self.ao_value = self.ao_on_grids()


    def ao_on_grids(self):
        r"""
        Calculate the atomic orbital values and their derivatives on the grids.

        Returns:
            numpy.ndarray: Atomic-orbital values and their derivatives on the grids.
        """
        # ao and its derivatives
        #ao_value = self.mol.eval_gto('GTOval_sph_deriv1', self.coords)
        # if we need to seperate batch
        ao_value = np.zeros((4, self.ngrids, self.mol.nao_nr()))
        blksize = min(8000, self.ngrids)
        for ip0, ip1 in lib.prange(0, self.ngrids, blksize):
            ao_value[:,ip0:ip1,:] = self.mol.eval_gto('GTOval_sph_deriv1', self.coords[ip0:ip1])
        return ao_value


    def orbital_on_grids(self, mo_coeff):
        r"""
        Calculate the molecular orbital values and their derivatives on the grids.

        Args:
            mo_coeff (numpy.ndarray): Molecular-orbital coefficients.

        Returns:
            numpy.ndarray: Molecular-orbital values and their derivatives on the grids.
        """
        # mo and its derivatives
        mo_value = np.einsum('xgm,mp->xgp', self.ao_value, mo_coeff)
        #mo_value = np.zeros((4, self.ngrids, mo_coeff.shape[1]))
        #blksize = min(8000, self.ngrids)
        #for ip0, ip1 in lib.prange(0, self.ngrids, blksize):
        #    mo_value[:,ip0:ip1,:] = np.einsum('pi,im->pm', self.ao_value[:,ip0:ip1,:], mo_coeff)
        return mo_value


    def density_on_grids(self, dm, xctype='GGA'):
        r"""
        Calculate the electron density and its derivatives on the grids.

        Args:
            dm (numpy.ndarray): Density matrix.
            xctype (str): Exchange-correlation functional type, e.g. ``'LDA'`` or ``'GGA'``.

        Returns:
            numpy.ndarray: Electron density and its derivatives on the grids.
        """
        # rho and its derivatives
        #rho_value = numint.eval_rho(self.mol, self.ao_value, dm, xctype=xctype)
        rho_value = np.zeros((4, self.ngrids))
        blksize = min(8000, self.ngrids)
        for ip0, ip1 in lib.prange(0, self.ngrids, blksize):
            rho_value[:,ip0:ip1] = numint.eval_rho(self.mol, self.ao_value[:,ip0:ip1,:], dm, xctype=xctype)
        return rho_value


    def plot_grid_values(self, values, title='density', filename='',
                         nums=None):
        r"""
        Plot the grid values in 3D space.
        Cubegen

        Args:
            values (numpy.ndarray): Grid values to plot.
            title (str): Plot title.
            filename (str): Cube-file prefix for saving the plot.
            nums (int, optional): Number of datasets to write when ``values`` contains multiple grids.
        """
        nx, ny, nz = self.cc.nx, self.cc.ny, self.cc.nz
        filename += title

        if 'orbital' in title:
            for iorb in orb_nums:
                self.cc.write(values[:, iorb].reshape(nx,ny,nz), filename+'_orb'+str(iorb)+'.cube')
        elif nums:
            for n in range(nums):
                self.cc.write(values[n].reshape(nx,ny,nz), filename+'_'+str(n+1)+'.cube')
        else:
            self.cc.write(values.reshape(nx,ny,nz), filename+'.cube')
