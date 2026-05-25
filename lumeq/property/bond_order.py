from lumeq import sys, np
from lumeq.embedding.fragment_entangle import (
    get_localized_orbital,
    get_localized_orbital_rdm,
)
from lumeq.embedding.mol_lo_tools import partition_lo_to_imps


def _ao_ranges(atom_slices):
    """Return ``(start, stop)`` AO ranges from PySCF-style atom slices."""
    ranges = np.asarray(atom_slices)
    if ranges.ndim != 2 or ranges.shape[1] not in (2, 4):
        raise ValueError("atom_slices must have shape (natm, 2) or (natm, 4)")
    if ranges.shape[1] == 4:
        ranges = ranges[:, 2:4]
    return ranges.astype(int)


def _sum_atom_blocks(ao_bond_order, atom_slices):
    ranges = _ao_ranges(atom_slices)
    atom_bond_order = np.zeros((len(ranges), len(ranges)), dtype=ao_bond_order.dtype)
    for i, (i0, i1) in enumerate(ranges):
        for j in range(i):
            j0, j1 = ranges[j]
            block_sum = np.sum(ao_bond_order[i0:i1, j0:j1])
            atom_bond_order[i, j] = atom_bond_order[j, i] = block_sum
    return atom_bond_order


def _dmet_bond_order(mf, fragments, lo_method='lowdin', extra_orb=0,
                     min_weight=.8, return_singular_values=False):
    """Calculate DMET fragment bond order from localized-orbital density blocks.

    The localized-orbital density matrix is built with
    ``get_localized_orbital_rdm``. For unrestricted references, alpha and beta
    orbitals are localized and partitioned separately. For each fragment pair
    and spin, the off-diagonal density block ``P_AB`` is decomposed as
    ``P_AB = U s V^T``. The bond order is
    ``sum(s_alpha**2) + sum(s_beta**2)``.
    """
    mol = mf.mol
    ovlp_ao = mf.get_ovlp()
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)

    def kernel(coeff_mo_in_ao, mo_occ, scale=1.):
        nocc = int(np.count_nonzero(mo_occ > 0))
        coeff_lo_in_ao = get_localized_orbital(mol, coeff_mo_in_ao, lo_method)
        dm_lo_in_ao = get_localized_orbital_rdm(
            coeff_lo_in_ao, coeff_mo_in_ao, ovlp_ao, nocc,
            scale=scale, extra_orb=extra_orb
        )
        fragment_lo_idx = partition_lo_to_imps(
            fragments, mol=mol, coeff_ao_lo=coeff_lo_in_ao,
            min_weight=min_weight
        )

        dmet_bo = np.zeros((len(fragments), len(fragments)))
        singular_values = {}
        for i in range(len(fragments)):
            for j in range(i):
                imp_lo_idx = fragment_lo_idx[i]
                env_lo_idx = fragment_lo_idx[j]
                dm_block = dm_lo_in_ao[np.ix_(imp_lo_idx, env_lo_idx)]
                _, s, _ = np.linalg.svd(dm_block, full_matrices=False)
                dmet_bo[j, i] = dmet_bo[i, j] = np.sum(s ** 2)
                singular_values[(j, i)] = s

        if return_singular_values:
            return dmet_bo, singular_values
        return dmet_bo

    if mo_occ.ndim == 1: # RKS
        return kernel(mo_coeff, mo_occ, scale=2.)

    # Unrestricted case: localize alpha and beta separately,
    # then sum bond orders.
    if return_singular_values:
        bo_alpha, singular_values_alpha = kernel(mo_coeff[0], mo_occ[0])
        bo_beta, singular_values_beta = kernel(mo_coeff[1], mo_occ[1])
        singular_values = [singular_values_alpha, singular_values_beta]
        return bo_alpha + bo_beta, singular_values
    else:
        bo_alpha = kernel(mo_coeff[0], mo_occ[0])
        bo_beta = kernel(mo_coeff[1], mo_occ[1])
        return bo_alpha + bo_beta


def bond_order(S=None, P=None, method='mayer', atom_slices=None, mf=None,
               fragments=None, lo_method='lowdin', extra_orb=0, min_weight=.8,
               return_singular_values=False):
    """Calculate the bond order of a molecule.

    Args:
        S (np.ndarray): The overlap matrix of the molecule.
        P (np.ndarray): The density matrix of the molecule.
        method (str, optional): The method to calculate bond order. Defaults to 'mayer'.
        atom_slices (np.ndarray, optional): AO ranges for each atom. Accepts
            ``(natm, 2)`` ``[start, stop]`` ranges or PySCF ``aoslice_by_atom()``
            ``(natm, 4)`` rows. If provided for Mayer bond order, AO
            contributions are summed into an atom-pair matrix.
        mf (object, optional): Converged PySCF mean-field object for DMET bond
            order.
        fragments (list, optional): Atom-index fragments for DMET bond order.
        lo_method (str, optional): Localized-orbital method for DMET. Defaults
            to ``'lowdin'``.
        extra_orb (int, optional): Extra orbitals included in the LO density.
        min_weight (float, optional): Minimum LO fragment assignment weight.
        return_singular_values (bool, optional): For DMET, also return the
            fragment-pair singular values and LO fragment indices.

    Returns:
        np.ndarray: The bond order matrix of the molecule.
    """
    if method == 'dmet':
        if mf is None or fragments is None:
            raise ValueError("DMET bond order requires mf and fragments")
        return _dmet_bond_order(
            mf, fragments, lo_method=lo_method, extra_orb=extra_orb,
            min_weight=min_weight,
            return_singular_values=return_singular_values,
        )

    S = np.asarray(S)
    P = np.asarray(P)
    if S.shape != P.shape or S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S and P must be square matrices with the same shape")

    if method == 'mayer':
        # Mayer bond order:
        #   D = P S
        #   b_mu,nu = D_mu,nu D_nu,mu
        #   B_A,B = sum_{mu in A} sum_{nu in B} b_mu,nu
        ps = np.dot(P, S)
        ao_bond_order = ps * ps.T
        if atom_slices is None:
            return ao_bond_order
        return _sum_atom_blocks(ao_bond_order, atom_slices)
    else:
        raise ValueError("Unknown bond order method: %s" % method)


def _test_h2o_molecule():
    from pyscf import gto, scf

    mol = gto.M(
        atom="""
            O  0.000000  0.000000  0.000000
            H  0.000000 -0.757160  0.586260
            H  0.000000  0.757160  0.586260
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    return mf.get_ovlp(), mf.make_rdm1(), mol.aoslice_by_atom(), "PySCF H2O/STO-3G"


def _test_h2o_dimer_molecule():
    from pyscf import gto, scf

    mol = gto.M(
        atom="""
            O  0.000000  0.000000  0.000000
            H  0.000000  0.000000  0.958400
            H  0.926627  0.000000 -0.239600
            O  0.000000  0.000000  2.800000
            H -0.756950  0.000000  3.387000
            H  0.756950  0.000000  3.387000
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    fragments = [[0, 1, 2], [3, 4, 5]]
    return mf, fragments, "PySCF hydrogen-bonded (H2O)2/STO-3G"


if __name__ == '__main__':
    np.set_printoptions(precision=8, suppress=True)

    S, P, atom_slices, label = _test_h2o_molecule()
    ao_bo = bond_order(S, P, method='mayer')
    atom_bo = bond_order(S, P, method='mayer', atom_slices=atom_slices)

    print("Testing Mayer bond order with", label)
    print("Overlap matrix S:")
    print(S)
    print("Density matrix P:")
    print(P)
    print("AO Mayer contribution matrix:")
    print(ao_bo)
    print("Atom-pair Mayer bond order matrix:")
    print(atom_bo)

    mf, fragments, label = _test_h2o_dimer_molecule()
    dmet_bo, singular_values = bond_order(
        method='dmet',
        mf=mf,
        fragments=fragments,
        lo_method='lowdin',
        return_singular_values=True,
    )

    print("\nTesting DMET bond order with", label)
    print("Fragments:", fragments)
    print("Singular values for fragment pair (0, 1):")
    print(singular_values[(0, 1)])
    print("DMET fragment bond order matrix:")
    print(dmet_bo)
