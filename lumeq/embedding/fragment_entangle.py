r"""DMET-style fragment entanglement for molecular orbital channels.

AO overlap is :math:`S`; column coefficient matrices are
:math:`C_{MO}=C^{AO,MO}`, :math:`C_{LO}=C^{AO,LO}`, and
:math:`C_{EO}=C^{AO,EO}`. The dagger denotes the conjugate transpose.
For an orthonormal target basis X and MO occupations n,

.. math::

    A_X=C_X^\dagger S C_{MO},\qquad
    D^X=A_X\,\mathrm{diag}(n)\,A_X^\dagger.

The LO density has impurity/environment blocks. Off-diagonal SVD,
:math:`D_{imp,env}=U\Sigma V^\dagger`, supplies environment bath vectors.
In impurity/environment row order, :math:`T=\mathrm{diag}(I_{imp},V)`,
:math:`C_{EO}=C_{LO}T`, and :math:`D^{EO}=T^\dagger D^{LO}T`.

Bath selection may use adjusted occupations :math:`\tilde n`; physical EO
populations and energy contractions use the original SCF occupations n.
Density variables ending in ``_in_ao`` are nevertheless expressed in their
LO or EO basis; only coefficient matrices have AO rows.
"""

from lumeq import sys, np
from lumeq.utils import print_matrix
from lumeq.embedding.mol_lo_tools import partition_lo_to_imps



def get_localized_orbital(mol, coeff, method='pipek_mezey', mo_occ=None,
                          **kwargs):
    r"""Construct orthonormal localized orbitals for one orbital channel.

    The AO coefficients satisfy :math:`C_{LO}^\dagger S C_{LO}=I` for retained,
    linearly independent columns. IAO orbitals are symmetrically orthonormalized.

    Args:
        mol: PySCF molecule, or periodic cell for IAO calculations.
        coeff (numpy.ndarray): MO coefficients :math:`C_{MO}`, shape ``(nao, nmo)``.
            For IAO without mo_occ, supply occupied columns only. With kpts, supply
            one matrix per k point; the occupied column counts may differ.
        method (str): ``pipek_mezey``, ``cholesky``, ``iao``, ``lowdin``, or
            ``meta_lowdin``. AO orthogonalization methods do not use coeff.
        mo_occ (numpy.ndarray, optional): Occupations :math:`n_p`, shape ``(nmo,)``
            or one array per k point. IAO selects columns with :math:`n_p>0`;
            other methods ignore these occupations.
        **kwargs: IAO options: ``minao`` (reference basis, default ``minao``) and
            ``kpts`` (k points, default None).

    Returns:
        numpy.ndarray: :math:`C_{LO}=C^{AO,LO}`, shape ``(nao, nlo)``. The column
        count is nmo for PM/Cholesky, the reference-basis size for IAO, and nao
        for Lowdin methods. K-point IAOs have shape ``(nkpts, nao, nlo)``.

    Raises:
        ValueError: The localization method is unknown.
        RuntimeError: Pipek-Mezey localization did not converge.
    """
    from pyscf import lo

    if method in ('lowdin', 'meta_lowdin'):
        # Return shape: (nao, nao), independent of MO coefficients and occupations.
        return lo.orth_ao(mol, method)

    if method == 'iao':
        kpts = kwargs.get('kpts', None)
        if kpts is not None:
            overlap = mol.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
            if mo_occ is not None:
                coeff = [c[:, np.asarray(o) > 0] for c, o in zip(coeff, mo_occ)]
        else:
            overlap = (mol.pbc_intor('int1e_ovlp', hermi=1)
                       if getattr(mol, 'dimension', 0) > 0
                       else mol.intor_symmetric('int1e_ovlp'))
            if mo_occ is not None:
                coeff = coeff[:, np.asarray(mo_occ) > 0]
        iaos = lo.iao.iao(mol, coeff, minao=kwargs.get('minao', 'minao'), kpts=kpts)
        if kpts is not None:
            # Return shape: (nkpts, nao, nref); orthonormal within each k point.
            return np.asarray([lo.vec_lowdin(c, s)
                               for c, s in zip(iaos, overlap)])
        # Return shape: (nao, nref); IAOs span the minimal reference space.
        return lo.vec_lowdin(iaos, overlap)

    if method == 'pipek_mezey':
        pm = lo.PM(mol, coeff)
        pm.conv_tol = 1e-8 # need high resolution
        # PySCF exposes the convergence flag through the iteration callback.
        converged = coeff.shape[1] <= 1
        def check_convergence(env):
            nonlocal converged
            converged = env['conv']
        coeff = pm.kernel(callback=check_convergence)
        if not converged:
            raise RuntimeError('Pipek-Mezey localization did not converge.')
        # Return shape: (nao, nmo), matching the selected MO coefficient matrix.
        return coeff
    elif method == 'cholesky':
        # Return shape: (nao, nmo), matching the selected MO coefficient matrix.
        return lo.cholesky_mos(coeff)
    raise ValueError(f'Unknown localization method: {method}')


def get_mo_in_lo(coeff_lo_in_ao, coeff_mo_in_ao, ovlp_ao):
    r"""Project MO coefficients into an orthonormal LO basis.

    .. math::

        A=C_{LO}^\dagger S C_{MO}.

    Args:
        coeff_lo_in_ao (numpy.ndarray): Target coefficients :math:`C_{LO}`,
            shape ``(nao, nlo)``, orthonormal in the AO metric S.
        coeff_mo_in_ao (numpy.ndarray): MO coefficients :math:`C_{MO}`,
            shape ``(nao, nmo)``.
        ovlp_ao (numpy.ndarray): AO overlap :math:`S`, shape ``(nao, nao)``.

    Returns:
        numpy.ndarray: MO coefficients :math:`A=C^{LO,MO}` in the target LO
        basis, shape ``(nlo, nmo)``. Broadcast-compatible leading spin axes
        are supported. Using EO coefficients as the target instead returns
        :math:`C^{EO,MO}`, shape ``(neo, nmo)``.
    """
    # Equivalent: np.einsum('...pi,pq,...qj->...ij', coeff_lo_in_ao.conj(), ovlp_ao, coeff_mo_in_ao)
    return coeff_lo_in_ao.conj().swapaxes(-1, -2) @ ovlp_ao @ coeff_mo_in_ao


def get_rdm_ortho(coeff_mo, mo_occ):
    r"""Construct an orthonormal-basis density from MO coefficients and occupations.

    .. math::

        D=C_{MO}\,\mathrm{diag}(n)\,C_{MO}^\dagger.

    Args:
        coeff_mo (numpy.ndarray): MO coefficients in an orthonormal target
            basis, shape ``(nbasis, nmo)``. For an LO or EO target these are
            :math:`C^{LO,MO}` or :math:`C^{EO,MO}`, respectively.
        mo_occ (numpy.ndarray): Occupations :math:`n`, shape ``(nmo,)``.
            Original or bath-adjusted occupations may be supplied.

    Returns:
        numpy.ndarray: Density :math:`D` in the target basis, shape
        ``(nbasis, nbasis)``. Broadcast-compatible leading axes are supported;
        separate spin densities are retained rather than summed.
    """
    # Equivalent: np.einsum('...ik,...k,...jk->...ij', coeff_mo, mo_occ, coeff_mo.conj())
    return (coeff_mo * mo_occ[..., None, :]) @ coeff_mo.conj().swapaxes(-1, -2)


def get_localized_orbital_rdm(coeff_lo_in_ao, coeff_mo_in_ao, ovlp_ao, mo_occ):
    r"""Project an occupation-weighted MO density into an orthonormal basis.

    .. math::

        A=C_{LO}^\dagger S C_{MO},\qquad D^{LO}=A\,\mathrm{diag}(n)\,A^\dagger.

    The target may also be an EO basis: replacing :math:`C_{LO}` by
    :math:`C_{EO}` gives :math:`D^{EO}`. The supplied occupations determine
    whether the density is physical or artificially adjusted for bath selection.

    Args:
        coeff_lo_in_ao (numpy.ndarray): Target coefficients :math:`C_{LO}`,
            shape ``(nao, nlo)``, orthonormal in the AO metric S.
        coeff_mo_in_ao (numpy.ndarray): MO coefficients :math:`C_{MO}`,
            shape ``(nao, nmo)``.
        ovlp_ao (numpy.ndarray): AO overlap :math:`S`, shape ``(nao, nao)``.
        mo_occ (numpy.ndarray): Occupations :math:`n`, shape ``(nmo,)``.

    Returns:
        numpy.ndarray: :math:`D^{LO}`, shape ``(nlo, nlo)``. Broadcast-compatible
        leading axes are supported; spin densities are never summed.
    """
    coeff_mo_in_lo = get_mo_in_lo(coeff_lo_in_ao, coeff_mo_in_ao, ovlp_ao)
    return get_rdm_ortho(coeff_mo_in_lo, mo_occ)


def fragment_localization(mol, coeff_mo_in_ao, ovlp_ao, frgm_list, mo_occ=None,
                          method='pipek_mezey', min_weight=.8):
    r"""Localize one orbital channel and assign LOs to atomic fragments.

    Args:
        mol: PySCF molecule supplying AO/atom information.
        coeff_mo_in_ao (numpy.ndarray): :math:`C_{MO}`, shape ``(nao, nmo)``.
        ovlp_ao (numpy.ndarray): :math:`S`, shape ``(nao, nao)``; retained for
            interface compatibility and not used by this function.
        frgm_list (list): Lists of zero-based atom indices for each fragment.
        mo_occ (numpy.ndarray, optional): MO occupations, shape ``(nmo,)``.
            Used by IAO to select occupied MOs; if omitted, IAO expects coeff_mo_in_ao
            to contain only occupied columns.
        method (str): Localization method accepted by get_localized_orbital.
        min_weight (float): Minimum fragment weight for assigning an LO. LOs below
            the threshold may remain unassigned.

    Returns:
        tuple: ``(coeff_lo_in_ao, frgm_lo_idx)``. The coefficients
        :math:`C^{AO,LO}` have shape ``(nao, nlo)``; frgm_lo_idx contains one list
        of zero-based LO indices per fragment. No density matrix is returned.
    """
    # Localized orbitals depend on the localization method
    coeff_lo_in_ao = get_localized_orbital(mol, coeff_mo_in_ao, method, mo_occ)
    frgm_lo_idx = partition_lo_to_imps(frgm_list, mol=mol,
                                       coeff_ao_lo=coeff_lo_in_ao,
                                       min_weight=min_weight)

    return coeff_lo_in_ao, frgm_lo_idx


def get_embedding_orbital(dm_lo_in_ao, coeff_lo_in_ao, ovlp_ao, imp_lo_idx,
                          env_lo_idx, method=0, threshold=1e-12, max_occ=2.):
    r"""Construct impurity and bath orbitals from one channel's LO density.

    For method 0, :math:`D_{imp,env}=U\Sigma V^\dagger`; retained columns of V
    span the bath. For method 1, retain environment-density eigenvectors with
    :math:`\epsilon<\lambda<n_{max}-\epsilon`. With impurity/environment rows
    ordered together, :math:`T=\mathrm{diag}(I_{imp},V)` and

    .. math::

        C_{EO}=C_{LO}T,\qquad D^{EO}=T^\dagger D^{LO}T.

    Args:
        dm_lo_in_ao (numpy.ndarray): :math:`D^{LO}`, shape ``(nlo, nlo)``;
            despite its name, this is an LO-basis density, possibly bath-adjusted.
        coeff_lo_in_ao (numpy.ndarray): :math:`C_{LO}`, shape ``(nao, nlo)``.
        ovlp_ao (numpy.ndarray): :math:`S`, shape ``(nao, nao)``.
        imp_lo_idx (array_like): Impurity LO indices, defining the first EO columns.
        env_lo_idx (array_like): Disjoint environment LO indices.
        method (int): Bath selection by off-diagonal SVD (0) or environment
            eigenvectors (1).
        threshold (float): :math:`\epsilon`, singular/eigenvalue cutoff.
        max_occ (float): :math:`n_{max}`, 2 for a restricted channel or 1 per
            unrestricted spin channel; used only by method 1.

    Returns:
        tuple: ``(coeff_eo_in_ao, dm_eo_in_ao)`` with shapes ``(nao, neo)`` and
        ``(neo, neo)``, where :math:`neo=n_{imp}+n_{bath}`. The density is in
        the EO basis, despite its name, and preserves the input occupation convention.

    Raises:
        ValueError: The bath-selection method is unknown.
    """
    def embed_spin_orbital(dm_lo_in_ao, iprint=0):
        r"""Select environment bath vectors for the enclosing orbital channel.

        Args:
            dm_lo_in_ao (numpy.ndarray): :math:`D^{LO}`, shape ``(nlo, nlo)``.
            iprint (int): Print the singular/eigenvalues when positive.

        Returns:
            numpy.ndarray: Bath coefficients V, shape ``(nenv, nbath)``.
        """
        if method == 0: # singular value vectors of off-diagonal block of the dm in lo
            dm_imp_env_lo = dm_lo_in_ao[np.ix_(imp_lo_idx, env_lo_idx)] # get environmental orbitals
            _, s, vt = np.linalg.svd(dm_imp_env_lo, full_matrices=False)
            if iprint > 0: print_matrix('singular values:'+str(np.sum(s)), s)
            v = vt[s>threshold].conj().T
        elif method == 1: # eigenvectors of environment diagonal block of the dm in lo
            dm_env_env_lo = dm_lo_in_ao[np.ix_(env_lo_idx, env_lo_idx)]
            s, v = np.linalg.eigh(dm_env_env_lo)
            if iprint > 0: print_matrix('eigen-values:', s)
            v = v[:, (s>threshold)&(s<max_occ-threshold)]
        else:
            raise ValueError(f'Unknown embedding method: {method}')
        return v

    V = embed_spin_orbital(dm_lo_in_ao, iprint=1)

    coeff_imp = coeff_lo_in_ao[..., imp_lo_idx] # idensity transformation
    # coeff_env = np.einsum('...pi,...ij->...pj', coeff_lo_in_ao[..., env_lo_idx], V)
    coeff_env = coeff_lo_in_ao[..., env_lo_idx] @ V
    coeff_eo_in_ao = np.concatenate((coeff_imp, coeff_env), axis=-1)
    # coeff_eo_in_lo = np.einsum('...pi,pq,...qj->...ij', coeff_lo_in_ao.conj(), ovlp_ao, coeff_eo_in_ao)
    # coeff_eo_in_lo = coeff_lo_in_ao.conj().T @ (ovlp_ao @ coeff_eo_in_ao)

    # computational efficient way
    nimp = len(imp_lo_idx)
    neo = nimp + V.shape[1]
    coeff_eo_in_lo = np.zeros((coeff_lo_in_ao.shape[1], neo), dtype=V.dtype)
    coeff_eo_in_lo[imp_lo_idx, np.arange(nimp)] = 1
    coeff_eo_in_lo[np.ix_(env_lo_idx, np.arange(nimp, neo))] = V

    # dm_eo_in_ao = np.einsum('...pi,...pq,...qj->...ij', coeff_eo_in_lo.conj(), dm_lo_in_ao, coeff_eo_in_lo)
    dm_eo_in_ao = coeff_eo_in_lo.conj().T @ (dm_lo_in_ao @ coeff_eo_in_lo)

    # Equivalent block construction in impurity/environment row order:
    # IV = np.block([[np.eye(nimp), np.zeros((nimp, V.shape[1]))],
    #                [np.zeros((len(env_lo_idx), nimp)), V]])
    # order = np.concatenate((np.asarray(imp_lo_idx, dtype=int),
    #                         np.asarray(env_lo_idx, dtype=int)))
    # dm_lo = dm_lo_in_ao[np.ix_(order, order)]
    # dm_eo_in_ao = IV.conj().T @ (dm_lo @ IV)
    # dm_eo_in_ao = np.einsum('ji,jk,kl->il', IV.conj(), dm_lo, IV)

    return coeff_eo_in_ao, dm_eo_in_ao


def get_embedding_electrons(dm_eo_in_ao, extra_orb=0):
    r"""Count electrons and orbitals in a supplied EO density.

    .. math::

        N_{EO}=\mathrm{round}(\mathrm{Re}\,\mathrm{Tr}\,D^{EO})-\Delta N.

    Args:
        dm_eo_in_ao (numpy.ndarray): EO-basis density :math:`D^{EO}`,
            shape ``(neo, neo)``. Use original SCF occupations for physical counts.
        extra_orb (int): Explicit electron correction :math:`\Delta N`, not an
            orbital count. Default 0; no correction is needed for a physical density.

    Returns:
        tuple: ``(nocc_eo, neo)``: rounded electron population and EO count.
        The unrounded population can be fractional in a projected space.
    """
    neo = dm_eo_in_ao.shape[-1]
    # effective number of electrons in embedding space
    nocc_eo = int(round(np.trace(dm_eo_in_ao).real)) # round up to an integer
    nocc_eo -= extra_orb
    return (nocc_eo, neo)


def get_embedding_energy(hcore_ao, fock_ao, coeff_eo_in_ao, dm_eo_in_ao,
                         neo_imp):
    r"""Evaluate one channel's impurity electronic energy.

    .. math::

        h^{EO}=C_{EO}^\dagger h C_{EO},\quad
        F^{EO}=C_{EO}^\dagger F C_{EO},\quad
        E_{imp}=\tfrac12\mathrm{Re}\sum_{i\in imp,j}
            (h^{EO}+F^{EO})_{ij}D^{EO}_{ji}.

    The implementation forms :math:`X=C_{EO}D^{EO}_{:,imp}` and contracts
    :math:`\tfrac12\mathrm{Re}\sum_{\mu i}C_{imp,\mu i}^*[(h+F)X]_{\mu i}`.
    AO matrix products therefore use only n_imp columns rather than all EOs.

    Args:
        hcore_ao (numpy.ndarray): AO core Hamiltonian h, shape ``(nao, nao)``.
        fock_ao (numpy.ndarray): Channel Fock matrix F, shape ``(nao, nao)``.
        coeff_eo_in_ao (numpy.ndarray): :math:`C_{EO}`, shape ``(nao, neo)``.
        dm_eo_in_ao (numpy.ndarray): Physical :math:`D^{EO}`, shape ``(neo, neo)``.
        neo_imp (int): Number of leading impurity EO columns.

    Returns:
        float: Channel impurity electronic energy, excluding nuclear repulsion.
        This is the Hartree-Fock contraction. A KS energy requires separate
        exchange-correlation energy accounting; inserting a KS Fock matrix alone
        does not give the full KS electronic energy.
    """
    # full matrix construction
    # h1e_eo = coeff_eo_in_ao.conj().T @ (hcore_ao @ coeff_eo_in_ao)
    # f1e_eo = coeff_eo_in_ao.conj().T @ (fock_ao @ coeff_eo_in_ao)
    # h1e_eo = np.einsum('pi,pq,qj->ij', coeff_eo_in_ao.conj(), hcore_ao, coeff_eo_in_ao)
    # f1e_eo = np.einsum('pi,pq,qj->ij', coeff_eo_in_ao.conj(), fock_ao, coeff_eo_in_ao)
    # energy = .5 * np.einsum('ij,ji->', (h1e_eo+f1e_eo)[:neo_imp], dm_eo_in_ao[:, :neo_imp]).real
    # energy = np.sum((h1e_eo+f1e_eo)[:neo_imp] * dm_eo_in_ao[:, :neo_imp].T).real * .5

    coeff_imp = coeff_eo_in_ao[:, :neo_imp]
    # density_imp = np.einsum('pi,ij->pj', coeff_eo_in_ao, dm_eo_in_ao[:, :neo_imp])
    density_imp = coeff_eo_in_ao @ dm_eo_in_ao[:, :neo_imp]
    # energy = .5 * np.einsum('pi,pq,qi->', coeff_imp.conj(), hcore_ao+fock_ao, density_imp).real
    energy = .5 * np.sum(coeff_imp.conj() * (
            (hcore_ao + fock_ao) @ density_imp)).real

    return energy


def configure_spin_channels(mf):
    r"""Extract orbital channels, converting ROHF/ROKS to UHF/UKS first.

    Args:
        mf: Converged molecular PySCF reference. The input object is not modified.

    Returns:
        dict: Keys ``total`` for a closed-shell restricted reference, or ``alpha``
        and ``beta`` for an unrestricted/converted reference. Each value contains
        ``occupation`` (:math:`n_{max}=2` or 1), ``mo_occ`` (original occupations,
        shape ``(nmo,)``), ``mo_coeff`` (:math:`C_{MO}`, shape ``(nao, nmo)``),
        and ``fock_ao`` (:math:`F_\sigma`, shape ``(nao, nao)``).
        Arrays may share storage with the reference.
    """
    if mf.istype('ROHF'):
        mf = mf.to_uks() if hasattr(mf, 'to_uks') else mf.to_uhf()

    nspin = 1 if np.asarray(mf.mo_occ).ndim == 1 else 2
    max_occ = 2. / nspin

    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    fock_ao = mf.get_fock()

    channels = {}
    spins = ['total'] if nspin == 1 else ['alpha', 'beta']
    for i, spin in enumerate(spins):
        channels[spin] = {'occupation': max_occ,
                          'mo_occ': mo_occ if nspin == 1 else mo_occ[i],
                          'mo_coeff': mo_coeff if nspin == 1 else mo_coeff[i],
                          'fock_ao': fock_ao if nspin == 1 else fock_ao[i]}
    return channels



class EmbeddingMeanField():
    r"""Embedding for one supplied spatial-orbital or spin-orbital channel.

    Coefficients :math:`C^{AO,LO}` and :math:`C^{AO,EO}` use AO coordinates;
    densities :math:`D^{LO}` and :math:`D^{EO}` use the named orbital basis.
    ``dm_eo_in_ao`` retains bath-adjusted occupations :math:`\tilde n`, whereas
    ``physical_dm_eo`` projects the original occupations n into the same EO basis.
    Spin conversion and assembly belong to EmbeddingKS.
    The molecule, overlap, occupations, MO coefficients, and Fock matrix are
    copied at initialization to protect the cached MO-to-LO transformation.
    Rebuild this object after changing its reference or localized orbitals.
    """
    def __init__(self, mol, ovlp_ao, channel, frgm_list,
                 method='pipek_mezey', min_weight=.8):
        r"""Store one channel and construct its localized orbitals and partition.

        Args:
            mol: PySCF molecule for this channel.
            ovlp_ao (numpy.ndarray): AO overlap :math:`S`, shape ``(nao, nao)``.
            channel (dict): Entry from configure_spin_channels containing
                ``occupation``, ``mo_occ``, ``mo_coeff``, and ``fock_ao``.
            frgm_list (list): Zero-based atom-index lists for each fragment.
            method (str): Localization method.
            min_weight (float): Minimum LO weight for fragment assignment.

        Returns:
            None: Copies channel data and initializes :math:`C_{LO}`, fragment
            indices, and cached :math:`C^{LO,MO}`. Call emb_basis_dmet before
            requesting embedded electrons, orbitals, or energies.
            Stores lo_orth_error, the Frobenius norm of
            :math:`C_{LO}^\dagger S C_{LO}-I`.

        Raises:
            ValueError: The LO orthonormality error exceeds 1e-8 or is nonfinite.
        """
        self.mol = mol.copy()
        self.ovlp_ao = ovlp_ao.copy()

        self.max_occ = channel['occupation']
        self.mo_occ = channel['mo_occ'].copy()
        self.coeff_mo_in_ao = channel['mo_coeff'].copy()
        self.fock_ao = channel['fock_ao'].copy()
        self.coeff_eo_in_ao = None

        self.coeff_lo_in_ao, self.frgm_lo_idx = fragment_localization(
                self.mol, self.coeff_mo_in_ao, self.ovlp_ao, frgm_list,
                self.mo_occ, method=method, min_weight=min_weight
        )

        metric = self.coeff_lo_in_ao.conj().T @ (self.ovlp_ao @ self.coeff_lo_in_ao)
        self.lo_orth_error = np.linalg.norm(metric - np.eye(metric.shape[0]))
        if not np.isfinite(self.lo_orth_error) or self.lo_orth_error > 1e-8:
            raise ValueError('Localized orbitals are not orthonormal in the AO metric.')

        # cache the MO coefficients in the LO basis for later density construction
        self.coeff_mo_in_lo = get_mo_in_lo(
                self.coeff_lo_in_ao, self.coeff_mo_in_ao, self.ovlp_ao)


    def emb_basis_dmet(self, ifrgm, embed_method=0, extra_orb=0, threshold=1e-12):
        r"""Select a fragment and build its impurity/bath basis.

        Bath selection uses adjusted occupations :math:`\tilde n`, but physical
        populations use :math:`D_{phys}^{EO}=A\,\mathrm{diag}(n)A^\dagger`, with
        :math:`A=C_{EO}^\dagger S C_{MO}` and the original SCF occupations n.

        Args:
            ifrgm (int): Zero-based fragment index (Python negative indexing is allowed).
            embed_method (int): Off-diagonal SVD (0) or environment eigenvectors (1).
            extra_orb (int): Signed orbital-count adjustment. Positive values occupy
                the next virtuals with n_max; negative values empty the last occupied
                MOs. Occupied MOs must precede virtuals. Each call starts from original
                occupations and clips the adjustment to available columns.
            threshold (float): Singular/eigenvalue cutoff passed to bath selection.

        Returns:
            tuple: ``(coeff_eo_in_ao, dm_eo_in_ao)``, shapes ``(nao, neo)`` and
            ``(neo, neo)``. This returned density uses adjusted occupations.
            Stores physical_dm_eo separately using the original occupations;
            both densities coincide when extra_orb is 0. Environment indices contain
            every non-impurity LO, including LOs not assigned to a fragment.

        Raises:
            ValueError: extra_orb is not an integer orbital count.
        """
        if int(extra_orb) != extra_orb:
            raise ValueError('extra_orb must be an integer orbital count.')
        extra_orb = int(extra_orb)

        self.imp_lo_idx = np.asarray(self.frgm_lo_idx[ifrgm], dtype=int)
        self.env_lo_idx = np.setdiff1d(
                np.arange(self.coeff_lo_in_ao.shape[-1]), self.imp_lo_idx)

        # number of impurity embedding orbitals
        self.neo_imp = len(self.imp_lo_idx)
        self.neo_env = len(self.env_lo_idx)

        mo_occ = self.mo_occ.copy()
        nocc = np.count_nonzero(mo_occ > 0)
        target = np.clip(nocc + extra_orb, 0, mo_occ.size)
        mo_occ[min(nocc, target):max(nocc, target)] = (
            self.max_occ if extra_orb > 0 else 0
        )

        dm_lo_in_ao = get_rdm_ortho(self.coeff_mo_in_lo, mo_occ)

        self.coeff_eo_in_ao, self.dm_eo_in_ao = get_embedding_orbital(
                dm_lo_in_ao, self.coeff_lo_in_ao, self.ovlp_ao,
                self.imp_lo_idx, self.env_lo_idx, method=embed_method,
                max_occ=self.max_occ, threshold=threshold)

        if extra_orb == 0: # no artifical adjustment
            self.physical_dm_eo = self.dm_eo_in_ao
        else:
            self.physical_dm_eo = get_localized_orbital_rdm(
                    self.coeff_eo_in_ao, self.coeff_mo_in_ao, self.ovlp_ao,
                    self.mo_occ)

        return self.coeff_eo_in_ao, self.dm_eo_in_ao


    def get_embedding_electrons(self):
        r"""Store and return the physical population of the selected EO space.

        Args:
            None.

        Returns:
            tuple: ``(self.nelec, self.neo)``, where
            :math:`N_{EO}=\mathrm{round}(\mathrm{Re}\,\mathrm{Tr}\,D_{phys}^{EO})`
            and neo is the EO count. Extra bath occupations are excluded.
            Stores electron_population (the unrounded trace) and
            electron_count_residual, :math:`\delta N=\mathrm{Tr}(D_{phys}^{EO})-N_{EO}`.

        Raises:
            RuntimeError: No embedding basis has been constructed.
        """
        if self.coeff_eo_in_ao is None:
            raise RuntimeError('Call emb_basis_dmet before get_embedding_electrons.')
        self.nelec, self.neo = get_embedding_electrons(self.physical_dm_eo)
        self.electron_population = np.trace(self.physical_dm_eo).real
        self.electron_count_residual = self.electron_population - self.nelec
        return self.nelec, self.neo


    def get_impurity_density(self, coeff_imp=None):
        r"""Project this channel's original density into a spatial impurity basis.

        .. math::

            A=C_{imp}^\dagger S C_{MO},\qquad
            d_{imp}=A\,\mathrm{diag}(n)A^\dagger.

        Args:
            coeff_imp (numpy.ndarray, optional): Orthonormal spatial coefficients,
                shape ``(nao, nimp)``. Defaults to the leading neo_imp EO columns.
                Supply the same basis to both channels when combining spin data.

        Returns:
            numpy.ndarray: Channel density, shape ``(nimp, nimp)``, using original
            SCF occupations without extra-orbital adjustments or rounding. For a
            restricted channel this is the spin-summed density. No spin splitting
            or spin expectation value is computed here.

        Raises:
            RuntimeError: No embedding basis has been constructed.
            ValueError: The supplied impurity basis is not orthonormal.
        """
        if self.coeff_eo_in_ao is None:
            raise RuntimeError('Call emb_basis_dmet before get_impurity_density.')

        if coeff_imp is None:
            return self.physical_dm_eo[:self.neo_imp, :self.neo_imp]

        metric = coeff_imp.conj().T @ (self.ovlp_ao @ coeff_imp)
        if not np.allclose(metric, np.eye(metric.shape[0]), atol=1e-8, rtol=0):
            raise ValueError('The common impurity basis must be orthonormal.')

        return get_localized_orbital_rdm(
                coeff_imp, self.coeff_mo_in_ao, self.ovlp_ao, self.mo_occ)


    def get_eomf(self):
        r"""Canonicalize the channel Fock matrix in the selected EO space.

        .. math::

            F^{EO}=C_{EO}^\dagger F C_{EO},\quad
            F^{EO}U=U\varepsilon,\quad C_{emb}=C_{EO}U.

        Args:
            None.

        Returns:
            dict: ``nelectron`` is the rounded physical EO electron population;
            ``mo_occ`` has shape ``(neo,)`` and fills the lowest energies up to max_occ
            with a possible remainder; ``mo_energy`` is :math:`\varepsilon`, shape
            ``(neo,)``; ``mo_coeff`` is :math:`C_{emb}`, shape ``(nao, neo)``.
            Zero-energy orbitals are retained. This is a channel orbital reference,
            not a new self-consistent solution; spin assembly is handled externally.

        Raises:
            RuntimeError: No embedding basis has been constructed.
            ValueError: The rounded electron count exceeds the channel capacity.
        """
        if self.coeff_eo_in_ao is None:
            raise RuntimeError('Call emb_basis_dmet before get_eomf.')

        # Count original SCF electrons in the selected EO space.
        nelec, neo = self.get_embedding_electrons()
        if not 0 <= nelec <= neo*self.max_occ:
            raise ValueError('Invalid embedded electron count.')

        coeff_eo_in_ao = self.coeff_eo_in_ao
        fock_eo = coeff_eo_in_ao.conj().T @ (self.fock_ao @ coeff_eo_in_ao)

        # embedding orbital and its orbital energy
        mo_energy, mo_coeff = np.linalg.eigh(fock_eo)
        mo_coeff = coeff_eo_in_ao @ mo_coeff
        mo_occ = np.zeros_like(mo_energy)
        nocc, remainder = divmod(nelec, int(self.max_occ))
        mo_occ[:nocc] = self.max_occ
        if remainder: mo_occ[nocc] = remainder
        print_matrix('mo_energy in embedding:', mo_energy)

        eomf = {'nelectron': nelec, 'mo_occ': mo_occ,
                'mo_coeff': mo_coeff, 'mo_energy': mo_energy}
        return eomf



class EmbeddingKS():
    r"""Assemble molecular restricted or unrestricted embedding channels.

    ``spin_channels`` holds the reference data; ``embedding`` holds one
    EmbeddingMeanField per channel. ROHF/ROKS references become UHF/UKS copies.
    Channel EO dimensions may differ. Physical energies/populations use original
    SCF occupations n; extra occupations :math:`\tilde n` affect bath selection.
    """
    def __init__(self, mf, frgm_list, method='pipek_mezey', min_weight=.8):
        r"""Configure reference channels and initialize their localized bases.

        Args:
            mf: Converged PySCF molecular mean-field reference.
            frgm_list (list): Zero-based atom-index lists defining fragments.
            method (str): Localization method for each channel.
            min_weight (float): Minimum LO weight for fragment assignment.

        Returns:
            None: Stores the reference (converted for ROHF/ROKS) and initializes the
            per-channel embeddings. Does not yet construct EO bases.
        """
        self.mf = mf
        if mf.istype('ROHF'):
            self.mf = mf.to_uks() if hasattr(mf, 'to_uks') else mf.to_uhf()

        self.spin_channels = configure_spin_channels(mf)
        self.embedding = {}

        for spin, channel in self.spin_channels.items():
            self.embedding[spin] = EmbeddingMeanField(
                    mf.mol, mf.get_ovlp(), channel, frgm_list,
                    method=method, min_weight=min_weight
            )


    def emb_basis_dmet(self, ifrgm, embed_method=0, extra_orb=0, threshold=1e-12):
        r"""Construct the selected fragment's EO basis in every channel.

        Args:
            ifrgm (int): Fragment index passed to each channel.
            embed_method (int): Off-diagonal SVD (0) or environment eigenvectors (1).
            extra_orb (int or pair of ints): Signed bath orbital adjustment. A scalar
                applies to every channel; a pair is ordered alpha, beta for UHF/UKS.
            threshold (float): Singular/eigenvalue cutoff passed to each channel.

        Returns:
            None: Updates each entry in self.embedding with :math:`C_{EO}`,
            bath-adjusted :math:`D^{EO}`, and physical :math:`D_{phys}^{EO}`.
        """
        extra_orb = np.broadcast_to(extra_orb, (len(self.embedding),))
        for i, (spin, channel) in enumerate(self.spin_channels.items()):
            self.embedding[spin].emb_basis_dmet(ifrgm,
                            embed_method=embed_method, extra_orb=extra_orb[i],
                            threshold=threshold)


    def get_eomf(self):
        r"""Assemble a PySCF reference from the selected channel EO spaces.

        Args:
            None.

        Returns:
            object: Mean-field copy with a copied molecule and rounded physical EO
            electron counts. For UHF/UKS, :math:`N=N_\alpha+N_\beta` and
            :math:`2S_z=N_\alpha-N_\beta`; coefficients, occupations, and energies are
            alpha/beta tuples with potentially different orbital counts. Restricted
            outputs are single arrays; spin is the number of singly occupied MOs.
            ``mo_coeff0`` and ``mo_occ0`` retain the full reference arrays.
            Orbital coefficients remain in the original AO basis. This does not run SCF.

        Raises:
            RuntimeError: An orbital channel has no constructed EO basis.
        """
        eomf = {}
        for spin, channel in self.spin_channels.items():
            eomf[spin] = self.embedding[spin].get_eomf()

        if len(eomf) == 2: # unrestricted
            nelec = eomf['alpha']['nelectron'] + eomf['beta']['nelectron']
            mo_occ = (eomf['alpha']['mo_occ'], eomf['beta']['mo_occ'])
            mo_coeff = (eomf['alpha']['mo_coeff'], eomf['beta']['mo_coeff'])
            mo_energy = (eomf['alpha']['mo_energy'], eomf['beta']['mo_energy'])
            spin = eomf['alpha']['nelectron'] - eomf['beta']['nelectron']
        else: # restricted
            nelec = eomf['total']['nelectron']
            mo_occ = eomf['total']['mo_occ']
            mo_coeff = eomf['total']['mo_coeff']
            mo_energy = eomf['total']['mo_energy']
            spin = int(np.count_nonzero(mo_occ == 1))

        # Change mol class electrons
        mol = self.mf.mol.copy()
        mol.nelectron = nelec # change effective electrons
        mol.spin = spin

        # Build embedded mf class
        mf = self.mf
        eomf = mf.copy()
        eomf.mol = mol
        eomf.mo_coeff0 = mf.mo_coeff # full system orbitals for dft
        eomf.mo_occ0 = mf.mo_occ # full system orbitals for dft
        # use effective embedding orbitals
        eomf.mo_coeff = mo_coeff
        eomf.mo_occ = mo_occ
        eomf.mo_energy = mo_energy

        return eomf


    def get_embedding_energy(self):
        r"""Sum physical impurity electronic energies over orbital channels.

        .. math::

            E_{imp}=\sum_\sigma E_{imp,\sigma}.

        Args:
            None.

        Returns:
            float: Impurity electronic energy using original SCF occupations projected
            into each channel's EO basis. Nuclear repulsion is excluded. The contraction
            is HF-based; a full KS energy requires exchange-correlation energy accounting.

        Raises:
            RuntimeError: An orbital channel has no constructed EO basis.
        """
        energy = 0.
        hcore_ao = self.mf.get_hcore()
        for spin, channel in self.embedding.items():
            if channel.coeff_eo_in_ao is None:
                raise RuntimeError('Call emb_basis_dmet before get_embedding_energy.')
            energy_spin = get_embedding_energy(hcore_ao, channel.fock_ao,
                    channel.coeff_eo_in_ao, channel.physical_dm_eo,
                    channel.neo_imp
            )
            energy += energy_spin
        return energy


    def get_fragment_s2(self, coeff_imp=None):
        r"""Return impurity-only spin polarization and spin square, excluding the bath.

        Both spin densities use the same spatial basis. With
        :math:`N_\sigma=\mathrm{Tr}(d_\sigma)`, Wick factorization gives

        .. math::

            \langle S_{z,A}\rangle=\tfrac12(N_\alpha-N_\beta),

            \langle S_A^2\rangle=\tfrac34(N_\alpha+N_\beta)
              +\tfrac14(N_\alpha-N_\beta)^2
              -\tfrac14\mathrm{Tr}(d_\alpha^2+d_\beta^2)
              -\mathrm{Tr}(d_\alpha d_\beta).

        Alpha and beta spatial orbitals need not be mutually orthogonal. Only
        the common impurity basis satisfies :math:`C_A^\dagger S C_A=I`.
        Its spatial projector is :math:`\hat P_A=\sum_{a\in A}|a\rangle\langle a|`;
        its action on AO coefficient vectors is :math:`P_A=C_A C_A^\dagger S`.
        The impurity-projected alpha/beta orbital overlap is

        .. math::

            M^A_{ij}=C_{\alpha i}^\dagger S C_A C_A^\dagger S C_{\beta j},
            \qquad d_\sigma=C_A^\dagger S C_\sigma
                \mathrm{diag}(n_\sigma)C_\sigma^\dagger S C_A,

            \mathrm{Tr}(d_\alpha d_\beta)
                =\sum_{ij}n_{\alpha i}n_{\beta j}|M^A_{ij}|^2.

        Thus the raising/lowering contribution already contains the required
        orbital overlaps through the cross-density contraction:

        .. math::

            \tfrac12\langle S_A^+S_A^-+S_A^-S_A^+\rangle
                =\tfrac12(N_\alpha+N_\beta)-\mathrm{Tr}(d_\alpha d_\beta),

            \langle S_{z,A}^2\rangle
                =\tfrac14\left[(N_\alpha-N_\beta)^2+N_\alpha+N_\beta
                    -\mathrm{Tr}(d_\alpha^2+d_\beta^2)\right].

        Their sum is the local spin-square expectation above (with hbar=1).

        Restricted densities are split equally between spins here. Populations
        are not rounded, and local spin fluctuations are retained. This is a
        determinant/independent-particle estimate; correlated states require
        a two-particle density matrix.

        Args:
            coeff_imp (numpy.ndarray, optional): Shared orthonormal spatial
                impurity coefficients in AO coordinates, shape ``(nao, nimp)``.
                Defaults to the restricted impurity EOs, or the ALPHA impurity
                EOs for an unrestricted reference. Beta density is projected
                into that same spatial basis, not its independently localized
                impurity basis. An explicit basis defines a different region.

        Returns:
            tuple: ``(sz, s2)``, local polarization :math:`\langle S_{z,A}\rangle`
            in units of :math:`\hbar`, and :math:`\langle S_A^2\rangle` in units
            of :math:`\hbar^2`. Both use original SCF occupations. s2 can be
            nonzero in a globally closed-shell state. Extra orbitals affect bath selection,
            not the local-spin populations. Different localization/projector
            choices can give different local spins.

        Raises:
            RuntimeError: A channel's embedding basis has not been constructed.
            ValueError: The supplied common impurity basis is not orthonormal,
                or spin channels use incompatible AO overlap matrices.
        """
        for channel in self.embedding.values():
            if channel.coeff_eo_in_ao is None:
                raise RuntimeError('Call emb_basis_dmet before get_fragment_s2.')

        channel = self.embedding.get('total', self.embedding.get('alpha'))

        if 'total' in self.embedding:
            dm_alpha = channel.get_impurity_density(coeff_imp) * .5
            dm_beta = dm_alpha
        else:
            beta = self.embedding['beta']
            if not np.allclose(channel.ovlp_ao, beta.ovlp_ao):
                raise ValueError('Spin channels must use the same AO basis.')

            if coeff_imp is None:
                # cache alpha impurity coefficients for beta projection
                coeff_imp = channel.coeff_eo_in_ao[:, :channel.neo_imp]
                dm_alpha = channel.get_impurity_density()
            else:
                dm_alpha = channel.get_impurity_density(coeff_imp)
            dm_beta = beta.get_impurity_density(coeff_imp)

        nalpha, nbeta = np.trace(dm_alpha).real, np.trace(dm_beta).real
        sz = .5 * (nalpha - nbeta)
        # Equivalent: np.einsum('ij,ji->', dm_alpha, dm_beta)
        s2 = (.75 * (nalpha + nbeta) + .25 * (nalpha - nbeta)**2
              - .25 * np.trace(dm_alpha @ dm_alpha + dm_beta @ dm_beta)
              - np.trace(dm_alpha @ dm_beta))
        return float(sz), float(np.real(s2))



if __name__ == '__main__':
    from lumeq.utils.pyscf_helper import *
    from pyscf import scf

    #infile = '../samples/formic_acid_6_h2o.in'
    #infile = sys.argv[1]
    #parameters = parser(infile)
    #results = run_pyscf_final(parameters)
    #mol, mf = results['mol'], results['mf']

    #natm = mol.natm
    #frgm_idx = parameters[section_names[1]]['impurity'].split('-')
    #frgm_idx = [list(range(int(frgm_idx[0])-1, int(frgm_idx[1]))), [0]]
    #frgm_idx[1] = list(set(range(natm)) - set(frgm_idx[0]))
    ##print('frgm_idx:', frgm_idx)

    mol = gto.Mole()
    mol.build(
        atom = """
           O         0.4183272099    0.1671038379    0.1010361156
           H         0.8784893276   -0.0368266484    0.9330933285
           H        -0.3195928737    0.7774121014    0.3045311682
           O         3.0208058979    0.6163509592   -0.7203724735
           H         3.3050376617    1.4762564664   -1.0295977027
           H         2.0477791789    0.6319690134   -0.7090745711
           O         2.5143150551   -0.2441947452    1.8660305097
           H         2.8954132119   -1.0661605274    2.1741344071
           H         3.0247679096    0.0221180670    1.0833062723
        """,
        basis = '6-311++g**',
        verbose=0
    )

    frgm_idx = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.max_cycle = 100
    mf.conv_tol  = 1e-8
    mf.conv_tol_grad = 1e-8
    mf.kernel()
    print_matrix('mo_energy:', mf.mo_energy)

    embed = EmbeddingKS(mf, frgm_idx)
    # Test extra orbitals for a fixed impurity fragment.
    for i in range(-mol.nelectron//2+1, mf.mo_coeff.shape[0]-mol.nelectron//2):
        print('i:', i, mol.nelectron//2+i)
        embed.emb_basis_dmet(0, extra_orb=i)
        e = embed.get_embedding_energy()
        print('fragment:', 0, 'electronic energy:', e)

    # Test the sum over impurity fragments without extra orbitals.
    energy_ref = mf.energy_elec()[0]
    energy = 0.
    for ifrgm in range(len(frgm_idx)):
        embed.emb_basis_dmet(ifrgm, extra_orb=0)
        e = embed.get_embedding_energy()
        sz, s2 = embed.get_fragment_s2()
        print('fragment:', ifrgm, 'electronic energy:', e, '<Sz>:', sz, '<S^2>:', s2)
        energy += e
    print('reference energy: %12.8f embedding: %12.8f, difference: %12.8f'
          % (energy_ref, energy, energy-energy_ref))
