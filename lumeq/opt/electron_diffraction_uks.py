"""Static UKS starting point for electron diffraction wavepacket studies.

This module builds an augmented molecular UKS calculation in which one extra
electron is initialized as a Gaussian s wavepacket with a plane-wave phase,

    psi_in(r) = s_alpha(r - R0) exp[i k0 . (r - R0)].

The momentum phase is represented in the AO basis through PySCF's analytic
Fourier-transform Gaussian AO-pair integrals.  This is intended as a compact
starting point for static or subsequent real-time calculations, not as a full
many-channel scattering solver.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from lumeq import np

from pyscf import dft, gto
from pyscf.gto import ft_ao


ArrayLike = Sequence[float]


@dataclass
class IncomingElectronPacket:
    """Incoming electron represented by a Gaussian s packet.

    All lengths and momenta are in atomic units.  ``exponent`` is the Gaussian
    primitive exponent alpha in exp(-alpha |r-R0|^2).  ``momentum`` is the
    central incoming momentum k0.  ``spin`` chooses whether the extra electron
    is added to the alpha or beta UKS density.
    """

    center: ArrayLike
    momentum: ArrayLike
    exponent: float
    spin: str = "alpha"
    label: str = "X"
    phase_origin: Optional[ArrayLike] = None


@dataclass
class GhostBasisSpec:
    """Ghost Gaussian basis used as the scattering/continuum space.

    ``centers`` are in Bohr.  If ``exponents`` is not supplied, an
    even-tempered sequence from ``exponent_min`` to ``exponent_max`` is used.
    ``angular_momenta=(0, 1, 2)`` gives s, p, and d shells on every ghost
    center.
    """

    centers: Sequence[ArrayLike]
    angular_momenta: Sequence[int] = (0, 1, 2)
    exponents: Optional[Sequence[float]] = None
    n_exponents: int = 4
    exponent_min: float = 0.02
    exponent_max: float = 0.5
    label: str = "X"


def _as_vec3(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    return arr


def _spin_index(spin: str) -> int:
    key = spin.lower()
    if key in ("alpha", "a", "up", "+"):
        return 0
    if key in ("beta", "b", "down", "-"):
        return 1
    raise ValueError("spin must be 'alpha' or 'beta'")


def _new_spin(target_spin: int, packet_spin: str) -> int:
    return int(target_spin) + (1 if _spin_index(packet_spin) == 0 else -1)


def _single_s_basis(exponent: float):
    exponent = float(exponent)
    if exponent <= 0:
        raise ValueError("packet exponent must be positive")
    return [[0, [exponent, 1.0]]]


def even_tempered_exponents(n: int, exponent_min: float, exponent_max: float) -> np.ndarray:
    """Return an even-tempered exponent sequence."""
    n = int(n)
    if n <= 0:
        raise ValueError("n_exponents must be positive")
    exponent_min = float(exponent_min)
    exponent_max = float(exponent_max)
    if exponent_min <= 0 or exponent_max <= 0:
        raise ValueError("ghost exponents must be positive")
    if n == 1:
        return np.asarray([exponent_min])
    return np.geomspace(exponent_min, exponent_max, n)


def line_ghost_centers(start: ArrayLike, stop: ArrayLike, n: int) -> np.ndarray:
    """Return ``n`` ghost centers on a line from ``start`` to ``stop``."""
    n = int(n)
    if n <= 0:
        raise ValueError("number of ghost centers must be positive")
    start = _as_vec3(start, "start")
    stop = _as_vec3(stop, "stop")
    if n == 1:
        return start.reshape(1, 3)
    return np.linspace(start, stop, n)


def ghost_basis_shells(ghost_basis: GhostBasisSpec):
    """Build uncontracted Gaussian shells for every requested l/exponent."""
    if ghost_basis.exponents is None:
        exponents = even_tempered_exponents(
            ghost_basis.n_exponents,
            ghost_basis.exponent_min,
            ghost_basis.exponent_max,
        )
    else:
        exponents = np.asarray(ghost_basis.exponents, dtype=float)
        if exponents.ndim != 1 or len(exponents) == 0:
            raise ValueError("ghost_basis.exponents must be a nonempty 1D sequence")
        if np.any(exponents <= 0):
            raise ValueError("ghost_basis.exponents must be positive")

    shells = []
    for angular_momentum in ghost_basis.angular_momenta:
        l = int(angular_momentum)
        if l < 0:
            raise ValueError("angular momenta must be non-negative")
        for exponent in exponents:
            shells.append([l, [float(exponent), 1.0]])
    return shells


def _build_packet_probe_mol(packet: IncomingElectronPacket) -> gto.Mole:
    probe = gto.Mole()
    probe.atom = [(packet.label, _as_vec3(packet.center, "packet.center"))]
    probe.unit = "Bohr"
    probe.basis = {packet.label: _single_s_basis(packet.exponent)}
    probe.charge = 0
    probe.spin = 0
    probe.verbose = 0
    probe.build()
    return probe


def build_augmented_mol(
    target_mol: gto.Mole,
    packet: IncomingElectronPacket,
    ghost_basis: Optional[GhostBasisSpec] = None,
    charge: Optional[int] = None,
    spin: Optional[int] = None,
) -> Tuple[gto.Mole, Optional[int]]:
    """Return the target plus a ghost scattering basis.

    If ``ghost_basis`` is omitted, the old compact model is used: one ghost s
    shell at the packet center, and the returned shell index points to it.  If
    ``ghost_basis`` is supplied, all requested ghost centers/shells are added
    and the returned probe shell is ``None``; projection then uses a temporary
    packet probe function against the full augmented AO space.
    """
    atoms = [
        (target_mol.atom_symbol(ia), target_mol.atom_coord(ia))
        for ia in range(target_mol.natm)
    ]

    basis = copy.deepcopy(target_mol._basis)
    if ghost_basis is None:
        atoms.append((packet.label, _as_vec3(packet.center, "packet.center")))
        basis[packet.label] = _single_s_basis(packet.exponent)
        probe_shell = target_mol.nbas
    else:
        centers = np.asarray([_as_vec3(center, "ghost center") for center in ghost_basis.centers])
        if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
            raise ValueError("ghost_basis.centers must contain at least one 3-vector")
        for center in centers:
            atoms.append((ghost_basis.label, center))
        basis[ghost_basis.label] = ghost_basis_shells(ghost_basis)
        probe_shell = None

    if charge is None:
        charge = int(target_mol.charge) - 1
    if spin is None:
        spin = _new_spin(int(target_mol.spin), packet.spin)

    mol = gto.Mole()
    mol.atom = atoms
    mol.unit = "Bohr"
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.cart = target_mol.cart
    mol.verbose = target_mol.verbose
    mol.max_memory = target_mol.max_memory
    if getattr(target_mol, "_ecp", None):
        mol.ecp = copy.deepcopy(target_mol._ecp)
    mol.build()
    return mol, probe_shell


def project_packet_coeff(
    mol: gto.Mole,
    packet: IncomingElectronPacket,
    probe_shell: Optional[int] = None,
    linear_dep_threshold: float = 1.0e-10,
) -> np.ndarray:
    """Project ``s(r-R0) exp[i k0.(r-R0)]`` into the AO basis of ``mol``.

    PySCF ``ft_aopair`` evaluates int chi_i(r) chi_j(r) exp(-i G.r) dr.
    Therefore the overlap with exp(+i k0.r) is obtained with ``G=-k0``.
    """
    k0 = _as_vec3(packet.momentum, "packet.momentum")
    phase_origin = packet.center if packet.phase_origin is None else packet.phase_origin
    phase_origin = _as_vec3(phase_origin, "packet.phase_origin")

    # ft_aopair returns int chi_mu(r) chi_probe(r) exp(-i G.r) dr.
    # Use G=-k0 to project the incoming probe chi_probe(r) exp(+i k0.r)
    # onto every AO chi_mu in the augmented target+ghost basis.  If the main
    # molecule has no dedicated probe shell, a temporary packet probe molecule
    # is concatenated only for this projection integral.
    if probe_shell is None:
        probe_mol = _build_packet_probe_mol(packet)
        integral_mol = gto.conc_mol(mol, probe_mol)
        probe_shell = mol.nbas
    else:
        integral_mol = mol

    pair = ft_ao.ft_aopair(
        integral_mol,
        -k0.reshape(1, 3),
        shls_slice=(0, mol.nbas, probe_shell, probe_shell + 1),
        aosym="s1",
        return_complex=True,
    )
    overlap = pair[0, :, 0] * np.exp(-1j * np.dot(k0, phase_origin))

    s_ao = mol.intor("int1e_ovlp")
    evals = np.linalg.eigvalsh(s_ao)
    if evals[-1] < linear_dep_threshold:
        raise np.linalg.LinAlgError(
            "AO overlap has no usable eigenvalues for packet projection"
        )
    if evals[0] < linear_dep_threshold:
        evals, evecs = np.linalg.eigh(s_ao)
        keep = evals > linear_dep_threshold
        overlap_orth = evecs[:, keep].conj().T @ overlap
        coeff = evecs[:, keep] @ (overlap_orth / evals[keep])
    else:
        coeff = np.linalg.solve(s_ao, overlap)
    norm = np.vdot(coeff, s_ao @ coeff)
    if abs(norm) < linear_dep_threshold:
        raise ValueError("projected incoming packet has near-zero AO norm")
    coeff /= np.sqrt(norm)
    return coeff


def packet_density_matrix(
    mol: gto.Mole,
    packet_coeff: np.ndarray,
    spin: str = "alpha",
) -> np.ndarray:
    """Return a spin-AO density matrix for one incoming electron."""
    nao = mol.nao_nr()
    coeff = np.asarray(packet_coeff, dtype=complex)
    if coeff.shape != (nao,):
        raise ValueError(f"packet_coeff must have shape ({nao},)")

    dm = np.zeros((2, nao, nao), dtype=complex)
    dm[_spin_index(spin)] = np.outer(coeff, coeff.conj())
    return dm


def embed_target_density(target_dm: np.ndarray, augmented_nao: int) -> np.ndarray:
    """Embed a target-molecule AO density into the augmented AO space."""
    dm = np.asarray(target_dm)
    if dm.ndim == 2:
        dm = np.asarray((dm * 0.5, dm * 0.5))
    if dm.shape[0] != 2 or dm.shape[1] != dm.shape[2]:
        raise ValueError("target_dm must have shape (nao, nao) or (2, nao, nao)")

    out = np.zeros((2, augmented_nao, augmented_nao), dtype=complex)
    nao = dm.shape[-1]
    out[:, :nao, :nao] = dm
    return out


def make_initial_density(
    augmented_mol: gto.Mole,
    target_dm: np.ndarray,
    packet_coeff: np.ndarray,
    packet_spin: str,
) -> np.ndarray:
    """Combine the target UKS density and one incoming wavepacket electron."""
    dm0 = embed_target_density(target_dm, augmented_mol.nao_nr())
    dm0 += packet_density_matrix(augmented_mol, packet_coeff, packet_spin)
    return 0.5 * (dm0 + dm0.conj().transpose(0, 2, 1))


def run_target_uks(
    mol: gto.Mole,
    xc: str = "lda,vwn",
    dm0: Optional[np.ndarray] = None,
    grid_level: Optional[int] = None,
    **kwargs,
):
    """Run a target UKS calculation used to seed the augmented calculation."""
    mf = dft.UKS(mol)
    mf.xc = xc
    if grid_level is not None:
        mf.grids.level = grid_level
    for key, value in kwargs.items():
        setattr(mf, key, value)
    mf.kernel(dm0=dm0)
    return mf


class ElectronDiffractionUKS:
    """Static UKS setup for an explicit incoming electron wavepacket.

    The class owns the target UKS reference, augmented target+ghost molecule,
    projected incoming packet coefficients, initial complex spin density
    ``dm0``, and optional augmented static UKS object.
    """

    def __init__(
        self,
        target_mol: gto.Mole,
        packet: IncomingElectronPacket,
        ghost_basis: Optional[GhostBasisSpec] = None,
        xc: str = "lda,vwn",
        target_mf=None,
        run_scf: bool = True,
        target_grid_level: Optional[int] = None,
        augmented_grid_level: Optional[int] = None,
        max_cycle: int = 50,
        conv_tol: float = 1.0e-9,
        linear_dep_threshold: float = 1.0e-10,
    ):
        self.target_mol = target_mol
        self.packet = packet
        self.ghost_basis = ghost_basis
        self.xc = xc
        self.target_grid_level = target_grid_level
        self.augmented_grid_level = augmented_grid_level
        self.linear_dep_threshold = float(linear_dep_threshold)

        if target_mf is None:
            target_mf = run_target_uks(target_mol, xc=xc, grid_level=target_grid_level)
        self.target_mf = target_mf

        self.mol, self.probe_shell = build_augmented_mol(
            target_mol,
            packet,
            ghost_basis=ghost_basis,
        )
        self.packet_coeff = project_packet_coeff(
            self.mol,
            packet,
            self.probe_shell,
            linear_dep_threshold=self.linear_dep_threshold,
        )
        self.target_dm_ao = embed_target_density(self.target_mf.make_rdm1(), self.mol.nao_nr())
        self.dm0 = make_initial_density(
            self.mol,
            self.target_mf.make_rdm1(),
            self.packet_coeff,
            packet.spin,
        )

        self.mf = dft.UKS(self.mol)
        self.mf.xc = xc
        self.mf.max_cycle = max_cycle
        self.mf.conv_tol = conv_tol
        if augmented_grid_level is not None:
            self.mf.grids.level = augmented_grid_level

        if run_scf:
            self.kernel()

    def kernel(self):
        """Run augmented static UKS from the explicit packet density."""
        self.mf.kernel(dm0=self.dm0)
        return self.mf

    run = kernel

    def density(self, relaxed: bool = False) -> np.ndarray:
        """Return ``dm0`` or the relaxed augmented UKS density."""
        if relaxed:
            if self.mf.mo_coeff is None or self.mf.mo_occ is None:
                raise RuntimeError("augmented UKS has not been run")
            return self.mf.make_rdm1()
        return self.dm0

    def diffraction(
        self,
        q_vectors: np.ndarray,
        dm: Optional[np.ndarray] = None,
        relaxed: bool = False,
        coulomb_prefactor: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Compute first-Born diffraction from ``dm`` or this object's density."""
        if dm is None:
            dm = self.density(relaxed=relaxed)
        return first_born_diffraction(
            self.mol,
            dm,
            q_vectors,
            coulomb_prefactor=coulomb_prefactor,
        )

    def diffraction_from_angles(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        dm: Optional[np.ndarray] = None,
        relaxed: bool = False,
        coulomb_prefactor: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Compute diffraction on an elastic detector-angle grid."""
        q_vectors, shape = momentum_transfer_grid(self.packet.momentum, theta, phi)
        out = self.diffraction(
            q_vectors,
            dm=dm,
            relaxed=relaxed,
            coulomb_prefactor=coulomb_prefactor,
        )
        out["shape"] = shape
        return out

    def as_dict(self) -> Dict[str, object]:
        """Compatibility dictionary used by older calling code."""
        return {
            "mol": self.mol,
            "mf": self.mf,
            "dm0": self.dm0,
            "target_mf": self.target_mf,
            "target_dm_ao": self.target_dm_ao,
            "packet_coeff": self.packet_coeff,
            "probe_shell": self.probe_shell,
            "packet": self.packet,
            "ghost_basis": self.ghost_basis,
            "builder": self,
        }


def run_static_uks_with_packet(
    target_mol: gto.Mole,
    packet: IncomingElectronPacket,
    ghost_basis: Optional[GhostBasisSpec] = None,
    xc: str = "lda,vwn",
    target_mf=None,
    run_scf: bool = True,
    target_grid_level: Optional[int] = None,
    augmented_grid_level: Optional[int] = None,
    max_cycle: int = 50,
    conv_tol: float = 1.0e-9,
) -> Dict[str, object]:
    """Build and optionally run a static UKS calculation with the packet.

    The returned ``dm0`` is the explicit incoming-electron starting density.
    For diffraction from the initial packet state, use ``dm0``.  For diffraction
    from the relaxed static UKS state, use ``mf.make_rdm1()`` after convergence.
    """
    builder = ElectronDiffractionUKS(
        target_mol,
        packet,
        ghost_basis=ghost_basis,
        target_mf=target_mf,
        run_scf=run_scf,
        target_grid_level=target_grid_level,
        augmented_grid_level=augmented_grid_level,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        xc=xc,
    )
    return builder.as_dict()


def electron_density_ft(mol: gto.Mole, dm: np.ndarray, q_vectors: np.ndarray) -> np.ndarray:
    """Fourier transform of the electronic density at momentum transfers q."""
    q_vectors = np.atleast_2d(np.asarray(q_vectors, dtype=float))
    dm = np.asarray(dm)
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    rho_ao = ft_ao.ft_aopair(mol, q_vectors, aosym="s1", return_complex=True)
    return np.einsum("qij,ji->q", rho_ao, dm)


def nuclear_charge_ft(mol: gto.Mole, q_vectors: np.ndarray) -> np.ndarray:
    """Fourier transform of point nuclear charge density."""
    q_vectors = np.atleast_2d(np.asarray(q_vectors, dtype=float))
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    phase = np.exp(-1j * q_vectors @ coords.T)
    return phase @ charges


def first_born_diffraction(
    mol: gto.Mole,
    dm: np.ndarray,
    q_vectors: np.ndarray,
    coulomb_prefactor: bool = True,
    q_eps: float = 1.0e-12,
) -> Dict[str, np.ndarray]:
    """Return a simple first-Born electron-diffraction amplitude.

    The amplitude is proportional to the Fourier transform of the electrostatic
    charge density, ``Z(q) - rho(q)``.  With ``coulomb_prefactor=True`` it is
    multiplied by ``4*pi/|q|^2`` away from q=0.  Overall physical constants are
    intentionally not included.
    """
    q_vectors = np.atleast_2d(np.asarray(q_vectors, dtype=float))
    rho_e = electron_density_ft(mol, dm, q_vectors)
    rho_n = nuclear_charge_ft(mol, q_vectors)
    charge_ft = rho_n - rho_e
    amplitude = charge_ft.copy()
    if coulomb_prefactor:
        q2 = np.einsum("qx,qx->q", q_vectors, q_vectors)
        scale = np.zeros_like(q2, dtype=float)
        mask = q2 > q_eps
        scale[mask] = 4.0 * np.pi / q2[mask]
        amplitude *= scale
    return {
        "q": q_vectors,
        "electron_density_ft": rho_e,
        "nuclear_charge_ft": rho_n,
        "charge_density_ft": charge_ft,
        "amplitude": amplitude,
        "intensity": np.abs(amplitude) ** 2,
    }


def momentum_transfer_grid(
    incoming_momentum: ArrayLike,
    theta: np.ndarray,
    phi: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Build elastic-scattering q vectors for detector angles.

    ``theta`` is the scattering angle away from the incoming direction and
    ``phi`` is the azimuth.  The returned q vectors are flattened; ``shape`` can
    be used to reshape diffraction intensities back to detector coordinates.

    The elastic convention used here is

        |k_out| = |k_in|
        q = k_out - k_in
        k_out(theta, phi) = |k_in| [
            cos(theta) ez
            + sin(theta) (cos(phi) ex + sin(phi) ey)
        ]

    where ``ez`` is the incoming-beam direction and ``ex, ey`` span the
    detector plane perpendicular to it.  With this sign convention, the
    diffraction routines evaluate Fourier factors ``exp(-i q.r)``.
    """
    k_in = _as_vec3(incoming_momentum, "incoming_momentum")
    k_norm = np.linalg.norm(k_in)
    if k_norm < 1.0e-14:
        raise ValueError("incoming momentum must be nonzero")
    ez = k_in / k_norm
    trial = np.asarray((1.0, 0.0, 0.0))
    if abs(np.dot(trial, ez)) > 0.9:
        trial = np.asarray((0.0, 1.0, 0.0))
    ex = trial - np.dot(trial, ez) * ez
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)

    th, ph = np.meshgrid(np.asarray(theta, dtype=float), np.asarray(phi, dtype=float), indexing="ij")
    k_out = k_norm * (
        np.cos(th)[..., None] * ez
        + np.sin(th)[..., None] * (np.cos(ph)[..., None] * ex + np.sin(ph)[..., None] * ey)
    )
    q = k_out.reshape(-1, 3) - k_in
    return q, th.shape


if __name__ == "__main__":
    mol = gto.M(
        atom="H 0 0 -0.37; H 0 0 0.37",
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        verbose=0,
    )
    packet = IncomingElectronPacket(
        center=(0.0, 0.0, -8.0),
        momentum=(0.0, 0.0, 2.0),
        exponent=0.05,
        spin="alpha",
    )
    ghost_basis = GhostBasisSpec(
        centers=line_ghost_centers((0.0, 0.0, -8.0), (0.0, 0.0, 8.0), 3),
        angular_momenta=(0, 1, 2),
        n_exponents=2,
        exponent_min=0.03,
        exponent_max=0.15,
    )
    diffraction = ElectronDiffractionUKS(
        mol,
        packet,
        ghost_basis=ghost_basis,
        xc="lda,vwn",
        run_scf=False,
        target_grid_level=1,
        augmented_grid_level=1,
        max_cycle=20,
    )
    dm = diffraction.dm0

    theta = np.linspace(0.01, 0.25, 16)
    phi = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    pattern = diffraction.diffraction_from_angles(theta, phi, dm=dm)
    shape = pattern["shape"]
    intensity = pattern["intensity"].reshape(shape)

    s_ao = diffraction.mol.intor("int1e_ovlp")
    nelec = np.einsum("ij,ji->", dm[0] + dm[1], s_ao).real
    print("augmented nao:", diffraction.mol.nao_nr())
    print("augmented nelec:", nelec)
    print("static augmented SCF run:", False)
    print("diffraction grid shape:", intensity.shape)
    print("intensity min/max:", float(intensity.min()), float(intensity.max()))
