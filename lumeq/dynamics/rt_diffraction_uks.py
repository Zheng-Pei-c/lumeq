"""Real-time UKS density-matrix dynamics for electron diffraction.

The initial state is a target UKS density plus one explicit incoming electron
wavepacket constructed as a ghost-center Gaussian s orbital with a plane-wave
phase.  The packet construction and Fourier-space diffraction helpers are
reused from :mod:`lumeq.opt.electron_diffraction_uks`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from lumeq import np
from lumeq.opt.electron_diffraction_uks import (
    ElectronDiffractionUKS,
    line_ghost_centers,
    GhostBasisSpec,
    IncomingElectronPacket,
)

from scipy import linalg
from pyscf import dft, gto
from pyscf.lib import logger


ArrayLike = Sequence[float]
SpinMatrices = Tuple[np.ndarray, np.ndarray]


def _as_spin_matrices(value: np.ndarray, name: str) -> SpinMatrices:
    arr = np.asarray(value)
    if arr.shape[0] != 2:
        raise ValueError(f"{name} must contain alpha and beta matrices")
    return arr[0], arr[1]


def _hermitize(mat: np.ndarray) -> np.ndarray:
    return (mat + mat.conj().T) * 0.5


def _hermitize_spin(dm: np.ndarray) -> np.ndarray:
    dma, dmb = _as_spin_matrices(dm, "dm")
    return np.asarray((_hermitize(dma), _hermitize(dmb)))


def _density_error(dm1: np.ndarray, dm0: np.ndarray) -> float:
    diff = np.asarray(dm1) - np.asarray(dm0)
    return float(np.linalg.norm(diff.reshape(-1)))


def _as_vec3(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    return arr


def _default_cap_radius(target_mol: gto.Mole, packet: IncomingElectronPacket, center: np.ndarray) -> float:
    target_coords = target_mol.atom_coords()
    target_radius = 0.0
    if len(target_coords):
        target_radius = float(np.max(np.linalg.norm(target_coords - center, axis=1)))
    packet_center = _as_vec3(packet.center, "packet.center")
    packet_radius = float(np.linalg.norm(packet_center - center))
    packet_extent = 2.0 / np.sqrt(float(packet.exponent))
    return max(target_radius + 6.0, packet_radius + packet_extent)


def radial_cap_values(
    coords: np.ndarray,
    strength: float,
    radius: float,
    width: float = 5.0,
    power: int = 2,
    center: ArrayLike = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Positive radial CAP profile Gamma(r).

    The profile is zero inside ``radius`` and rises as

        Gamma(r) = strength * ((|r-center| - radius) / width)**power

    outside the absorber onset.  It enters the effective Hamiltonian as
    ``H_eff = F - i Gamma``.
    """
    if strength < 0:
        raise ValueError("cap strength must be non-negative")
    if radius < 0:
        raise ValueError("cap radius must be non-negative")
    if width <= 0:
        raise ValueError("cap width must be positive")
    if power <= 0:
        raise ValueError("cap power must be positive")
    center = _as_vec3(center, "cap center")
    r = np.linalg.norm(np.asarray(coords) - center, axis=1)
    x = np.maximum(r - float(radius), 0.0) / float(width)
    return float(strength) * x**int(power)


def build_radial_cap_ao(
    mol: gto.Mole,
    strength: float,
    radius: float,
    width: float = 5.0,
    power: int = 2,
    center: ArrayLike = (0.0, 0.0, 0.0),
    grid_level: Optional[int] = None,
    max_memory: int = 2000,
) -> np.ndarray:
    """Build the AO matrix of a radial CAP with DFT numerical grids."""
    nao = mol.nao_nr()
    if strength == 0:
        return np.zeros((nao, nao), dtype=complex)

    grids = dft.gen_grid.Grids(mol)
    if grid_level is not None:
        grids.level = grid_level
    grids.build(with_non0tab=True)

    ni = dft.numint.NumInt()
    cap_ao = np.zeros((nao, nao), dtype=complex)
    for ao, _mask, weight, coords in ni.block_loop(mol, grids, nao, 0, max_memory=max_memory):
        gamma = radial_cap_values(
            coords,
            strength=strength,
            radius=radius,
            width=width,
            power=power,
            center=center,
        )
        cap_ao += np.einsum("gi,g,gj->ij", ao.conj(), weight * gamma, ao)
    return _hermitize(cap_ao)


class RTDiffractionUKS:
    """Real-time UKS propagation from an explicit incoming electron packet.

    Parameters
    ----------
    target_mol
        PySCF molecule for the target before adding the incoming electron.
    packet
        Incoming Gaussian s wavepacket with plane-wave phase.
    ghost_basis
        Optional multi-center ghost basis used as the continuum/scattering
        space.  If omitted, the compact one-ghost-s starting model is used.
    target_mf
        Optional converged target UKS object.  If omitted, a target UKS
        calculation is run before constructing the augmented packet density.
    dt
        Time step in atomic units.
    xc
        Exchange-correlation functional for the target and augmented UKS Fock.
    propagator
        ``"pc"`` for self-consistent midpoint predictor-corrector or
        ``"etrs"`` for a one-predictor midpoint step.
    cap_strength
        Strength of the complex absorbing potential.  ``0`` disables the CAP.
        When enabled, the propagation uses ``H_eff = F - i Gamma`` and electron
        number decreases as outgoing density reaches the absorber.

    Notes
    -----
    The propagated density is initialized from ``dm0`` built by
    ``run_static_uks_with_packet(..., run_scf=False)``.  No static augmented SCF
    relaxation is performed by default, so the incoming momentum phase is kept
    at t=0.  The fixed orthonormal propagation basis is obtained by
    diagonalizing the initial spin Fock matrix in the augmented AO space.
    """

    def __init__(
        self,
        target_mol: gto.Mole,
        packet: IncomingElectronPacket,
        ghost_basis: Optional[GhostBasisSpec] = None,
        target_mf=None,
        dt: float = 0.05,
        xc: str = "lda,vwn",
        target_grid_level: Optional[int] = None,
        augmented_grid_level: Optional[int] = None,
        propagator: str = "pc",
        conv_tol: float = 1.0e-8,
        max_cycle: int = 30,
        cap_strength: float = 0.0,
        cap_radius: Optional[float] = None,
        cap_width: float = 5.0,
        cap_power: int = 2,
        cap_center: Optional[ArrayLike] = None,
        cap_grid_level: Optional[int] = None,
        verbose: Optional[int] = None,
    ):
        self.packet = packet
        self.ghost_basis = ghost_basis
        self.dt = float(dt)
        self.xc = xc
        self.propagator = propagator.lower()
        self.conv_tol = float(conv_tol)
        self.max_cycle = int(max_cycle)
        self.cap_strength = float(cap_strength)
        self.cap_width = float(cap_width)
        self.cap_power = int(cap_power)

        self.static = ElectronDiffractionUKS(
            target_mol,
            packet,
            ghost_basis=ghost_basis,
            xc=xc,
            target_mf=target_mf,
            run_scf=False,
            target_grid_level=target_grid_level,
            augmented_grid_level=augmented_grid_level,
        )
        self.target_mf = self.static.target_mf
        self.mol = self.static.mol
        self.mf = self.static.mf
        self.dm0 = _hermitize_spin(self.static.dm0)
        self.packet_coeff = np.asarray(self.static.packet_coeff, dtype=complex)
        self.probe_shell = self.static.probe_shell

        self.verbose = self.mf.verbose if verbose is None else verbose
        self.log = logger.new_logger(self.mf, self.verbose)
        self.mf.verbose = self.verbose

        self.nao = self.mol.nao_nr()
        self.nelec = tuple(self.mol.nelec)
        self.hcore_ao = np.asarray(self.mf.get_hcore())
        self.ovlp_ao = np.asarray(self.mf.get_ovlp())
        self.target_dm_ao = self.static.target_dm_ao
        self.nuclear_charge = float(np.sum(self.mol.atom_charges()))

        if cap_center is None:
            cap_center = np.mean(target_mol.atom_coords(), axis=0)
        self.cap_center = _as_vec3(cap_center, "cap_center")
        if cap_radius is None:
            cap_radius = _default_cap_radius(target_mol, packet, self.cap_center)
        self.cap_radius = float(cap_radius)
        self.cap_ao = build_radial_cap_ao(
            self.mol,
            strength=self.cap_strength,
            radius=self.cap_radius,
            width=self.cap_width,
            power=self.cap_power,
            center=self.cap_center,
            grid_level=cap_grid_level if cap_grid_level is not None else augmented_grid_level,
            max_memory=self.mf.max_memory,
        )

        self.mo_coeff, self.mo_energy = self._initial_reference_orbitals(self.dm0)
        self.mo_occ = self._occupations_from_nelec()
        self.mf.mo_coeff = self.mo_coeff
        self.mf.mo_energy = self.mo_energy
        self.mf.mo_occ = self.mo_occ
        self.cap_mo = self._ao_to_mo_spin(self.cap_ao)

        self.time = 0.0
        self.p_mo = self.ao_density_to_mo(self.dm0)
        self.initial_electron_number = float(np.sum(self.electron_number(dm=self.dm0).real))
        self.history: Dict[str, List[np.ndarray]] = {}
        self.reset_history()

    def reset_history(self) -> None:
        self.history = {
            "time": [],
            "energy": [],
            "nelec": [],
            "total_nelec": [],
            "absorbed_nelec": [],
            "charge": [],
            "packet_population": [],
            "pc_error": [],
            "pc_cycle": [],
        }

    def kernel(
        self,
        nsteps: int,
        save_density: bool = False,
        callback: Optional[Callable[["RTDiffractionUKS"], None]] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """Propagate for ``nsteps`` and return the recorded history."""
        self.record(save_density=save_density)
        for _ in range(int(nsteps)):
            self.step()
            self.record(save_density=save_density)
            if callback is not None:
                callback(self)
        return self.history

    run = kernel

    def step(self) -> np.ndarray:
        """Advance the spin density matrices by one time step."""
        if self.propagator in ("pc", "predictor-corrector", "midpoint-pc"):
            p_new, error, cycle = self._step_predictor_corrector()
        elif self.propagator in ("etrs", "midpoint"):
            p_new, error, cycle = self._step_etrs()
        else:
            raise ValueError(f"unknown propagator {self.propagator!r}")

        self.p_mo = _hermitize_spin(p_new)
        self.time += self.dt
        self._last_pc_error = error
        self._last_pc_cycle = cycle
        return self.p_mo

    def record(self, save_density: bool = False) -> None:
        dm = self.make_rdm1()
        nelec = np.asarray(self.electron_number(dm=dm)).real
        total_nelec = float(np.sum(nelec))
        self.history["time"].append(float(self.time))
        self.history["energy"].append(float(self.energy_tot(dm=dm).real))
        self.history["nelec"].append(nelec)
        self.history["total_nelec"].append(total_nelec)
        self.history["absorbed_nelec"].append(self.initial_electron_number - total_nelec)
        self.history["charge"].append(self.nuclear_charge - total_nelec)
        self.history["packet_population"].append(float(self.packet_population(dm=dm).real))
        self.history["pc_error"].append(float(getattr(self, "_last_pc_error", 0.0)))
        self.history["pc_cycle"].append(int(getattr(self, "_last_pc_cycle", 0)))
        if save_density:
            self.history.setdefault("dm", []).append(dm.copy())
            self.history.setdefault("p_mo", []).append(self.p_mo.copy())

    def make_rdm1(self, p_mo: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the current alpha/beta AO density matrix."""
        if p_mo is None:
            p_mo = self.p_mo
        return self.mo_density_to_ao(p_mo)

    def mo_density_to_ao(self, p_mo: np.ndarray) -> np.ndarray:
        p_a, p_b = _as_spin_matrices(p_mo, "p_mo")
        dm_a = self.mo_coeff[0] @ p_a @ self.mo_coeff[0].conj().T
        dm_b = self.mo_coeff[1] @ p_b @ self.mo_coeff[1].conj().T
        return _hermitize_spin(np.asarray((dm_a, dm_b)))

    def ao_density_to_mo(self, dm_ao: np.ndarray) -> np.ndarray:
        dm_a, dm_b = _as_spin_matrices(dm_ao, "dm_ao")
        s = self.ovlp_ao
        p_a = self.mo_coeff[0].conj().T @ s @ dm_a @ s @ self.mo_coeff[0]
        p_b = self.mo_coeff[1].conj().T @ s @ dm_b @ s @ self.mo_coeff[1]
        return _hermitize_spin(np.asarray((p_a, p_b)))

    def fock_mo(self, p_mo: Optional[np.ndarray] = None) -> np.ndarray:
        dm = self.make_rdm1(p_mo)
        vhf = self.mf.get_veff(self.mol, dm, hermi=1)
        fock_ao = np.asarray(self.hcore_ao + vhf)
        return self._ao_to_mo_spin(fock_ao)

    def energy_tot(self, dm: Optional[np.ndarray] = None) -> float:
        if dm is None:
            dm = self.make_rdm1()
        dm = _hermitize_spin(dm)
        vhf = self.mf.get_veff(self.mol, dm, hermi=1)
        e_elec = self.mf.energy_elec(dm=dm, h1e=self.hcore_ao, vhf=vhf)[0]
        return e_elec + self.mf.energy_nuc()

    def electron_number(self, dm: Optional[np.ndarray] = None) -> np.ndarray:
        if dm is None:
            dm = self.make_rdm1()
        dm_a, dm_b = _as_spin_matrices(dm, "dm")
        s = self.ovlp_ao
        return np.asarray((np.einsum("ij,ji->", dm_a, s), np.einsum("ij,ji->", dm_b, s)))

    def packet_population(
        self,
        dm: Optional[np.ndarray] = None,
        spin: Optional[str] = None,
        excess: bool = True,
    ) -> complex:
        """Occupation of the initial packet orbital.

        With ``excess=True`` the embedded target density is subtracted first,
        so the value tracks the incoming electron contribution rather than the
        background target-electron overlap with the same orbital.
        """
        if dm is None:
            dm = self.make_rdm1()
        if spin is None:
            spin = self.packet.spin
        spin_idx = 0 if spin.lower() in ("alpha", "a", "up", "+") else 1
        if excess:
            dm = dm - self.target_dm_ao
        s = self.ovlp_ao
        c = self.packet_coeff
        return np.vdot(c, s @ dm[spin_idx] @ s @ c)

    def diffraction(
        self,
        q_vectors: np.ndarray,
        dm: Optional[np.ndarray] = None,
        coulomb_prefactor: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Compute a first-Born diffraction snapshot from the current density."""
        if dm is None:
            dm = self.make_rdm1()
        return self.static.diffraction(
            q_vectors=q_vectors,
            dm=dm,
            coulomb_prefactor=coulomb_prefactor,
        )

    def diffraction_from_angles(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        dm: Optional[np.ndarray] = None,
        coulomb_prefactor: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Compute diffraction on an elastic detector-angle grid."""
        if dm is None:
            dm = self.make_rdm1()
        return self.static.diffraction_from_angles(
            theta,
            phi,
            dm=dm,
            coulomb_prefactor=coulomb_prefactor,
        )

    def _step_etrs(self) -> Tuple[np.ndarray, float, int]:
        f0 = self.fock_mo(self.p_mo)
        p_pred = self._propagate_with_fock(self.p_mo, f0, self.dt)
        f1 = self.fock_mo(p_pred)
        f_mid = (f0 + f1) * 0.5
        p_new = self._propagate_with_fock(self.p_mo, f_mid, self.dt)
        return p_new, _density_error(p_new, p_pred), 1

    def _step_predictor_corrector(self) -> Tuple[np.ndarray, float, int]:
        f0 = self.fock_mo(self.p_mo)
        p_old = self._propagate_with_fock(self.p_mo, f0, self.dt)
        error = np.inf

        for cycle in range(1, self.max_cycle + 1):
            f1 = self.fock_mo(p_old)
            f_mid = (f0 + f1) * 0.5
            p_new = self._propagate_with_fock(self.p_mo, f_mid, self.dt)
            error = _density_error(p_new, p_old)
            if error < self.conv_tol:
                return p_new, error, cycle
            p_old = p_new

        self.log.warn("RTDiffractionUKS predictor-corrector did not converge; error=%g", error)
        return p_old, error, self.max_cycle

    def _propagate_with_fock(self, p_mo: np.ndarray, fock_mo: np.ndarray, dt: float) -> np.ndarray:
        p_a, p_b = _as_spin_matrices(p_mo, "p_mo")
        f_a, f_b = _as_spin_matrices(fock_mo, "fock_mo")
        out = []
        cap_a, cap_b = _as_spin_matrices(self.cap_mo, "cap_mo")
        for p, f, gamma in ((p_a, f_a, cap_a), (p_b, f_b, cap_b)):
            h_eff = _hermitize(f) - 1j * _hermitize(gamma)
            u = linalg.expm(-1j * h_eff * dt)
            out.append(u @ p @ u.conj().T)
        return _hermitize_spin(np.asarray(out))

    def _ao_to_mo_spin(self, h_ao: np.ndarray) -> np.ndarray:
        h = np.asarray(h_ao)
        if h.ndim == 2:
            h = np.asarray((h, h))
        h_a, h_b = _as_spin_matrices(h, "h_ao")
        return np.asarray(
            (
                self.mo_coeff[0].conj().T @ h_a @ self.mo_coeff[0],
                self.mo_coeff[1].conj().T @ h_b @ self.mo_coeff[1],
            )
        )

    def _initial_reference_orbitals(self, dm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vhf = self.mf.get_veff(self.mol, dm, hermi=1)
        fock_ao = np.asarray(self.hcore_ao + vhf)
        mo_energy = []
        mo_coeff = []
        for spin in range(2):
            e, c = linalg.eigh(_hermitize(fock_ao[spin]), self.ovlp_ao)
            mo_energy.append(e)
            mo_coeff.append(c)
        return np.asarray(mo_coeff), np.asarray(mo_energy)

    def _occupations_from_nelec(self) -> np.ndarray:
        occ = np.zeros((2, self.nao))
        occ[0, : self.nelec[0]] = 1.0
        occ[1, : self.nelec[1]] = 1.0
        return occ


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
    rt = RTDiffractionUKS(
        mol,
        packet,
        ghost_basis=ghost_basis,
        dt=0.05,
        xc="lda,vwn",
        target_grid_level=1,
        augmented_grid_level=1,
        cap_strength=0.05,
        cap_radius=9.0,
        cap_width=3.0,
        cap_grid_level=1,
        conv_tol=1.0e-10,
        max_cycle=20,
        verbose=0,
    )
    hist = rt.kernel(5)
    theta = np.linspace(0.01, 0.2, 8)
    phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    pattern = rt.diffraction_from_angles(theta, phi)
    intensity = pattern["intensity"].reshape(pattern["shape"])

    print("final time:", hist["time"][-1])
    print("final energy:", hist["energy"][-1])
    print("final nelec:", hist["nelec"][-1])
    print("absorbed electrons:", hist["absorbed_nelec"][-1])
    print("charge expectation:", hist["charge"][-1])
    print("packet population:", hist["packet_population"][-1])
    print("diffraction grid shape:", intensity.shape)
    print("intensity min/max:", float(intensity.min()), float(intensity.max()))
