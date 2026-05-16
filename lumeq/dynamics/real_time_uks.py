"""Real-time unrestricted Kohn-Sham TDDFT utilities.

The implementation follows the density-matrix RT-TDDFT equation used in
Yang, Pei, Deng et al., Phys. Chem. Chem. Phys. 2020, 22, 26838:

    i dP(t) / dt = [F[P(t)], P(t)]

The propagated density matrices are represented in the orthonormal canonical
MO basis of the unperturbed UKS reference.  AO density matrices are rebuilt
only when PySCF is asked for a new UKS Fock matrix.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from lumeq import np

from scipy import linalg
from pyscf import dft
from pyscf.dft import uks
from pyscf.lib import logger
from pyscf.tdscf import uks as tduks


ArrayLike = Union[np.ndarray, Sequence[float]]
SpinMatrices = Tuple[np.ndarray, np.ndarray]
PerturbationLike = Union[
    ArrayLike,
    Dict[str, Any],
    Callable[[float], Union[ArrayLike, Dict[str, Any]]],
]


def _as_spin_matrices(value: ArrayLike, name: str) -> SpinMatrices:
    arr = np.asarray(value)
    if arr.shape[0] != 2:
        raise ValueError(f"{name} must contain alpha and beta matrices")
    return arr[0], arr[1]


def _hermitize(mat: np.ndarray) -> np.ndarray:
    return (mat + mat.conj().T) * 0.5


def _hermitize_spin(dm: ArrayLike) -> np.ndarray:
    dma, dmb = _as_spin_matrices(dm, "dm")
    return np.asarray((_hermitize(dma), _hermitize(dmb)))


def _call_if_needed(value, time: Optional[float]):
    if callable(value):
        if time is None:
            raise ValueError("time-dependent perturbation requires a time value")
        return value(time)
    return value


def _density_error(dm1: np.ndarray, dm0: np.ndarray) -> float:
    diff = np.asarray(dm1) - np.asarray(dm0)
    return float(np.linalg.norm(diff.reshape(-1)))


def _unit_vector(vec: ArrayLike, name: str) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1.0e-14:
        raise ValueError(f"{name} must be a nonzero vector")
    return vec / norm


def _default_perpendicular(vec: np.ndarray) -> np.ndarray:
    trial = np.asarray((1.0, 0.0, 0.0))
    if abs(np.dot(vec, trial)) > 0.9:
        trial = np.asarray((0.0, 1.0, 0.0))
    perp = trial - np.dot(trial, vec) * vec
    return _unit_vector(perp, "minor_axis")


def _resolve_time_window(
    t_start: float,
    t_end: Optional[float],
    duration: Optional[float],
) -> Tuple[float, float]:
    start = float(t_start)
    if duration is not None:
        duration = float(duration)
        if duration <= 0:
            raise ValueError("duration must be positive")
        end = start + duration
        if t_end is not None and not np.isclose(float(t_end), end):
            raise ValueError("t_end must equal t_start + duration when both are supplied")
    elif t_end is None:
        raise ValueError("either t_end or duration must be supplied")
    else:
        end = float(t_end)

    if end <= start:
        raise ValueError("t_end must be greater than t_start")
    return start, end


def _resolve_gaussian_laser_window(
    center: Optional[float],
    sigma: Optional[float],
    t_start: Optional[float],
    t_end: Optional[float],
    duration: Optional[float],
    cutoff: float,
) -> Tuple[float, float, float, float]:
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    if duration is not None:
        start = 0.0 if t_start is None else float(t_start)
        start, end = _resolve_time_window(start, t_end, duration)
    else:
        start = None if t_start is None else float(t_start)
        end = None if t_end is None else float(t_end)

    if center is None:
        if start is None or end is None:
            raise ValueError("center is required unless duration or t_start/t_end define the pulse window")
        center = 0.5 * (start + end)
    else:
        center = float(center)

    if sigma is None:
        if start is None or end is None:
            raise ValueError("sigma is required unless duration or t_start/t_end define the pulse window")
        sigma = (end - start) / (2.0 * cutoff)
    else:
        sigma = float(sigma)

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if start is None:
        start = center - cutoff * sigma
    if end is None:
        end = center + cutoff * sigma
    if end <= start:
        raise ValueError("t_end must be greater than t_start")

    return center, sigma, start, end


def gaussian_packet_laser_field(
    amplitude: float,
    omega: float,
    center: Optional[float] = None,
    sigma: Optional[float] = None,
    major_axis: ArrayLike = (0.0, 0.0, 1.0),
    minor_axis: Optional[ArrayLike] = None,
    ellipticity: float = 0.0,
    phase: float = 0.0,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    duration: Optional[float] = None,
    cutoff: float = 4.0,
    normalize_intensity: bool = False,
) -> Callable[[float], np.ndarray]:
    """Return a finite-time elliptically polarized Gaussian laser field.

    The returned callable evaluates

        E(t) = A exp[-(t-center)^2/(2 sigma^2)]
               [e1 cos(omega (t-center) + phase)
                + ellipticity e2 sin(omega (t-center) + phase)]

    and returns zero outside the finite window.  If ``t_start`` and ``t_end``
    are not supplied, the window is ``center +/- cutoff * sigma``.
    Alternatively, pass ``duration`` and optional ``t_start``; then ``t_end``
    is ``t_start + duration``.  If ``center`` or ``sigma`` is omitted, the
    center is placed at the middle of the window and ``sigma`` is chosen so
    the window spans ``center +/- cutoff * sigma``.
    ``ellipticity=0`` gives linear polarization and ``ellipticity=1`` gives
    circular polarization when the two axes are orthonormal.
    """
    center, sigma, start, end = _resolve_gaussian_laser_window(
        center=center,
        sigma=sigma,
        t_start=t_start,
        t_end=t_end,
        duration=duration,
        cutoff=cutoff,
    )

    e1 = _unit_vector(major_axis, "major_axis")
    if minor_axis is None:
        e2 = _default_perpendicular(e1)
    else:
        e2 = np.asarray(minor_axis, dtype=float)
        e2 = e2 - np.dot(e2, e1) * e1
        e2 = _unit_vector(e2, "minor_axis")

    eta = float(ellipticity)
    amp = float(amplitude)
    if normalize_intensity:
        amp /= np.sqrt(1.0 + eta * eta)

    def field(time: float) -> np.ndarray:
        t = float(time)
        if t < start or t > end:
            return np.zeros(3)
        tau = t - center
        envelope = np.exp(-0.5 * (tau / sigma) ** 2)
        angle = omega * tau + phase
        return amp * envelope * (np.cos(angle) * e1 + eta * np.sin(angle) * e2)

    field.amplitude = amp
    field.omega = float(omega)
    field.center = float(center)
    field.sigma = float(sigma)
    field.t_start = start
    field.t_end = end
    field.duration = end - start
    field.major_axis = e1
    field.minor_axis = e2
    field.ellipticity = eta
    return field


class RTUKS:
    """Real-time TDDFT propagator for a PySCF :class:`dft.UKS` reference.

    Parameters
    ----------
    mf
        A converged PySCF unrestricted Kohn-Sham object.
    dt
        Time step in atomic units.
    field
        Backward-compatible external electric field used during propagation.
        It can be either a 3-vector or a callable ``field(t) -> (Ex, Ey, Ez)``.
    external_potential
        Optional general one-electron perturbation used during propagation.
        Supported forms are a 3-vector electric field, an AO matrix, a spin-AO
        matrix, a callable returning one of these, or a dictionary such as
        ``{"type": "magnetic_field", "field": B}`` or
        ``{"type": "one_electron", "matrix": h1ao}``.
    origin
        Gauge origin for electric-dipole integrals.
    propagator
        ``"pc"`` for self-consistent midpoint predictor-corrector or
        ``"etrs"`` for one predictor midpoint step.
    conv_tol
        Predictor-corrector density convergence threshold.
    max_cycle
        Maximum predictor-corrector iterations per time step.
    """

    def __init__(
        self,
        mf: uks.UKS,
        dt: float = 0.05,
        field: Optional[Union[ArrayLike, Callable[[float], ArrayLike]]] = None,
        external_potential: Optional[PerturbationLike] = None,
        origin: Optional[ArrayLike] = None,
        propagator: str = "pc",
        conv_tol: float = 1.0e-8,
        max_cycle: int = 30,
        verbose: Optional[int] = None,
    ):
        if not isinstance(mf, uks.UKS):
            raise TypeError("RTUKS expects a PySCF dft.UKS/uks.UKS object")
        if getattr(mf, "mo_coeff", None) is None or getattr(mf, "mo_occ", None) is None:
            raise RuntimeError("UKS object must be initialized before RT propagation")
        if np.asarray(mf.mo_coeff).shape[0] != 2:
            raise ValueError("UKS mo_coeff must contain alpha and beta orbitals")

        self.mf = mf
        self.mol = mf.mol
        self.dt = float(dt)
        self.field = field
        self.external_potential = external_potential
        self.origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
        self.propagator = propagator.lower()
        self.conv_tol = float(conv_tol)
        self.max_cycle = int(max_cycle)
        self.verbose = mf.verbose if verbose is None else verbose
        self.log = logger.new_logger(mf, self.verbose)

        self.mo_coeff = np.asarray(mf.mo_coeff)
        self.mo_occ = np.asarray(mf.mo_occ)
        self.mo_energy = np.asarray(mf.mo_energy)
        self.nao = self.mo_coeff.shape[-2]
        self.nmo = self.mo_coeff.shape[-1]
        self.nelec = tuple(mf.nelec)

        self.hcore_ao = np.asarray(mf.get_hcore())
        self.ovlp_ao = np.asarray(mf.get_ovlp())
        self.dipole_ao = self._make_dipole_integrals()
        self.angular_momentum_ao = self._make_angular_momentum_integrals()
        self.quadrupole_ao = self._make_quadrupole_integrals()
        self.hcore_mo0 = self._ao_to_mo_one_electron(self.hcore_ao)

        self.time = 0.0
        self.p_mo = self._ground_state_density_mo()
        self.history: Dict[str, List[np.ndarray]] = {}
        self.td = None
        self.reset_history()

    @classmethod
    def from_mol(
        cls,
        mol,
        xc: str = "pbe0",
        dt: float = 0.05,
        dm0: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "RTUKS":
        """Build and run a PySCF UKS reference before constructing RTUKS."""
        mf = dft.UKS(mol)
        mf.xc = xc
        for key in ("max_memory", "verbose", "conv_tol", "max_cycle"):
            if key in kwargs:
                setattr(mf, key, kwargs.pop(key))
        mf.kernel(dm0=dm0)
        return cls(mf, dt=dt, **kwargs)

    def reset_history(self) -> None:
        self.history = {
            "time": [],
            "energy": [],
            "dipole": [],
            "electronic_dipole": [],
            "nelec": [],
            "pc_error": [],
            "pc_cycle": [],
        }

    def run_tduks(self, nstates: int = 10, **kwargs):
        """Run PySCF TDUKS on the unperturbed UKS reference."""
        self.td = tduks.TDDFT(self.mf)
        for key, value in kwargs.items():
            setattr(self.td, key, value)
        self.td.kernel(nstates=nstates)
        return self.td

    def initialize_ground_state(self) -> np.ndarray:
        """Reset the propagated density to the unperturbed UKS determinant."""
        self.time = 0.0
        self.p_mo = self._ground_state_density_mo()
        self.reset_history()
        return self.p_mo

    def initialize_from_ao_density(self, dm_ao: ArrayLike) -> np.ndarray:
        """Initialize the MO-basis density from an AO spin density matrix."""
        self.time = 0.0
        self.p_mo = self.ao_density_to_mo(dm_ao)
        self.reset_history()
        return self.p_mo

    def initialize_from_static_field(
        self,
        field: ArrayLike,
        dm0: Optional[np.ndarray] = None,
        max_cycle: Optional[int] = None,
    ) -> np.ndarray:
        """Prepare the initial state with a weak static electric field.

        The field-polarized UKS density is transformed back to the unperturbed
        canonical MO basis.  Propagation then remains field-free unless
        ``self.field`` is set.
        """
        return self.initialize_from_static_potential(
            {"type": "electric_field", "field": field},
            dm0=dm0,
            max_cycle=max_cycle,
        )

    def initialize_from_static_potential(
        self,
        perturbation: PerturbationLike,
        dm0: Optional[np.ndarray] = None,
        max_cycle: Optional[int] = None,
    ) -> np.ndarray:
        """Prepare the initial state with a static one-electron perturbation.

        ``perturbation`` accepts the same specification as
        :meth:`set_external_potential`: electric-field vectors, magnetic-field
        dictionaries, AO/spin-AO matrices, or callable/dictionary forms.
        The perturbed UKS density is transformed back to the unperturbed
        canonical MO basis for subsequent propagation.
        """
        mf_field = self._perturbed_uks(perturbation)
        if max_cycle is not None:
            mf_field.max_cycle = max_cycle
        if dm0 is None:
            dm0 = self.mf.make_rdm1()
        mf_field.kernel(dm0=dm0)
        if not mf_field.converged:
            self.log.warn("field-polarized UKS calculation did not converge")
        return self.initialize_from_ao_density(mf_field.make_rdm1())

    def initialize_from_laser_field(
        self,
        laser_field: Callable[[float], ArrayLike],
        t_end: Optional[float] = None,
        t_start: float = 0.0,
        duration: Optional[float] = None,
        reset: bool = True,
        keep_history: bool = False,
        save_density: bool = False,
        field_after: Optional[Union[ArrayLike, Callable[[float], ArrayLike]]] = None,
    ) -> np.ndarray:
        """Prepare the initial density by propagating under a finite laser pulse.

        Usage::

            rt.initialize_from_laser_pulse(
                amplitude=1.0e-3,
                omega=0.35,
                duration=200.0,
                major_axis=(0.0, 0.0, 1.0),
            )
            hist = rt.kernel(4000)

        After the pulse, the propagated density is kept as ``self.p_mo`` and
        ``self.field`` is set to ``field_after``.  The default ``field_after``
        is ``None``, which leaves the state ready for field-free propagation.
        """
        t_start, t_end = _resolve_time_window(t_start, t_end, duration)

        dt0 = self.dt
        if reset:
            self.initialize_ground_state()
        else:
            self.reset_history()

        self.time = float(t_start)
        self.field = laser_field
        self._last_pc_error = 0.0
        self._last_pc_cycle = 0

        if keep_history:
            self.record(save_density=save_density)

        try:
            while self.time < t_end - 1.0e-14:
                self.dt = min(dt0, t_end - self.time)
                # The laser Hamiltonian is added inside step() through
                # fock_mo(..., time) -> _external_potential_ao(time).
                self.step()
                if keep_history:
                    self.record(save_density=save_density)
        finally:
            self.dt = dt0

        prepared_density = self.p_mo.copy()
        if keep_history:
            self.preparation_history = self.history
        self.field = field_after
        self.time = 0.0
        self._last_pc_error = 0.0
        self._last_pc_cycle = 0
        self.reset_history()
        return prepared_density

    def initialize_from_laser_pulse(
        self,
        amplitude: float,
        omega: float,
        center: Optional[float] = None,
        sigma: Optional[float] = None,
        major_axis: ArrayLike = (0.0, 0.0, 1.0),
        minor_axis: Optional[ArrayLike] = None,
        ellipticity: float = 0.0,
        phase: float = 0.0,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        duration: Optional[float] = None,
        cutoff: float = 4.0,
        normalize_intensity: bool = False,
        reset: bool = True,
        keep_history: bool = False,
        save_density: bool = False,
        field_after: Optional[Union[ArrayLike, Callable[[float], ArrayLike]]] = None,
    ) -> np.ndarray:
        """Prepare the initial state with an elliptical Gaussian laser packet.

        ``duration`` is the simplest way to define the active pulse window. If
        ``center`` and ``sigma`` are omitted, the pulse is centered in that
        window and ``sigma = duration / (2 * cutoff)``.
        """
        laser = gaussian_packet_laser_field(
            amplitude=amplitude,
            omega=omega,
            center=center,
            sigma=sigma,
            major_axis=major_axis,
            minor_axis=minor_axis,
            ellipticity=ellipticity,
            phase=phase,
            t_start=t_start,
            t_end=t_end,
            duration=duration,
            cutoff=cutoff,
            normalize_intensity=normalize_intensity,
        )
        return self.initialize_from_laser_field(
            laser,
            t_end=laser.t_end,
            t_start=laser.t_start,
            duration=laser.duration,
            reset=reset,
            keep_history=keep_history,
            save_density=save_density,
            field_after=field_after,
        )

    def apply_delta_kick(self, kick: ArrayLike) -> np.ndarray:
        """Apply an instantaneous electric dipole kick to the density.

        ``kick`` is the time-integrated electric field in atomic units.  This
        is a convenient alternative to ``initialize_from_static_field`` for
        linear-response spectra.
        """
        return self.apply_kick({"type": "electric_field", "field": kick})

    def apply_magnetic_kick(
        self,
        kick: ArrayLike,
        include_orbital: bool = True,
        include_spin: bool = True,
        g_factor: float = 2.00231930436256,
        spin_axis: ArrayLike = (0.0, 0.0, 1.0),
        include_diamagnetic: bool = False,
    ) -> np.ndarray:
        """Apply an instantaneous magnetic-field kick.

        ``kick`` is the time-integrated magnetic field in atomic units.  The
        collinear UKS representation supports the orbital Zeeman term and the
        spin-Zeeman term projected onto ``spin_axis``.  Transverse spin-flip
        magnetic fields require a noncollinear spinor treatment and are not
        represented by this class.
        """
        return self.apply_kick(
            {
                "type": "magnetic_field",
                "field": kick,
                "include_orbital": include_orbital,
                "include_spin": include_spin,
                "g_factor": g_factor,
                "spin_axis": spin_axis,
                "include_diamagnetic": include_diamagnetic,
            }
        )

    def apply_kick(self, perturbation: PerturbationLike) -> np.ndarray:
        """Apply an instantaneous kick from a one-electron Hamiltonian.

        The argument is interpreted as the time-integrated perturbation.  A
        3-vector is treated as an electric dipole kick for backward
        compatibility; pass ``{"type": "one_electron", "matrix": h1ao}``
        for an arbitrary AO one-electron kick.
        """
        v_mo = self._external_potential_mo(perturbation)
        p_new = []
        for spin in range(2):
            u = linalg.expm(-1j * v_mo[spin])
            p_new.append(u @ self.p_mo[spin] @ u.conj().T)
        self.p_mo = _hermitize_spin(np.asarray(p_new))
        return self.p_mo

    def set_external_potential(
        self,
        perturbation: Optional[PerturbationLike] = None,
        field: Optional[Union[ArrayLike, Callable[[float], ArrayLike]]] = None,
    ) -> None:
        """Set propagation-time one-electron perturbations.

        ``field`` is the legacy electric field channel.  ``perturbation`` is
        the general channel and may be a dictionary, matrix, list of terms, or
        callable returning any supported specification.
        """
        self.field = field
        self.external_potential = perturbation

    def kernel(
        self,
        nsteps: int,
        save_density: bool = False,
        callback: Optional[Callable[["RTUKS"], None]] = None,
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
        self.history["time"].append(float(self.time))
        self.history["energy"].append(float(self.energy_tot(dm=dm).real))
        self.history["dipole"].append(np.asarray(self.dipole(dm=dm, total=True)).real)
        self.history["electronic_dipole"].append(
            np.asarray(self.dipole(dm=dm, total=False)).real
        )
        self.history["nelec"].append(np.asarray(self.electron_number(dm=dm)).real)
        self.history["pc_error"].append(float(getattr(self, "_last_pc_error", 0.0)))
        self.history["pc_cycle"].append(int(getattr(self, "_last_pc_cycle", 0)))
        if save_density:
            self.history.setdefault("dm", []).append(dm.copy())
            self.history.setdefault("p_mo", []).append(self.p_mo.copy())

    def make_rdm1(self, p_mo: Optional[ArrayLike] = None) -> np.ndarray:
        """Return the current alpha/beta AO density matrix."""
        if p_mo is None:
            p_mo = self.p_mo
        return self.mo_density_to_ao(p_mo)

    def mo_density_to_ao(self, p_mo: ArrayLike) -> np.ndarray:
        p_a, p_b = _as_spin_matrices(p_mo, "p_mo")
        dm_a = self.mo_coeff[0] @ p_a @ self.mo_coeff[0].conj().T
        dm_b = self.mo_coeff[1] @ p_b @ self.mo_coeff[1].conj().T
        return _hermitize_spin(np.asarray((dm_a, dm_b)))

    def ao_density_to_mo(self, dm_ao: ArrayLike) -> np.ndarray:
        dm_a, dm_b = _as_spin_matrices(dm_ao, "dm_ao")
        s = self.ovlp_ao
        p_a = self.mo_coeff[0].conj().T @ s @ dm_a @ s @ self.mo_coeff[0]
        p_b = self.mo_coeff[1].conj().T @ s @ dm_b @ s @ self.mo_coeff[1]
        return _hermitize_spin(np.asarray((p_a, p_b)))

    def fock_mo(
        self,
        p_mo: Optional[ArrayLike] = None,
        time: Optional[float] = None,
    ) -> np.ndarray:
        dm = self.make_rdm1(p_mo)
        vhf = self.mf.get_veff(self.mol, dm, hermi=1)
        fock_ao = np.asarray(self.hcore_ao + vhf)
        if time is not None:
            # This is where the propagation-time field enters the Fock:
            # F(t) = hcore + veff[P(t)] + V_ext(t).  During laser preparation
            # self.field is the laser callable set by initialize_from_laser_field().
            fock_ao = fock_ao + self._external_potential_ao(time)
        return self._ao_to_mo_spin(fock_ao)

    def energy_tot(self, dm: Optional[ArrayLike] = None) -> float:
        if dm is None:
            dm = self.make_rdm1()
        dm = _hermitize_spin(dm)
        vhf = self.mf.get_veff(self.mol, dm, hermi=1)
        e_elec = self.mf.energy_elec(dm=dm, h1e=self.hcore_ao, vhf=vhf)[0]
        return e_elec + self.mf.energy_nuc()

    def electron_number(self, dm: Optional[ArrayLike] = None) -> np.ndarray:
        if dm is None:
            dm = self.make_rdm1()
        dm_a, dm_b = _as_spin_matrices(dm, "dm")
        s = self.ovlp_ao
        return np.asarray((np.einsum("ij,ji->", dm_a, s), np.einsum("ij,ji->", dm_b, s)))

    def dipole(self, dm: Optional[ArrayLike] = None, total: bool = True) -> np.ndarray:
        """Return dipole vector in atomic units.

        The electronic term uses the usual electron-charge sign.  If
        ``total=True`` the nuclear contribution is added.
        """
        if dm is None:
            dm = self.make_rdm1()
        dm_sum = np.asarray(dm[0] + dm[1])
        electronic = -np.einsum("xij,ji->x", self.dipole_ao, dm_sum)
        if not total:
            return electronic
        nuclear = np.zeros(3)
        for ia in range(self.mol.natm):
            nuclear += self.mol.atom_charge(ia) * self.mol.atom_coord(ia)
        return nuclear + electronic

    def _step_etrs(self) -> Tuple[np.ndarray, float, int]:
        f0 = self.fock_mo(self.p_mo, self.time)
        p_pred = self._propagate_with_fock(self.p_mo, f0, self.dt)
        f1 = self.fock_mo(p_pred, self.time + self.dt)
        f_mid = (f0 + f1) * 0.5
        p_new = self._propagate_with_fock(self.p_mo, f_mid, self.dt)
        return p_new, _density_error(p_new, p_pred), 1

    def _step_predictor_corrector(self) -> Tuple[np.ndarray, float, int]:
        f0 = self.fock_mo(self.p_mo, self.time)
        p_old = self._propagate_with_fock(self.p_mo, f0, self.dt)
        error = np.inf

        for cycle in range(1, self.max_cycle + 1):
            f1 = self.fock_mo(p_old, self.time + self.dt)
            f_mid = (f0 + f1) * 0.5
            p_new = self._propagate_with_fock(self.p_mo, f_mid, self.dt)
            error = _density_error(p_new, p_old)
            if error < self.conv_tol:
                return p_new, error, cycle
            p_old = p_new

        self.log.warn("RTUKS predictor-corrector did not converge; error=%g", error)
        return p_old, error, self.max_cycle

    @staticmethod
    def _propagate_with_fock(p_mo: ArrayLike, fock_mo: ArrayLike, dt: float) -> np.ndarray:
        p_a, p_b = _as_spin_matrices(p_mo, "p_mo")
        f_a, f_b = _as_spin_matrices(fock_mo, "fock_mo")
        out = []
        for p, f in ((p_a, f_a), (p_b, f_b)):
            u = linalg.expm(-1j * _hermitize(f) * dt)
            out.append(u @ p @ u.conj().T)
        return _hermitize_spin(np.asarray(out))

    def _ground_state_density_mo(self) -> np.ndarray:
        return np.asarray((np.diag(self.mo_occ[0]), np.diag(self.mo_occ[1])), dtype=complex)

    def _make_dipole_integrals(self) -> np.ndarray:
        with self.mol.with_common_orig(self.origin):
            return np.asarray(self.mol.intor("int1e_r", comp=3, hermi=1))

    def _make_angular_momentum_integrals(self) -> np.ndarray:
        with self.mol.with_common_orig(self.origin):
            return np.asarray(self.mol.intor("int1e_cg_irxp", comp=3, hermi=2))

    def _make_quadrupole_integrals(self) -> np.ndarray:
        with self.mol.with_common_orig(self.origin):
            rr = self.mol.intor("int1e_rr", comp=9, hermi=1)
        return np.asarray(rr).reshape(3, 3, self.nao, self.nao)

    def _ao_to_mo_one_electron(self, h_ao: np.ndarray) -> np.ndarray:
        h_a = self.mo_coeff[0].conj().T @ h_ao @ self.mo_coeff[0]
        h_b = self.mo_coeff[1].conj().T @ h_ao @ self.mo_coeff[1]
        return np.asarray((h_a, h_b))

    def _ao_to_mo_spin(self, h_ao: ArrayLike) -> np.ndarray:
        h = np.asarray(h_ao)
        if h.ndim == 2:
            return self._ao_to_mo_one_electron(h)
        h_a, h_b = _as_spin_matrices(h, "h_ao")
        return np.asarray(
            (
                self.mo_coeff[0].conj().T @ h_a @ self.mo_coeff[0],
                self.mo_coeff[1].conj().T @ h_b @ self.mo_coeff[1],
            )
        )

    def _mo_to_ao_spin(self, h_mo: ArrayLike) -> np.ndarray:
        h = np.asarray(h_mo, dtype=complex)
        if h.shape == (self.nmo, self.nmo):
            h = np.asarray((h, h))
        elif h.shape != (2, self.nmo, self.nmo):
            raise ValueError(
                "MO one-electron perturbation must have shape "
                f"({self.nmo}, {self.nmo}) or (2, {self.nmo}, {self.nmo})"
            )

        out = []
        s = self.ovlp_ao
        for spin in range(2):
            c = self.mo_coeff[spin]
            out.append(s @ c @ h[spin] @ c.conj().T @ s)
        return _hermitize_spin(np.asarray(out))

    def _as_spin_operator_ao(self, h_ao: ArrayLike, name: str = "h_ao") -> np.ndarray:
        h = np.asarray(h_ao, dtype=complex)
        if h.shape == (self.nao, self.nao):
            h = np.asarray((h, h))
        elif h.shape != (2, self.nao, self.nao):
            raise ValueError(
                f"{name} must have shape ({self.nao}, {self.nao}) "
                f"or (2, {self.nao}, {self.nao})"
            )
        return _hermitize_spin(h)

    def _collapse_spin_operator_ao(self, h_ao: ArrayLike) -> np.ndarray:
        h = self._as_spin_operator_ao(h_ao)
        if np.allclose(h[0], h[1]):
            return h[0]
        return h

    def _electric_field_potential_ao(self, field: ArrayLike) -> np.ndarray:
        efield = np.asarray(field, dtype=float)
        if efield.shape != (3,):
            raise ValueError("electric field must be a 3-vector")
        v_ao = np.einsum("x,xij->ij", efield, self.dipole_ao)
        return self._as_spin_operator_ao(v_ao, "electric perturbation")

    def _magnetic_field_potential_ao(
        self,
        field: ArrayLike,
        include_orbital: bool = True,
        include_spin: bool = True,
        g_factor: float = 2.00231930436256,
        spin_axis: ArrayLike = (0.0, 0.0, 1.0),
        include_diamagnetic: bool = False,
    ) -> np.ndarray:
        bfield = np.asarray(field, dtype=float)
        if bfield.shape != (3,):
            raise ValueError("magnetic field must be a 3-vector")

        h = np.zeros((2, self.nao, self.nao), dtype=complex)
        if include_orbital:
            orbital = -0.5j * np.einsum("x,xij->ij", bfield, self.angular_momentum_ao)
            h[0] += orbital
            h[1] += orbital

        if include_spin:
            axis = _unit_vector(spin_axis, "spin_axis")
            spin_projection = float(np.dot(bfield, axis))
            spin_shift = 0.25 * float(g_factor) * spin_projection
            h[0] += spin_shift * self.ovlp_ao
            h[1] -= spin_shift * self.ovlp_ao

        if include_diamagnetic:
            rr_trace = self.quadrupole_ao[0, 0] + self.quadrupole_ao[1, 1] + self.quadrupole_ao[2, 2]
            b_rr_b = np.einsum("x,y,xyij->ij", bfield, bfield, self.quadrupole_ao)
            diamagnetic = 0.125 * (np.dot(bfield, bfield) * rr_trace - b_rr_b)
            h[0] += diamagnetic
            h[1] += diamagnetic

        return _hermitize_spin(h)

    def _external_potential_mo(
        self,
        perturbation: PerturbationLike,
        time: Optional[float] = None,
    ) -> np.ndarray:
        return self._ao_to_mo_spin(self._external_potential_ao_from_spec(perturbation, time))

    def _external_potential_ao(self, time: float) -> np.ndarray:
        v_ao = np.zeros((2, self.nao, self.nao), dtype=complex)
        if self.field is not None:
            field = _call_if_needed(self.field, time)
            v_ao += self._electric_field_potential_ao(field)
        if self.external_potential is not None:
            v_ao += self._external_potential_ao_from_spec(self.external_potential, time)
        return _hermitize_spin(v_ao)

    def _external_potential_ao_from_spec(
        self,
        perturbation: PerturbationLike,
        time: Optional[float] = None,
    ) -> np.ndarray:
        perturbation = _call_if_needed(perturbation, time)
        if perturbation is None:
            return np.zeros((2, self.nao, self.nao), dtype=complex)

        if isinstance(perturbation, dict):
            return self._external_potential_ao_from_dict(perturbation, time)

        if isinstance(perturbation, (list, tuple)):
            arr = np.asarray(perturbation)
            if arr.dtype == object:
                total = np.zeros((2, self.nao, self.nao), dtype=complex)
                for term in perturbation:
                    total += self._external_potential_ao_from_spec(term, time)
                return _hermitize_spin(total)

        arr = np.asarray(perturbation)
        if arr.ndim == 1 and arr.size == 3:
            return self._electric_field_potential_ao(arr)
        return self._as_spin_operator_ao(arr, "one-electron perturbation")

    def _external_potential_ao_from_dict(
        self,
        spec: Dict[str, Any],
        time: Optional[float] = None,
    ) -> np.ndarray:
        if "terms" in spec and "type" not in spec and "kind" not in spec:
            return self._external_potential_ao_from_spec(spec["terms"], time)

        kind = str(spec.get("type", spec.get("kind", ""))).lower().replace("-", "_")
        if not kind:
            if any(key in spec for key in ("matrix", "h1", "hcore")):
                kind = "one_electron"
            elif any(key in spec for key in ("field", "efield", "electric_field")):
                kind = "electric_field"
            else:
                raise ValueError("perturbation dictionary requires a type/kind")

        if kind in ("sum", "list", "terms"):
            return self._external_potential_ao_from_spec(spec["terms"], time)

        scale = _call_if_needed(spec.get("scale", 1.0), time)
        scale = complex(scale)

        if kind in (
            "electric",
            "electric_field",
            "electric_kick",
            "electric_potential",
            "efield",
            "dipole",
            "electronic_field",
            "electronic_kick",
        ):
            field = spec.get(
                "field",
                spec.get("vector", spec.get("efield", spec.get("electric_field"))),
            )
            field = _call_if_needed(field, time)
            return scale * self._electric_field_potential_ao(field)

        if kind in ("magnetic", "magnetic_field", "magnetic_kick", "bfield", "zeeman"):
            field = spec.get(
                "field",
                spec.get("vector", spec.get("b", spec.get("bfield", spec.get("magnetic_field")))),
            )
            field = _call_if_needed(field, time)
            return scale * self._magnetic_field_potential_ao(
                field,
                include_orbital=bool(spec.get("include_orbital", True)),
                include_spin=bool(spec.get("include_spin", True)),
                g_factor=float(spec.get("g_factor", 2.00231930436256)),
                spin_axis=spec.get("spin_axis", (0.0, 0.0, 1.0)),
                include_diamagnetic=bool(spec.get("include_diamagnetic", False)),
            )

        if kind in (
            "one_electron",
            "one_electron_hamiltonian",
            "h1",
            "h1e",
            "hcore",
            "hamiltonian",
            "matrix",
            "ao_matrix",
            "perturbation",
        ):
            matrix = spec.get("matrix", spec.get("h1", spec.get("hcore")))
            matrix = _call_if_needed(matrix, time)
            basis = str(spec.get("basis", "ao")).lower()
            if basis == "mo":
                return scale * self._mo_to_ao_spin(matrix)
            if basis != "ao":
                raise ValueError("one-electron perturbation basis must be 'ao' or 'mo'")
            return scale * self._as_spin_operator_ao(matrix, "one-electron perturbation")

        raise ValueError(f"unknown external perturbation type {kind!r}")

    def _field_uks(self, field: ArrayLike) -> uks.UKS:
        return self._perturbed_uks({"type": "electric_field", "field": field})

    @staticmethod
    def _install_spin_hcore_energy(mf_field: uks.UKS) -> None:
        def energy_elec(dm=None, h1e=None, vhf=None):
            if dm is None:
                dm = mf_field.make_rdm1()
            if h1e is None:
                h1e = mf_field.get_hcore()
            h1e_arr = np.asarray(h1e)
            if h1e_arr.ndim != 3:
                return uks.energy_elec(mf_field, dm=dm, h1e=h1e, vhf=vhf)
            if vhf is None or getattr(vhf, "ecoul", None) is None:
                vhf = mf_field.get_veff(mf_field.mol, dm)
            dm_arr = np.asarray(dm)
            if dm_arr.ndim == 2:
                dm_arr = np.asarray((dm_arr * 0.5, dm_arr * 0.5))
            e1 = (
                np.einsum("ij,ji->", h1e_arr[0], dm_arr[0])
                + np.einsum("ij,ji->", h1e_arr[1], dm_arr[1])
            ).real
            ecoul = getattr(vhf, "ecoul", 0.0).real
            exc = getattr(vhf, "exc", 0.0).real
            e2 = ecoul + exc
            mf_field.scf_summary["e1"] = e1
            mf_field.scf_summary["coul"] = ecoul
            mf_field.scf_summary["exc"] = exc
            return e1 + e2, e2

        mf_field.energy_elec = energy_elec

    def _perturbed_uks(self, perturbation: PerturbationLike) -> uks.UKS:
        mf_field = self.mf.__class__(self.mol)
        mf_field.xc = self.mf.xc
        mf_field.max_memory = self.mf.max_memory
        mf_field.verbose = self.mf.verbose
        mf_field.conv_tol = self.mf.conv_tol
        mf_field.max_cycle = self.mf.max_cycle
        mf_field.grids.level = self.mf.grids.level
        mf_field.grids.prune = self.mf.grids.prune
        if getattr(self.mf, "nlc", ""):
            mf_field.nlc = self.mf.nlc
        v_ao = self._collapse_spin_operator_ao(
            self._external_potential_ao_from_spec(perturbation, time=0.0)
        )
        hcore = self.hcore_ao + v_ao
        mf_field.get_hcore = lambda *args: hcore
        if np.asarray(hcore).ndim == 3:
            self._install_spin_hcore_energy(mf_field)
        return mf_field


if __name__ == "__main__":
    from pyscf import gto

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        verbose=0,
    )
    mf = dft.UKS(mol)
    mf.xc = "lda,vwn"
    mf.kernel()

    rt = RTUKS(mf, dt=0.05, propagator="pc")
    rt.initialize_from_laser_pulse(
        amplitude=1.0e-4,
        omega=0.5,
        duration=0.5,
        major_axis=(0.0, 0.0, 1.0),
        keep_history=True,
    )
    hist = rt.kernel(10)
    print("laser preparation steps:", len(rt.preparation_history["time"]))
    print("final time:", hist["time"][-1])
    print("final energy:", hist["energy"][-1])
    print("final dipole:", hist["dipole"][-1])
