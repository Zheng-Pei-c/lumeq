"""Thermostats for molecular dynamics with DFT electronic states.

The classes in this module act on nuclear velocities and masses.  They are
agnostic to how the conservative forces are produced, so the same thermostat
can be used with Born-Oppenheimer DFT, Ehrenfest/RT-TDDFT, or extended
Lagrangian DFT force callbacks.  The helper ``thermostatted_velocity_verlet_step``
expects a force callback that updates the electronic state at the new nuclear
geometry and returns ``(energy, force)``.

All quantities are in atomic units unless noted otherwise.  Nuclear masses are
electron-mass units, coordinates are Bohr, velocities are Bohr / atomic time,
forces are Hartree / Bohr, and temperatures are Kelvin by default.

References:
    * H. C. Andersen, "Molecular dynamics simulations at constant pressure
      and/or temperature", J. Chem. Phys. 72, 2384-2393 (1980),
      https://doi.org/10.1063/1.439486.
    * H. J. C. Berendsen, J. P. M. Postma, W. F. van Gunsteren, A. DiNola,
      and J. R. Haak, "Molecular dynamics with coupling to an external bath",
      J. Chem. Phys. 81, 3684-3690 (1984), https://doi.org/10.1063/1.448118.
    * T. Schneider and E. Stoll, "Molecular-dynamics study of a three-
      dimensional one-component model for distortive phase transitions",
      Phys. Rev. B 17, 1302-1322 (1978),
      https://doi.org/10.1103/PhysRevB.17.1302.
    * B. Leimkuhler and C. Matthews, "Rational construction of stochastic
      numerical methods for molecular sampling", Appl. Math. Res. Express
      2013, 34-56 (2013), https://doi.org/10.1093/amrx/abs010.
    * D. Marx and J. Hutter, Ab Initio Molecular Dynamics: Basic Theory and
      Advanced Methods, Cambridge University Press (2009).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from lumeq import np


BOLTZMANN_HARTREE_PER_KELVIN = 3.166811563e-6


@dataclass(frozen=True)
class ThermostatResult:
    """Result returned by a thermostat velocity update.

    Attributes:
        velocity: Updated velocity array with the same shape as the input.
        kinetic_energy: Nuclear kinetic energy after thermostat application.
        temperature: Instantaneous kinetic temperature in Kelvin.
        scale: Deterministic velocity scale factor when one is defined.
        metadata: Thermostat-specific diagnostic values.
    """

    velocity: np.ndarray
    kinetic_energy: float
    temperature: float
    scale: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThermostattedMDStep:
    """One velocity-Verlet molecular-dynamics step with optional thermostat.

    Attributes:
        coordinate: Updated nuclear coordinates.
        velocity: Updated nuclear velocities.
        energy: Potential energy returned by the DFT/electronic force callback.
        force: Nuclear force at ``coordinate``.
        thermostat: Thermostat diagnostics, or ``None`` for NVE propagation.
    """

    coordinate: np.ndarray
    velocity: np.ndarray
    energy: float
    force: np.ndarray
    thermostat: Optional[ThermostatResult] = None


def _validate_velocity_mass(
    velocity: np.ndarray,
    mass: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return validated velocity and broadcastable mass-column arrays."""

    velocity = np.asarray(velocity, dtype=float)
    mass = np.asarray(mass, dtype=float)
    if velocity.ndim < 1:
        raise ValueError("velocity must have at least one dimension.")
    if mass.ndim == 0:
        mass = np.full(velocity.shape[0], float(mass))
    if mass.ndim != 1 or mass.shape[0] != velocity.shape[0]:
        raise ValueError("mass must be scalar or have length velocity.shape[0].")
    if np.any(mass <= 0.0):
        raise ValueError("all masses must be positive.")
    mass_shape = (mass.size,) + (1,) * (velocity.ndim - 1)
    return velocity, mass.reshape(mass_shape)


def kinetic_energy(velocity: np.ndarray, mass: np.ndarray) -> float:
    """Return classical nuclear kinetic energy.

    Args:
        velocity: Velocity array, usually ``(natoms, 3)``.
        mass: Scalar mass or per-particle masses matching ``velocity.shape[0]``.

    Returns:
        Kinetic energy in Hartree.
    """

    velocity, mass_col = _validate_velocity_mass(velocity, mass)
    return float(0.5 * np.sum(mass_col * velocity * velocity))


def kinetic_temperature(
    velocity: np.ndarray,
    mass: np.ndarray,
    dof: Optional[int] = None,
    k_b: float = BOLTZMANN_HARTREE_PER_KELVIN,
) -> float:
    """Return the instantaneous kinetic temperature.

    Args:
        velocity: Velocity array.
        mass: Scalar or per-particle masses.
        dof: Number of active degrees of freedom.  If omitted, all velocity
            components are counted.  Use ``3 * natoms - 3`` or
            ``3 * natoms - 6`` after removing translation or translation plus
            rotation.
        k_b: Boltzmann constant in energy units per Kelvin.

    Returns:
        Temperature in Kelvin when ``k_b`` is the default Hartree/Kelvin value.
    """

    velocity = np.asarray(velocity, dtype=float)
    if dof is None:
        dof = velocity.size
    if dof <= 0:
        raise ValueError("dof must be positive.")
    return 2.0 * kinetic_energy(velocity, mass) / (float(dof) * float(k_b))


def target_kinetic_energy(
    temperature: float,
    dof: int,
    k_b: float = BOLTZMANN_HARTREE_PER_KELVIN,
) -> float:
    """Return the canonical target kinetic energy ``0.5 * dof * k_B * T``."""

    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")
    if dof <= 0:
        raise ValueError("dof must be positive.")
    return 0.5 * float(dof) * float(k_b) * float(temperature)


def maxwell_boltzmann_velocities(
    mass: np.ndarray,
    temperature: float,
    shape: Tuple[int, ...],
    rng: np.random.Generator,
    k_b: float = BOLTZMANN_HARTREE_PER_KELVIN,
) -> np.ndarray:
    """Sample velocities from the Maxwell-Boltzmann distribution.

    Args:
        mass: Scalar or per-particle masses.
        temperature: Bath temperature in Kelvin.
        shape: Desired velocity shape, usually ``(natoms, 3)``.
        rng: NumPy random generator.
        k_b: Boltzmann constant in energy units per Kelvin.

    Returns:
        Random velocity array with the requested shape.
    """

    dummy_velocity = np.zeros(shape)
    _, mass_col = _validate_velocity_mass(dummy_velocity, mass)
    sigma = np.sqrt(float(k_b) * float(temperature) / mass_col)
    return rng.normal(loc=0.0, scale=sigma, size=shape)


def remove_center_of_mass_velocity(
    velocity: np.ndarray,
    mass: np.ndarray,
) -> np.ndarray:
    """Remove center-of-mass translational velocity.

    This is useful after stochastic thermostats, which can inject a small net
    momentum unless constrained explicitly.
    """

    velocity, mass_col = _validate_velocity_mass(velocity, mass)
    total_mass = float(np.sum(mass_col.reshape(-1)))
    momentum = np.sum(mass_col * velocity, axis=0)
    return velocity - momentum / total_mass


class NullThermostat:
    """No-op thermostat for NVE molecular dynamics."""

    def apply(
        self,
        velocity: np.ndarray,
        mass: np.ndarray,
        dt: float,
        dof: Optional[int] = None,
    ) -> ThermostatResult:
        """Return velocities unchanged."""

        velocity = np.asarray(velocity, dtype=float)
        return ThermostatResult(
            velocity=velocity,
            kinetic_energy=kinetic_energy(velocity, mass),
            temperature=kinetic_temperature(velocity, mass, dof=dof),
            scale=1.0,
            metadata={"dt": float(dt)},
        )


@dataclass
class VelocityRescaleThermostat:
    """Instantaneously rescale velocities to the target temperature.

    This deterministic rescaling is useful for initialization and debugging,
    but it is not a canonical thermostat for production sampling.
    """

    temperature: float
    k_b: float = BOLTZMANN_HARTREE_PER_KELVIN

    def apply(
        self,
        velocity: np.ndarray,
        mass: np.ndarray,
        dt: float,
        dof: Optional[int] = None,
    ) -> ThermostatResult:
        """Return velocities scaled exactly to ``self.temperature``."""

        velocity = np.asarray(velocity, dtype=float)
        if dof is None:
            dof = velocity.size
        current_k = kinetic_energy(velocity, mass)
        target_k = target_kinetic_energy(self.temperature, dof, self.k_b)
        if current_k <= 0.0:
            raise ValueError("cannot rescale zero-temperature velocities.")
        scale = np.sqrt(target_k / current_k)
        velocity_new = scale * velocity
        return ThermostatResult(
            velocity=velocity_new,
            kinetic_energy=kinetic_energy(velocity_new, mass),
            temperature=kinetic_temperature(
                velocity_new, mass, dof=dof, k_b=self.k_b
            ),
            scale=float(scale),
            metadata={"dt": float(dt), "target_kinetic_energy": target_k},
        )


@dataclass
class BerendsenThermostat:
    """Weak-coupling velocity-rescaling thermostat.

    The Berendsen thermostat relaxes the instantaneous kinetic temperature
    toward the target temperature over a coupling time ``tau``.  It is robust
    for equilibration, but it does not generate the exact canonical kinetic
    energy distribution.
    """

    temperature: float
    tau: float
    k_b: float = BOLTZMANN_HARTREE_PER_KELVIN

    def apply(
        self,
        velocity: np.ndarray,
        mass: np.ndarray,
        dt: float,
        dof: Optional[int] = None,
    ) -> ThermostatResult:
        """Apply one Berendsen weak-coupling velocity update."""

        if self.tau <= 0.0:
            raise ValueError("tau must be positive.")
        velocity = np.asarray(velocity, dtype=float)
        current_t = kinetic_temperature(velocity, mass, dof=dof, k_b=self.k_b)
        if current_t <= 0.0:
            raise ValueError("cannot thermostat zero-temperature velocities.")
        factor = 1.0 + float(dt) / float(self.tau) * (
            float(self.temperature) / current_t - 1.0
        )
        if factor < 0.0:
            raise ValueError("Berendsen scale factor is negative; reduce dt/tau.")
        scale = np.sqrt(factor)
        velocity_new = scale * velocity
        return ThermostatResult(
            velocity=velocity_new,
            kinetic_energy=kinetic_energy(velocity_new, mass),
            temperature=kinetic_temperature(
                velocity_new, mass, dof=dof, k_b=self.k_b
            ),
            scale=float(scale),
            metadata={"dt": float(dt), "tau": float(self.tau)},
        )


@dataclass
class AndersenThermostat:
    """Andersen stochastic-collision thermostat.

    During each step, atoms collide with a heat bath with probability
    ``1 - exp(-collision_frequency * dt)``.  Colliding atoms receive fresh
    Maxwell-Boltzmann velocities at the target temperature.
    """

    temperature: float
    collision_frequency: float
    seed: Optional[int] = None
    remove_com: bool = False
    k_b: float = BOLTZMANN_HARTREE_PER_KELVIN
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        if self.collision_frequency < 0.0:
            raise ValueError("collision_frequency must be non-negative.")
        self._rng = np.random.default_rng(self.seed)

    def apply(
        self,
        velocity: np.ndarray,
        mass: np.ndarray,
        dt: float,
        dof: Optional[int] = None,
    ) -> ThermostatResult:
        """Apply one Andersen stochastic-collision update."""

        velocity = np.asarray(velocity, dtype=float)
        probability = 1.0 - np.exp(-float(self.collision_frequency) * float(dt))
        random_velocity = maxwell_boltzmann_velocities(
            mass=mass,
            temperature=self.temperature,
            shape=velocity.shape,
            rng=self._rng,
            k_b=self.k_b,
        )
        collision_mask = self._rng.random(velocity.shape[0]) < probability
        velocity_new = np.array(velocity, copy=True)
        velocity_new[collision_mask] = random_velocity[collision_mask]
        if self.remove_com:
            velocity_new = remove_center_of_mass_velocity(velocity_new, mass)
        return ThermostatResult(
            velocity=velocity_new,
            kinetic_energy=kinetic_energy(velocity_new, mass),
            temperature=kinetic_temperature(
                velocity_new, mass, dof=dof, k_b=self.k_b
            ),
            metadata={
                "dt": float(dt),
                "collision_probability": float(probability),
                "n_collisions": int(np.count_nonzero(collision_mask)),
            },
        )


@dataclass
class LangevinThermostat:
    """Langevin Ornstein-Uhlenbeck velocity thermostat.

    This implements the stochastic velocity update used as the ``O`` step in
    splitting schemes such as BAOAB.  With a conservative DFT force callback,
    apply it after a full velocity-Verlet step or use
    ``thermostatted_velocity_verlet_step`` below.
    """

    temperature: float
    friction: float
    seed: Optional[int] = None
    remove_com: bool = False
    k_b: float = BOLTZMANN_HARTREE_PER_KELVIN
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        if self.friction < 0.0:
            raise ValueError("friction must be non-negative.")
        self._rng = np.random.default_rng(self.seed)

    def apply(
        self,
        velocity: np.ndarray,
        mass: np.ndarray,
        dt: float,
        dof: Optional[int] = None,
    ) -> ThermostatResult:
        """Apply one Langevin heat-bath velocity update."""

        velocity, mass_col = _validate_velocity_mass(velocity, mass)
        decay = np.exp(-float(self.friction) * float(dt))
        noise_scale = np.sqrt(
            (1.0 - decay * decay) * self.k_b * self.temperature / mass_col
        )
        velocity_new = decay * velocity + noise_scale * self._rng.normal(
            size=velocity.shape
        )
        if self.remove_com:
            velocity_new = remove_center_of_mass_velocity(velocity_new, mass)
        return ThermostatResult(
            velocity=velocity_new,
            kinetic_energy=kinetic_energy(velocity_new, mass),
            temperature=kinetic_temperature(
                velocity_new, mass, dof=dof, k_b=self.k_b
            ),
            scale=float(decay),
            metadata={"dt": float(dt), "friction": float(self.friction)},
        )


def make_thermostat(name: Optional[str], **kwargs) -> Any:
    """Create a thermostat by name.

    Args:
        name: One of ``None``, ``"none"``, ``"rescale"``, ``"berendsen"``,
            ``"andersen"``, or ``"langevin"``.
        **kwargs: Constructor arguments for the selected thermostat.

    Returns:
        Thermostat object with an ``apply(velocity, mass, dt, dof=None)``
        method.
    """

    if name is None:
        return NullThermostat()
    key = str(name).strip().lower().replace("-", "_")
    if key in {"none", "null", "nve"}:
        return NullThermostat()
    if key in {"rescale", "velocity_rescale"}:
        return VelocityRescaleThermostat(**kwargs)
    if key == "berendsen":
        return BerendsenThermostat(**kwargs)
    if key == "andersen":
        return AndersenThermostat(**kwargs)
    if key == "langevin":
        return LangevinThermostat(**kwargs)
    raise ValueError(f"Unknown thermostat: {name}")


def thermostatted_velocity_verlet_step(
    coordinate: np.ndarray,
    velocity: np.ndarray,
    mass: np.ndarray,
    force: np.ndarray,
    dt: float,
    force_callback: Callable[..., Tuple[float, np.ndarray]],
    thermostat: Optional[Any] = None,
    dof: Optional[int] = None,
    force_kwargs: Optional[Dict[str, Any]] = None,
) -> ThermostattedMDStep:
    """Advance one DFT-driven molecular-dynamics step.

    The force callback is called at the new nuclear geometry and should update
    the electronic state consistently with the chosen DFT dynamics method.  A
    Born-Oppenheimer DFT callback typically solves the SCF problem at the new
    geometry; an extended-Lagrangian or RT-TDDFT callback may instead propagate
    the electronic variables before returning the nuclear force.

    Args:
        coordinate: Current nuclear coordinates with shape ``(natoms, 3)``.
        velocity: Current nuclear velocities with shape ``(natoms, 3)``.
        mass: Per-nucleus masses in electron-mass units.
        force: Current nuclear force.
        dt: Time step in atomic units.
        force_callback: Callable ``energy, force = f(coordinate, **kwargs)``.
        thermostat: Optional object with an ``apply`` method.  If omitted,
            the step is standard NVE velocity Verlet.
        dof: Active degrees of freedom for thermostat temperature diagnostics.
        force_kwargs: Optional keyword arguments passed to ``force_callback``.

    Returns:
        ``ThermostattedMDStep`` containing updated coordinates, velocities,
        potential energy, force, and thermostat diagnostics.
    """

    coordinate = np.asarray(coordinate, dtype=float)
    velocity, mass_col = _validate_velocity_mass(velocity, mass)
    force = np.asarray(force, dtype=float)
    if force.shape != velocity.shape or coordinate.shape != velocity.shape:
        raise ValueError("coordinate, velocity, and force must have the same shape.")

    half_velocity = velocity + 0.5 * float(dt) * force / mass_col
    coordinate_new = coordinate + float(dt) * half_velocity
    force_kwargs = {} if force_kwargs is None else dict(force_kwargs)
    energy_new, force_new = force_callback(coordinate_new, **force_kwargs)
    force_new = np.asarray(force_new, dtype=float)
    if force_new.shape != velocity.shape:
        raise ValueError("force_callback returned a force with the wrong shape.")

    velocity_new = half_velocity + 0.5 * float(dt) * force_new / mass_col
    thermostat_result = None
    if thermostat is not None:
        thermostat_result = thermostat.apply(
            velocity_new, mass, dt=float(dt), dof=dof
        )
        velocity_new = thermostat_result.velocity

    return ThermostattedMDStep(
        coordinate=coordinate_new,
        velocity=velocity_new,
        energy=float(energy_new),
        force=force_new,
        thermostat=thermostat_result,
    )
