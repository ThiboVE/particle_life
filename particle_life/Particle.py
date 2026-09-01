from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Self

import jax.numpy as jnp
import numpy as np

from particle_life.CellList import CellList
from particle_life.force import ForceSettings, compute_forces_jax


@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    radius: int
    colour: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Particle):
            return NotImplemented

        return (
            self.position[0] == other.position[0]
            and self.position[1] == other.position[1]
            and self.velocity[0] == other.velocity[0]
            and self.velocity[1] == other.velocity[1]
            and self.radius == other.radius
            and self.colour == other.colour
        )


@dataclass
class ParticleSystem:
    positions: np.ndarray  # (N, 2)
    velocities: np.ndarray  # (N, 2)
    colours: np.ndarray  # (N,)
    colour_pairs: np.ndarray  # (N, N)

    cell_list: CellList = field(init=False)

    def __post_init__(self) -> None:
        num_cells = int(1 // ForceSettings.r_max)
        self.cell_list = CellList(num_cells)

    @classmethod
    def from_setup(
        cls,
        n_particles: int,
        n_colours: int,
        seed: int | None = None,
    ) -> Self:
        rng = np.random.default_rng(seed or 42)

        positions = rng.uniform(0, 1, size=(n_particles, 2))
        velocities = rng.uniform(0, 1, size=(n_particles, 2))
        colours = rng.integers(0, n_colours, size=n_particles)

        attraction_matrix = rng.uniform(-1, 1, size=(n_colours, n_colours))
        colour_pairs = attraction_matrix[colours[:, None], colours[None, :]]

        return cls(positions=positions, velocities=velocities, colours=colours, colour_pairs=colour_pairs)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int) -> Particle:
        return Particle(
            position=self.positions[i],
            velocity=self.velocities[i],
            radius=3,
            colour=self.colours[i],
        )

    def __iter__(self) -> Generator[Particle, None, None]:
        for i in range(len(self.positions)):
            yield self.__getitem__(i)

    def update(self, dt: float = 0.02, beta: float = 0.3) -> None:
        self.cell_list.build(self.positions)

        # force = compute_forces_cell_list(
        #     positions=self.positions, colour_pairs=self.colour_pairs, cell_list=self.cell_list, beta=beta
        # )
        # force = compute_forces(self.positions, self.colour_pairs, beta=beta)

        jnp_positions = jnp.array(self.positions)
        jnp_colour_pairs = jnp.array(self.colour_pairs)

        force = np.array(compute_forces_jax(jnp_positions, jnp_colour_pairs, beta=beta))

        self.velocities *= ForceSettings.frictionFactor
        self.velocities += force * dt

        self.positions += self.velocities * dt
        self.positions %= 1.0
