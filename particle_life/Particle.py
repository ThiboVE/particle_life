from collections.abc import Generator
from dataclasses import dataclass

import numpy as np

from particle_life.force import ForceSettings, compute_forces


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
        force = compute_forces(self.positions, self.colour_pairs, beta=beta)

        positions = self.positions
        velocities = self.velocities

        velocities *= ForceSettings.frictionFactor
        velocities += force * dt

        positions += velocities * dt
        positions %= 1.0

        self.positions = positions
        self.velocities = velocities
