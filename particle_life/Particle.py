from dataclasses import dataclass

import numpy as np

from particle_life.force import compute_forces


@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    radius: int
    colour: int


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

    def update(self, dt: float = 0.02, beta: float = 0.3) -> None:
        force = compute_forces(self.positions, self.colour_pairs, beta=beta)

        frictionHalfLife: float = 0.040
        frictionFactor: float = pow(0.5, dt / frictionHalfLife)

        positions = self.positions
        velocities = self.velocities

        velocities *= frictionFactor
        velocities += force * dt

        positions += velocities * dt
        positions %= 1.0

        self.positions = positions
        self.velocities = velocities
