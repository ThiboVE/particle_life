import numpy as np
import pytest

from particle_life import Particle, ParticleSystem

NUM_PARTICLES = 20
NUM_COLOURS = 3


@pytest.fixture
def system() -> ParticleSystem:
    rng = np.random.default_rng(seed=100)

    positions = rng.random(size=(NUM_PARTICLES, 2))
    velocities = rng.random(size=(NUM_PARTICLES, 2))
    colours = rng.integers(0, NUM_COLOURS, size=(NUM_PARTICLES,))

    attraction_matrix = rng.uniform(-1, 1, size=(NUM_COLOURS, NUM_COLOURS))

    colour_pairs = attraction_matrix[colours[:, None], colours[None, :]]

    return ParticleSystem(
        positions=positions,
        velocities=velocities,
        colours=colours,
        colour_pairs=colour_pairs,
    )


def test_system_size(system: ParticleSystem) -> None:
    assert len(system) == NUM_PARTICLES


def test_get_particle(system: ParticleSystem) -> None:
    particle_id = 5

    test_particle = Particle(
        position=system.positions[particle_id],
        velocity=system.velocities[particle_id],
        radius=3,
        colour=system.colours[particle_id],
    )

    assert test_particle == system[particle_id]


def test_iter_particle(system: ParticleSystem) -> None:
    for i in range(4):
        test_particle = Particle(
            position=system.positions[i],
            velocity=system.velocities[i],
            radius=3,
            colour=system.colours[i],
        )

        assert test_particle == system[i]
