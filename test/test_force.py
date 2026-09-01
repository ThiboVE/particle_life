import jax.numpy as jnp
import numpy as np

from particle_life import compute_forces, compute_forces_jax

NUM_PARTICLES = 10
NUM_COLOURS = 3


def test_jax_force() -> None:
    rng = np.random.default_rng(42)

    positions = rng.uniform(0, 1, size=(NUM_PARTICLES, 2))
    colours = rng.integers(0, NUM_COLOURS, size=(NUM_PARTICLES,))
    attraction_matrix = rng.uniform(-1, 1, size=(NUM_COLOURS, NUM_COLOURS))
    colour_pairs = attraction_matrix[colours[:, None], colours[None, :]]

    ref_forces = compute_forces(positions, colour_pairs)
    jax_forces = compute_forces_jax(jnp.array(positions), jnp.array(colour_pairs))

    assert np.allclose(ref_forces, jax_forces)
