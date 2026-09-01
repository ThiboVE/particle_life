import timeit

import jax.numpy as jnp
import numpy as np

from particle_life import CellList, compute_forces, compute_forces_cell_list, compute_forces_jax


def test_jax() -> None:
    NUM_PARTICLES = 2000
    NUM_COLOURS = 3
    rng = np.random.default_rng(42)

    positions = rng.uniform(0, 1, size=(NUM_PARTICLES, 2))
    colours = rng.integers(0, NUM_COLOURS, size=(NUM_PARTICLES,))
    attraction_matrix = rng.uniform(-1, 1, size=(NUM_COLOURS, NUM_COLOURS))
    colour_pairs = attraction_matrix[colours[:, None], colours[None, :]]

    positions_jax = jnp.array(positions)
    colour_pairs_jax = jnp.array(colour_pairs)

    cell_list = CellList(6)
    cell_list.build(positions)

    # --- warm-up / trigger compilation (excluded from timing) ---
    compute_forces_jax(positions_jax, colour_pairs_jax, beta=0.3).block_until_ready()

    t_numpy = timeit.timeit(lambda: compute_forces(positions, colour_pairs, beta=0.3), number=20) / 20

    t_celllist = timeit.timeit(lambda: compute_forces_cell_list(positions, colour_pairs, cell_list), number=20) / 20

    t_jax_jit = (
        timeit.timeit(
            lambda: compute_forces_jax(positions_jax, colour_pairs_jax, beta=0.3).block_until_ready(), number=20
        )
        / 20
    )

    print(f"numpy: {t_numpy * 1000:.3f} ms")
    print(f"celllist: {t_celllist * 1000:.3f} ms")
    print(f"jax jit (CPU): {t_jax_jit * 1000:.3f} ms")
    print(f"speedup: {t_numpy / t_jax_jit:.2f}")


if __name__ == "__main__":
    test_jax()
