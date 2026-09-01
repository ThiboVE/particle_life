from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from particle_life.CellList import CellList


@dataclass
class ForceSettings:
    dt: float = 0.02
    r_max: float = 0.15
    # frictionHalfLife: float = 0.040
    frictionHalfLife: float = 0.030
    frictionFactor: float = pow(0.5, dt / frictionHalfLife)


def force_func(dist: np.ndarray, interaction: np.ndarray, beta: float = 0.3) -> np.ndarray:
    """This function implements the following force function.

    If dist < beta:
        return (dist / beta) - 1
    elif beta < dist < 1:
        return interaction * (1 - abs(2 * dist - 1 - beta) / (1 - beta))
    return 0
    """
    close_range = dist < beta
    mid_range = (dist >= beta) & (dist < 1)

    close_val = (dist / beta) - 1
    mid_val = interaction * (1 - np.abs(2 * dist - 1 - beta) / (1 - beta))

    return np.select([close_range, mid_range], [close_val, mid_val], default=0.0)


def force_func_jax(dist: jnp.ndarray, interaction: jnp.ndarray, beta: float = 0.3) -> jnp.ndarray:
    """This function implements the following force function.

    If dist < beta:
        return (dist / beta) - 1
    elif beta < dist < 1:
        return interaction * (1 - abs(2 * dist - 1 - beta) / (1 - beta))
    return 0
    """
    close_range = dist < beta
    mid_range = (dist >= beta) & (dist < 1)

    close_val = (dist / beta) - 1
    mid_val = interaction * (1 - jnp.abs(2 * dist - 1 - beta) / (1 - beta))

    return jnp.select([close_range, mid_range], [close_val, mid_val], default=0.0)


def compute_forces(
    positions: np.ndarray,
    colour_pairs: np.ndarray,
    beta: float = 0.3,
) -> np.ndarray:
    diff = positions[None, :, :] - positions[:, None, :]  # (N, N, 2)
    diff -= np.round(diff)
    dist = np.linalg.norm(diff, axis=-1)  # (N, N)

    min_dist = 1e-3
    dist_safe = np.where(dist < min_dist, min_dist, dist)

    r_scaled = dist_safe / ForceSettings.r_max

    force_mag = force_func(r_scaled, colour_pairs, beta)
    np.fill_diagonal(force_mag, 0.0)

    direction = diff / dist_safe[..., None]  # (N, N, 2)
    return (direction * force_mag[..., None]).sum(axis=1)  # (N, 2)


@jax.jit(static_argnames=["beta"])
def compute_forces_jax(
    positions: jnp.ndarray,
    colour_pairs: jnp.ndarray,
    beta: float = 0.3,
) -> jnp.ndarray:
    diff = positions[None, :, :] - positions[:, None, :]  # (N, N, 2)
    diff -= jnp.round(diff)
    dist = jnp.linalg.norm(diff, axis=-1)  # (N, N)

    min_dist = 1e-3
    dist_safe = jnp.where(dist < min_dist, min_dist, dist)  # (N, N)

    r_scaled = dist_safe / ForceSettings.r_max  # (N, N)

    force_mag = force_func_jax(r_scaled, colour_pairs, beta)  # (N, N)
    mask = jnp.eye(force_mag.shape[0], dtype=bool)
    force_mag = jnp.where(mask, 0.0, force_mag)

    direction = diff / dist_safe[..., None]  # (N, N, 2)
    return (direction * force_mag[..., None]).sum(axis=1)  # (N, 2)


def compute_forces_cell_list(
    positions: np.ndarray, colour_pairs: np.ndarray, cell_list: CellList, beta: float = 0.3
) -> np.ndarray:
    force = np.zeros_like(positions)
    for cell_id in cell_list.all_cells:
        cell_particles = cell_list.get_particles_in_cell(cell_id)

        if cell_particles.size == 0:
            continue

        neighbour_cell_ids = cell_list.get_cell_neighbours(cell_id=cell_id)

        candidates = np.concatenate([cell_list.get_particles_in_cell(c) for c in neighbour_cell_ids])

        cell_positions = positions[cell_particles]
        neighbour_positions = positions[candidates]

        diff = neighbour_positions[None, :, :] - cell_positions[:, None, :]  # (N_i, N_j, 2)

        diff -= np.round(diff)
        dist = np.linalg.norm(diff, axis=-1)  # (N_i, N_j)

        min_dist = 1e-3
        dist_safe = np.where(dist < min_dist, min_dist, dist)

        r_scaled = dist_safe / ForceSettings.r_max

        relevant_colour_pairs = colour_pairs[np.ix_(cell_particles, candidates)]  # (N_i, N_j)

        force_mag = force_func(r_scaled, relevant_colour_pairs, beta)
        self_mask = cell_particles[:, None] == candidates[None, :]
        force_mag = np.where(self_mask, 0.0, force_mag)

        direction = diff / dist_safe[..., None]  # (N_i, N_j, 2)
        force[cell_particles] += (direction * force_mag[..., None]).sum(axis=1)  # (N_i, 2)

    return force
