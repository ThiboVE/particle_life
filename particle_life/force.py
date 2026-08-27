from dataclasses import dataclass

import numpy as np


@dataclass
class ForceSettings:
    dt: float = 0.02
    r_max: float = 0.15
    frictionHalfLife: float = 0.040
    frictionFactor: float = pow(0.5, dt / frictionHalfLife)
    forceFactor: int = 40


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


def compute_forces(
    positions: np.ndarray,
    type_pairs: np.ndarray,
    beta: float = 0.3,
) -> np.ndarray:
    diff = positions[None, :, :] - positions[:, None, :]  # (N, N, 2)
    diff -= np.round(diff)
    dist = np.linalg.norm(diff, axis=-1)  # (N, N)

    min_dist = 1e-3
    dist_safe = np.where(dist < min_dist, min_dist, dist)

    r_scaled = dist_safe / ForceSettings.r_max

    force_mag = force_func(r_scaled, type_pairs, beta)
    np.fill_diagonal(force_mag, 0.0)

    direction = diff / dist_safe[..., None]  # (N, N, 2)
    force = (direction * force_mag[..., None]).sum(axis=1)  # (N, 2)
    return force


def test_cell_grid() -> None:
    num_cells = 1 // ForceSettings.r_max
    cell_size = 1 / num_cells

    # rng = np.random.default_rng(seed=100)

    # positions = rng.random(size=(10, 2))

    positions = np.array([[0.45, 0.36], [0.75, 0.12], [0.46, 0.34], [0.96, 0.44], [0.99999, 0.99999]])

    cell_coords = (positions // cell_size).astype(int)
    cell_ids = cell_coords[:, 0] * num_cells + cell_coords[:, 1]

    order = np.argsort(cell_ids)
    sorted_cell_ids = cell_ids[order]

    cell_starts = np.searchsorted(sorted_cell_ids, np.arange(num_cells**2), side="left")
    cell_ends = np.searchsorted(sorted_cell_ids, np.arange(num_cells**2), side="right")

    # def get_neighbours(cell_coords: np.ndarray):
    dirs = [-1, 0, 1]

    offsets = np.array([[dx, dy] for dx in dirs for dy in dirs])
    neighbours = (cell_coords[:, None, :] + offsets[None, :, :]) % num_cells

    neighbour_cell_ids = (neighbours[..., 0] * num_cells + neighbours[..., 1]).astype(int)

    cell_ids_for_i = neighbour_cell_ids[-1].astype(int)
    candidates = np.concatenate([order[cell_starts[c] : cell_ends[c]] for c in cell_ids_for_i])

    print(candidates)

    print(neighbour_cell_ids)


if __name__ == "__main__":
    test_cell_grid()
