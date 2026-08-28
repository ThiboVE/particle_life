import timeit
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from particle_life.CellList import CellList


@dataclass
class ForceSettings:
    dt: float = 0.02
    r_max: float = 0.15
    frictionHalfLife: float = 0.040
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


def test_cell_grid() -> None:
    num_cells = int(1 // ForceSettings.r_max)
    num_colours = 5
    # num_particles = 800
    beta = 0.3

    particle_nums = np.arange(1500, step=100)
    n2_times = []
    cell_times = []

    for num_particles in particle_nums:
        rng = np.random.default_rng(seed=100)
        positions = rng.random(size=(num_particles, 2))
        # positions = np.array([[0.45, 0.36], [0.75, 0.12], [0.46, 0.34], [0.96, 0.44], [0.99999, 0.99999]])

        colours = np.random.randint(0, num_colours, size=(num_particles,))

        attraction_matrix = np.random.uniform(-1, 1, size=(num_colours, num_colours))

        colour_pairs = attraction_matrix[colours[:, None], colours[None, :]]

        cell_list = CellList(num_cells=num_cells)
        cell_list.build(positions)

        n2_time = timeit.timeit(lambda: compute_forces(positions, colour_pairs, beta=beta), number=10)
        cell_time = timeit.timeit(
            lambda: compute_forces_cell_list(positions, colour_pairs, cell_list, beta=beta), number=10
        )

        n2_times.append(n2_time / 10 * 1000)
        cell_times.append(cell_time / 10 * 1000)

        # print(f"N² version:    {n2_time / 10 * 1000:.3f} ms/call")
        # print(f"Grid version:   {cell_time / 10 * 1000:.3f} ms/call")

    plt.plot(particle_nums, n2_times, label="O($N^2$)", color="gray")
    plt.plot(particle_nums, cell_times, label="Cell-List Algorithm", color="red")

    plt.xlabel("Number of particles", fontsize=14)
    plt.ylabel("Time (ms/call)", fontsize=14)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_cell_grid()
