import timeit

import matplotlib.pyplot as plt
import numpy as np

from particle_life.CellList import CellList
from particle_life.force import ForceSettings, compute_forces, compute_forces_cell_list


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
