import numpy as np
import pytest

from particle_life import CellList


@pytest.fixture
def cell_list() -> CellList:
    return CellList(num_cells=6)


def test_cell_ids(cell_list: CellList) -> None:
    positions = np.array([[0.45, 0.36], [0.75, 0.12], [0.96, 0.44], [0.99999, 0.99999]])

    ref_cell_ids = np.array([14, 24, 32, 35])

    cell_ids = cell_list.get_cell_ids(positions)

    assert np.allclose(ref_cell_ids, cell_ids)


def test_get_cell_particle_indices(cell_list: CellList) -> None:
    cell_ids = np.array([18, 1, 19, 25, 19, 2, 9, 16])
    cell_list.build_from_ids(cell_ids)

    assert np.allclose(cell_list.get_particles_in_cell(19), np.array([2, 4]))
    assert np.allclose(cell_list.get_particles_in_cell(18), np.array([0]))
    assert np.allclose(cell_list.get_particles_in_cell(9), np.array([6]))
