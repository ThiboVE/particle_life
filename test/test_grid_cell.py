import numpy as np

NUM_CELLS = 6
CELL_SIZE = 1 / NUM_CELLS


def get_cell_ids(positions: np.ndarray) -> np.ndarray:
    """Take an array of particle positions and determine in which grid cell they are."""
    cell_coords = (positions // CELL_SIZE).astype(int)
    return cell_coords[:, 0] * NUM_CELLS + cell_coords[:, 1]


def test_cell_ids() -> None:
    positions = np.array([[0.45, 0.36], [0.75, 0.12], [0.96, 0.44], [0.99999, 0.99999]])

    ref_cell_ids = np.array([14, 24, 32, 35])

    cell_ids = get_cell_ids(positions)

    assert np.allclose(ref_cell_ids, cell_ids)
