import numpy as np


class CellList:
    """Class that contains the logic for the Cell-List algorithm.

    Vars:
        order: (N,) Particle indices sorted by cell id, i.e. particles in the same cell are contiguous. Result of argsort(cell_ids).
        cell_starts: (num_cells^2,) For each cell id, the index in `order` where that cell's particles begin.
        cell_ends: (num_cells^2,) For each cell id, the index in `order` where that cell's particles end (exclusive).
    """

    def __init__(self, num_cells: int) -> None:
        self.num_cells = num_cells
        self.cell_size = 1 / num_cells

        self.order = np.array([])
        self.cell_starts = np.array([])
        self.cell_ends = np.array([])

        self.all_cells = np.arange(self.num_cells**2, dtype=int)

    def get_offsets(self) -> np.ndarray:
        dirs = [-1, 0, 1]
        return np.array([[dx, dy] for dx in dirs for dy in dirs])  # (9, 2)

    def get_cell_ids(self, positions: np.ndarray) -> np.ndarray:
        """Take an array of particle positions and determine in which cells they are."""
        # convert cartesian coordinates to cell coordinates to cell ids: (x, y) -> (cx, cy) -> c_id
        cell_coords = (positions // self.cell_size).astype(int)
        return cell_coords[:, 0] * self.num_cells + cell_coords[:, 1]

    def build_from_ids(self, cell_ids: np.ndarray) -> None:
        self.order = np.argsort(cell_ids)
        sorted_ids = cell_ids[self.order]

        self.cell_starts = np.searchsorted(sorted_ids, self.all_cells, side="left")
        self.cell_ends = np.searchsorted(sorted_ids, self.all_cells, side="right")

    def build(self, positions: np.ndarray) -> None:
        """Rebuild the grid from current particle positions."""
        cell_ids = self.get_cell_ids(positions)

        self.build_from_ids(cell_ids)

    def get_particles_in_cell(self, cell_id: int) -> np.ndarray:
        """Return the original particle indices of all particles currently in the given cell.

        Returns:
            (k,) array of original (= not sorted) particle indices belonging to `cell_id`,
            where k is however many particles are in that cell (possibly 0).
        """
        return self.order[self.cell_starts[cell_id] : self.cell_ends[cell_id]]

    def get_cell_neighbours(self, cell_id: int) -> np.ndarray:
        """Given a cell id, return the neighbouring cell ids (including the cell itself)."""
        cell_coords = np.array([cell_id // self.num_cells, cell_id % self.num_cells])

        neighbours = (cell_coords[None, :] + self.get_offsets()) % self.num_cells
        return (neighbours[..., 0] * self.num_cells + neighbours[..., 1]).astype(int)  # (9,)
