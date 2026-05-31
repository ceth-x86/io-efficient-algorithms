from .io_simulator import IOSimulator

class VirtualMatrix:
    """
    Represents a 2D matrix stored contiguously on the virtual disk in Row-Major layout.
    """
    def __init__(self, sim: IOSimulator, rows: int, cols: int):
        self.sim = sim
        self.rows = rows
        self.cols = cols
        self.start_idx = sim.disk.allocate(rows * cols)

    def close(self):
        if self.start_idx is not None:
            self.sim.disk.free(self.start_idx, self.rows * self.cols)
            self.start_idx = None

    def _get_flat_index(self, r: int, c: int) -> int:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise IndexError("Matrix index out of bounds!")
        return self.start_idx + r * self.cols + c

    def get_element(self, r: int, c: int) -> int:
        flat_idx = self._get_flat_index(r, c)
        return self.sim.read_element(flat_idx)

    def set_element(self, r: int, c: int, value: int) -> None:
        flat_idx = self._get_flat_index(r, c)
        self.sim.write_element(flat_idx, value)

    def get_submatrix(self, r_start: int, r_end: int, c_start: int, c_end: int) -> list:
        """
        Reads a submatrix into memory.
        """
        sub_rows = r_end - r_start
        sub_cols = c_end - c_start
        submat = []
        for r in range(sub_rows):
            row_data = []
            for c in range(sub_cols):
                row_data.append(self.get_element(r_start + r, c_start + c))
            submat.append(row_data)
        return submat

    def set_submatrix(self, r_start: int, c_start: int, submat: list) -> None:
        """
        Writes a submatrix from memory back to the virtual disk.
        """
        for r in range(len(submat)):
            for c in range(len(submat[r])):
                self.set_element(r_start + r, c_start + c, submat[r][c])
