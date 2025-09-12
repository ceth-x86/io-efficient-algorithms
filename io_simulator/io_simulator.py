import numpy as np


class IOSimulator:
    def __init__(self, matrix: np.ndarray, block_size: int, memory_size: int) -> None:
        self.disk = matrix.copy()
        self.n = matrix.shape[0]
        self.block_size = block_size
        self.memory_size = memory_size
        self.io_count = 0
        self.memory = {}
        self.memory_limit = max(1, memory_size // block_size)

    def read_block(self, row: int, block_id: int) -> np.ndarray:
        if (row, block_id) in self.memory:
            return self.memory[(row, block_id)]
        if len(self.memory) >= self.memory_limit:
            self.memory.pop(next(iter(self.memory)))
        start = block_id * self.block_size
        end = min(start + self.block_size, self.disk.shape[1])  # Use actual number of columns
        block = self.disk[row, start:end].copy()
        self.memory[(row, block_id)] = block
        self.io_count += 1
        return block

    def write_block(self, row: int, block_id: int) -> None:
        if (row, block_id) not in self.memory:
            return
        start = block_id * self.block_size
        end = min(start + self.block_size, self.disk.shape[1])  # Use actual number of columns
        self.disk[row, start:end] = self.memory[(row, block_id)]
        self.io_count += 1

    def get_submatrix(self, i_start: int, i_end: int, j_start: int, j_end: int) -> np.ndarray:
        """Read submatrix into numpy array (with I/O)."""
        return np.array([[self.get_element(i, j) for j in range(j_start, j_end)] for i in range(i_start, i_end)])

    def set_submatrix(self, i_start: int, j_start: int, submat: np.ndarray) -> None:
        """Write numpy array back (with I/O)."""
        rows, cols = submat.shape

        # Write each row as a block to be more efficient
        for x in range(rows):
            row = i_start + x
            for y in range(cols):
                col = j_start + y
                block_id = col // self.block_size
                # Read the block if not in memory
                if (row, block_id) not in self.memory:
                    self.read_block(row, block_id)
                # Modify the element in the block
                col_in_block = col % self.block_size
                self.memory[(row, block_id)][col_in_block] = submat[x, y]

    def get_element(self, i: int, j: int) -> float:
        block_id = j // self.block_size
        block = self.read_block(i, block_id)
        return block[j % self.block_size]

    def set_element(self, i: int, j: int, value: float) -> None:
        block_id = j // self.block_size
        block = self.read_block(i, block_id)
        block[j % self.block_size] = value
        self.memory[(i, block_id)] = block

    def flush_memory(self) -> None:
        for row, block_id in list(self.memory.keys()):
            self.write_block(row, block_id)
        self.memory.clear()
