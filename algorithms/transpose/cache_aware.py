import math
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from io_simulator.io_simulator import IOSimulator

# Error messages
_ELEMENT_WISE_NOT_IMPLEMENTED = "Element-wise transpose not implemented for tile_size == 1"


class TransposeDiskWrapper:
    def __init__(self, disk: IOSimulator):
        self.disk = disk
        self.block_size = disk.block_size
        self.memory_size = disk.memory_size
        self.memory_limit = disk.memory_limit

    @property
    def io_count(self):
        return self.disk.io_count

    @io_count.setter
    def io_count(self, val):
        self.disk.io_count = val

    def get_element(self, i: int, j: int, n_cols: int) -> float:
        return self.disk.read_element(i * n_cols + j)

    def set_element(self, i: int, j: int, value: float, n_cols: int) -> None:
        self.disk.write_element(i * n_cols + j, value)

    def get_submatrix(self, i_start: int, i_end: int, j_start: int, j_end: int, n_cols: int) -> np.ndarray:
        rows = i_end - i_start
        cols = j_end - j_start
        result = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                result[r, c] = self.get_element(i_start + r, j_start + c, n_cols)
        return result

    def set_submatrix(self, i_start: int, j_start: int, submat: np.ndarray, n_cols: int) -> None:
        rows, cols = submat.shape
        for r in range(rows):
            for c in range(cols):
                self.set_element(i_start + r, j_start + c, submat[r, c], n_cols)

    def flush_memory(self) -> None:
        self.disk.flush_memory()


def transpose_cache_aware(disk: IOSimulator, n_rows: int, n_cols: int) -> tuple[np.ndarray, int]:
    # Check if matrix is square
    if n_rows != n_cols:
        error_msg = "Matrix must be square for in-place transpose"
        raise ValueError(error_msg)

    wdisk = TransposeDiskWrapper(disk)

    # For square matrices, we can do in-place transpose
    # Cache-aware tile size calculation: t = sqrt(M) - B
    # This accounts for "protruding" blocks in row-major storage
    B = wdisk.block_size  # noqa: N806
    M = wdisk.memory_size  # noqa: N806

    """
    Calculate tile size according to the cache-aware algorithm.
    Condition: 2*(t^2 + 2*B*t) <= M
    Approximation: t = sqrt(M) - B
    """
    tile_size = int(math.sqrt(M)) - B
    if tile_size <= 0:
        tile_size = 1

    if tile_size == 1:
        raise NotImplementedError(_ELEMENT_WISE_NOT_IMPLEMENTED)

    for i in range(0, n_rows, tile_size):
        for j in range(i, n_rows, tile_size):  # Start from i to avoid processing pairs twice
            i_end = min(i + tile_size, n_rows)
            j_end = min(j + tile_size, n_rows)

            if i == j:
                # Diagonal tile - transpose in-place
                tile = wdisk.get_submatrix(i, i_end, j, j_end, n_cols)
                tile_transposed = tile.T
                wdisk.set_submatrix(i, j, tile_transposed, n_cols)

            else:
                # Symmetric tile pair - swap and transpose
                tile1 = wdisk.get_submatrix(i, i_end, j, j_end, n_cols)
                tile2 = wdisk.get_submatrix(j, j_end, i, i_end, n_cols)

                # Swap and transpose
                wdisk.set_submatrix(i, j, tile2.T, n_cols)
                wdisk.set_submatrix(j, i, tile1.T, n_cols)

            # Flush after each tile pair to ensure changes are persisted
            wdisk.flush_memory()

    # Flush all changes to disk at the end
    wdisk.flush_memory()
    return np.array(disk.disk.disk[:n_rows*n_cols]), wdisk.io_count


# Example
if __name__ == "__main__":
    A = np.arange(16).reshape(4, 4)
    print("Original:")  # noqa: T201
    print(A)  # noqa: T201

    from io_simulator import VirtualDisk
    vd = VirtualDisk(size=16)
    vd.disk = list(A.flatten())
    disk = IOSimulator(vd, block_size=4, cache_memory_size=64)
    AT_flat, io_count = transpose_cache_aware(disk, 4, 4)

    # Reshape back to matrix
    AT = AT_flat.reshape(4, 4)

    print("\nTransposed:")  # noqa: T201
    print(AT)  # noqa: T201
    print(f"\nNumber of I/O operations: {io_count}")  # noqa: T201

    # Verify correctness
    expected_transpose = A.T
    if np.all(expected_transpose == AT):
        print("✓ Transpose is correct!")  # noqa: T201
    else:
        print("✗ Transpose is incorrect!")  # noqa: T201
        print("Expected:")  # noqa: T201
        print(expected_transpose)  # noqa: T201
