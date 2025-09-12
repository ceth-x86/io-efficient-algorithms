import math
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from io_simulator.io_simulator import IOSimulator

# Error messages
_ELEMENT_WISE_NOT_IMPLEMENTED = "Element-wise transpose not implemented for tile_size == 1"


def transpose_cache_aware(disk: IOSimulator, n_rows: int, n_cols: int) -> tuple[np.ndarray, int]:
    # Check if matrix is square
    if n_rows != n_cols:
        error_msg = "Matrix must be square for in-place transpose"
        raise ValueError(error_msg)

    # For square matrices, we can do in-place transpose
    # Cache-aware tile size calculation: t = sqrt(M) - B
    # This accounts for "protruding" blocks in row-major storage
    B = disk.block_size  # noqa: N806
    M = disk.memory_size  # noqa: N806

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
                tile = disk.get_submatrix(i, i_end, j, j_end, n_cols)
                tile_transposed = tile.T
                disk.set_submatrix(i, j, tile_transposed, n_cols)

            else:
                # Symmetric tile pair - swap and transpose
                tile1 = disk.get_submatrix(i, i_end, j, j_end, n_cols)
                tile2 = disk.get_submatrix(j, j_end, i, i_end, n_cols)

                # Swap and transpose
                disk.set_submatrix(i, j, tile2.T, n_cols)
                disk.set_submatrix(j, i, tile1.T, n_cols)

            # Flush after each tile pair to ensure changes are persisted
            disk.flush_memory()

    # Flush all changes to disk at the end
    disk.flush_memory()
    return disk.disk, disk.io_count


# Example
if __name__ == "__main__":
    A = np.arange(16).reshape(4, 4)
    print("Original:")  # noqa: T201
    print(A)  # noqa: T201

    disk = IOSimulator(A, block_size=4, memory_size=64)
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
