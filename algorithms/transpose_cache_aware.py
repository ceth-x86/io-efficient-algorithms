import math
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from io_simulator.io_simulator import IOSimulator


def transpose_cache_aware(sim: IOSimulator) -> tuple[np.ndarray, int]:
    n = sim.n
    m = sim.disk.shape[1]

    # Check if matrix is square
    if n != m:
        error_msg = "Matrix must be square for in-place transpose"
        raise ValueError(error_msg)

    # For square matrices, we can do in-place transpose
    # Cache-aware tile size calculation: t = sqrt(M) - B
    # This accounts for "protruding" blocks in row-major storage
    B = sim.block_size
    M = sim.memory_size

    # Calculate tile size according to the cache-aware algorithm
    # Condition: 2*(t^2 + 2*B*t) <= M
    # Approximation: t = sqrt(M) - B
    tile_size = int(math.sqrt(M)) - B
    if tile_size <= 0:
        tile_size = 1

    for i in range(0, n, tile_size):
        for j in range(i, n, tile_size):  # Start from i to avoid processing pairs twice
            i_end = min(i + tile_size, n)
            j_end = min(j + tile_size, n)

            if i == j:
                # Diagonal tile - transpose in-place
                tile = sim.get_submatrix(i, i_end, j, j_end)
                tile_transposed = tile.T
                sim.set_submatrix(i, j, tile_transposed)

            else:
                # Symmetric tile pair - swap and transpose
                tile1 = sim.get_submatrix(i, i_end, j, j_end)
                tile2 = sim.get_submatrix(j, j_end, i, i_end)

                # Swap and transpose
                sim.set_submatrix(i, j, tile2.T)
                sim.set_submatrix(j, i, tile1.T)

            # Flush memory after each tile to ensure changes are written
            sim.flush_memory()

    return sim.disk, sim.io_count


# Example
if __name__ == "__main__":
    A = np.arange(16).reshape(4, 4)
    print("Original:")
    print(A)

    sim = IOSimulator(A, block_size=2, memory_size=8)
    AT, io_count = transpose_cache_aware(sim)

    print("\nTransposed:")
    print(AT)
    print(f"\nNumber of I/O operations: {io_count}")

    # Verify correctness
    expected_transpose = A.T
    if np.all(expected_transpose == AT):
        print("✓ Transpose is correct!")
    else:
        print("✗ Transpose is incorrect!")
        print("Expected:")
        print(expected_transpose)
