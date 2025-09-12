import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from io_simulator.io_simulator import IOSimulator

# Error messages
_MATRIX_MUST_BE_SQUARE = "Matrix must be square"


def transpose_cache_oblivious(disk: IOSimulator, n_rows: int, n_cols: int) -> tuple[np.ndarray, int]:
    """
    Cache-oblivious matrix transpose using recursive divide-and-conquer.

    The algorithm recursively divides the matrix into 4 submatrices:
    - Top-left and bottom-right are processed recursively (diagonal parts)
    - Top-right and bottom-left are swapped and transposed recursively

    Args:
        disk: IOSimulator instance containing the matrix
        n_rows: Number of rows in the matrix
        n_cols: Number of columns in the matrix

    Returns:
        Tuple of (transposed_matrix_flat, io_count)

    Raises:
        ValueError: If matrix is not square
    """
    if n_rows != n_cols:
        raise ValueError(_MATRIX_MUST_BE_SQUARE)

    # Reset I/O count to track only the recursive operations
    initial_io_count = disk.io_count

    # Start the recursive transpose (modifies the matrix in-place)
    _transpose_recursive(disk, 0, n_rows, 0, n_cols, n_cols)

    # For very small matrices, ensure we flush and do at least some I/O
    disk.flush_memory()

    # Calculate total I/O operations performed
    io_count = disk.io_count - initial_io_count

    # Return the transposed matrix (now stored in disk.disk)
    return disk.disk.copy(), io_count


def _transpose_recursive(
    disk: IOSimulator, row_start: int, row_end: int, col_start: int, col_end: int, n_cols: int
) -> None:
    """
    Recursively transpose a submatrix using cache-oblivious approach.
    """
    n_rows = row_end - row_start
    n_cols_sub = col_end - col_start

    # Base case: if submatrix is 1x1, it's already transposed
    if n_rows <= 1 and n_cols_sub <= 1:
        # For 1x1, we still need to "read" it to simulate I/O cost
        if n_rows == 1 and n_cols_sub == 1:
            _ = disk.get_element(row_start, col_start, n_cols)
        return

    # Base case: if submatrix is small enough, transpose in place
    if n_rows <= 2 and n_cols_sub <= 2:
        _transpose_small_matrix(disk, row_start, row_end, col_start, col_end, n_cols)
        return

    # Recursive case: divide into 4 submatrices
    mid_row = row_start + n_rows // 2
    mid_col = col_start + n_cols_sub // 2

    # Process diagonal parts recursively (top-left and bottom-right)
    _transpose_recursive(disk, row_start, mid_row, col_start, mid_col, n_cols)
    _transpose_recursive(disk, mid_row, row_end, mid_col, col_end, n_cols)

    # Process off-diagonal parts: swap and transpose
    _swap_and_transpose(disk, row_start, mid_row, mid_col, col_end, mid_row, row_end, col_start, mid_col, n_cols)


def _transpose_small_matrix(
    disk: IOSimulator, row_start: int, row_end: int, col_start: int, col_end: int, n_cols: int
) -> None:
    """
    Transpose a small matrix (2x2 or smaller) in place.
    """
    n_rows = row_end - row_start
    n_cols_sub = col_end - col_start

    # For 1x1 matrix, nothing to do
    if n_rows <= 1 and n_cols_sub <= 1:
        return

    # For 2x2 matrix, swap off-diagonal elements
    if n_rows == 2 and n_cols_sub == 2:
        # Get off-diagonal elements
        val01 = disk.get_element(row_start + 0, col_start + 1, n_cols)
        val10 = disk.get_element(row_start + 1, col_start + 0, n_cols)

        # Swap off-diagonal elements (diagonal elements stay the same)
        disk.set_element(row_start + 0, col_start + 1, val10, n_cols)
        disk.set_element(row_start + 1, col_start + 0, val01, n_cols)
        disk.flush_memory()  # Ensure changes are persisted
        return

    # For other small matrices, transpose element by element
    for i in range(n_rows):
        for j in range(i + 1, n_cols_sub):  # Only process upper triangle
            # Check bounds to avoid swapping with elements outside the submatrix
            if row_start + j < row_end and col_start + i < col_end:
                # Get elements
                val1 = disk.get_element(row_start + i, col_start + j, n_cols)
                val2 = disk.get_element(row_start + j, col_start + i, n_cols)

                # Swap them
                disk.set_element(row_start + i, col_start + j, val2, n_cols)
                disk.set_element(row_start + j, col_start + i, val1, n_cols)

    # Flush changes at the end
    disk.flush_memory()


def _swap_and_transpose(
    disk: IOSimulator,
    row1_start: int,
    row1_end: int,
    col1_start: int,
    col1_end: int,
    row2_start: int,
    row2_end: int,
    col2_start: int,
    col2_end: int,
    n_cols: int,
) -> None:
    """
    Swap two submatrices and transpose them.

    This function swaps the top-right and bottom-left submatrices,
    transposing each in the process.
    """
    # Read both submatrices
    submatrix1 = disk.get_submatrix(row1_start, row1_end, col1_start, col1_end, n_cols)
    submatrix2 = disk.get_submatrix(row2_start, row2_end, col2_start, col2_end, n_cols)

    # Swap and transpose: submatrix1 goes to position 2 transposed,
    # submatrix2 goes to position 1 transposed
    disk.set_submatrix(row1_start, col1_start, submatrix2.T, n_cols)
    disk.set_submatrix(row2_start, col2_start, submatrix1.T, n_cols)
    disk.flush_memory()  # Ensure changes are persisted


# Example usage and testing
if __name__ == "__main__":
    # Test with a small matrix
    print("Testing cache-oblivious transpose...")

    # Create a test matrix
    n = 4
    A = np.arange(n * n).reshape(n, n)
    print(f"Original matrix ({n}x{n}):")
    print(A)

    # Create IOSimulator
    disk = IOSimulator(A, block_size=2, memory_size=8)

    # Perform transpose
    result_flat, io_count = transpose_cache_oblivious(disk, n, n)
    result = result_flat.reshape(n, n)

    print("\nTransposed matrix:")
    print(result)
    print(f"I/O count: {io_count}")

    # Verify correctness
    expected = A.T
    if np.array_equal(result, expected):
        print("✓ Transpose is correct!")
    else:
        print("✗ Transpose is incorrect!")
        print("Expected:")
        print(expected)
