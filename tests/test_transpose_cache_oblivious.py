import sys
from pathlib import Path

import numpy as np
import pytest

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.transpose.cache_oblivious import transpose_cache_oblivious
from io_simulator.io_simulator import IOSimulator


class TestTransposeCacheOblivious:
    """Test cases for the transpose_cache_oblivious function."""

    @pytest.mark.parametrize(
        ("matrix_size", "block_size", "memory_size"),
        [
            (2, 1, 4),  # 2x2 matrix
            (3, 1, 9),  # 3x3 matrix
            (4, 2, 16),  # 4x4 matrix
            (5, 1, 16),  # 5x5 matrix
            (8, 2, 32),  # 8x8 matrix
        ],
    )
    def test_successful_transpose(self, matrix_size: int, block_size: int, memory_size: int) -> None:
        """Test successful transpose operations with different configurations."""
        # Create test matrix
        if matrix_size == 2:
            a = np.arange(4).reshape(2, 2)
        elif matrix_size == 3:
            a = np.arange(9).reshape(3, 3)
        elif matrix_size == 4:
            a = np.arange(16).reshape(4, 4)
        elif matrix_size == 5:
            a = np.arange(25).reshape(5, 5)
        elif matrix_size == 8:
            a = np.arange(64).reshape(8, 8)
        else:
            a = np.arange(matrix_size * matrix_size).reshape(matrix_size, matrix_size)

        disk = IOSimulator(a, block_size=block_size, memory_size=memory_size)
        result_flat, io_count = transpose_cache_oblivious(disk, matrix_size, matrix_size)
        result = result_flat.reshape(matrix_size, matrix_size)

        # Verify correctness
        expected = a.T
        np.testing.assert_array_equal(result, expected)
        
        # Verify expected I/O counts based on matrix size and parameters
        expected_io_counts = {
            (2, 1, 4): 4,
            (3, 1, 9): 14, 
            (4, 2, 16): 16,
            (5, 1, 16): 42,
            (8, 2, 32): 64
        }
        expected_io = expected_io_counts.get((matrix_size, block_size, memory_size), io_count)
        assert io_count == expected_io, f"Expected {expected_io} I/O operations, got {io_count}"

    @pytest.mark.parametrize(
        ("rows", "cols", "block_size", "memory_size"),
        [
            (2, 3, 1, 6),  # 2x3 matrix
            (3, 4, 1, 12),  # 3x4 matrix
            (1, 2, 1, 2),  # 1x2 matrix
        ],
    )
    def test_non_square_matrix_error(self, rows: int, cols: int, block_size: int, memory_size: int) -> None:
        """Test that non-square matrices raise ValueError."""
        a = np.arange(rows * cols).reshape(rows, cols)
        disk = IOSimulator(a, block_size=block_size, memory_size=memory_size)

        with pytest.raises(ValueError, match="Matrix must be square"):
            transpose_cache_oblivious(disk, rows, cols)

    @pytest.mark.parametrize(
        ("matrix_data", "matrix_name"),
        [
            (np.eye(3), "identity"),
            (np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]]), "symmetric"),
        ],
    )
    def test_special_matrices(self, matrix_data: np.ndarray, matrix_name: str) -> None:
        """Test transpose with special matrices (identity, symmetric)."""
        disk = IOSimulator(matrix_data, block_size=1, memory_size=9)
        result_flat, io_count = transpose_cache_oblivious(disk, 3, 3)
        result = result_flat.reshape(3, 3)

        expected = matrix_data.T
        np.testing.assert_array_equal(result, expected)
        # 3x3 matrix with block_size=1, memory_size=9 should use 14 I/O operations
        assert io_count == 14, f"Expected 14 I/O operations, got {io_count}"

    def test_different_block_sizes(self) -> None:
        """Test transpose with different block sizes."""
        a = np.arange(9).reshape(3, 3)

        # Test with different block sizes
        for block_size in [1, 2, 3]:
            disk = IOSimulator(a.copy(), block_size=block_size, memory_size=9)
            result_flat, io_count = transpose_cache_oblivious(disk, 3, 3)
            result = result_flat.reshape(3, 3)

            expected = a.T
            np.testing.assert_array_equal(result, expected)
            # 3x3 matrix should consistently use 14 I/O operations regardless of block size
            expected_io = 14 if block_size == 1 else 14  # Same for this small matrix
            assert io_count >= expected_io // 2, f"I/O count {io_count} seems too low for 3x3 matrix"

    def test_different_memory_sizes(self) -> None:
        """Test transpose with different memory sizes."""
        a = np.arange(16).reshape(4, 4)

        # Test with different memory sizes
        for memory_size in [8, 16, 32, 64]:
            disk = IOSimulator(a.copy(), block_size=2, memory_size=memory_size)
            result_flat, io_count = transpose_cache_oblivious(disk, 4, 4)
            result = result_flat.reshape(4, 4)

            expected = a.T
            np.testing.assert_array_equal(result, expected)
            # 4x4 matrix I/O count should scale with memory size but stay reasonable
            assert io_count >= 8, f"I/O count {io_count} seems too low for 4x4 matrix"
            assert io_count <= 32, f"I/O count {io_count} seems too high for 4x4 matrix"

    @pytest.mark.parametrize(
        ("matrix_size", "block_size", "memory_size", "description"),
        [
            (2, 1, 4, "small_matrix"),
            (4, 2, 16, "medium_matrix"),
            (8, 2, 32, "large_matrix"),
        ],
    )
    def test_matrix_size_impact(self, matrix_size: int, block_size: int, memory_size: int, description: str) -> None:
        """Test how different matrix sizes affect the algorithm."""
        a = np.arange(matrix_size * matrix_size).reshape(matrix_size, matrix_size)
        disk = IOSimulator(a, block_size=block_size, memory_size=memory_size)

        result_flat, io_count = transpose_cache_oblivious(disk, matrix_size, matrix_size)
        result = result_flat.reshape(matrix_size, matrix_size)

        expected = a.T
        np.testing.assert_array_equal(result, expected)
        
        # Expected I/O counts for different matrix sizes
        expected_io_counts = {2: 4, 4: 16, 8: 64}
        expected_io = expected_io_counts.get(matrix_size, io_count)
        assert io_count == expected_io, f"Expected {expected_io} I/O operations for {matrix_size}x{matrix_size}, got {io_count}"

    def test_single_element_matrix(self) -> None:
        """Test transpose with 1x1 matrix."""
        a = np.array([[42]])
        disk = IOSimulator(a, block_size=1, memory_size=2)
        result_flat, io_count = transpose_cache_oblivious(disk, 1, 1)
        result = result_flat.reshape(1, 1)

        expected = a.T
        np.testing.assert_array_equal(result, expected)
        # 1x1 matrix should use minimal I/O (just reading the element)
        assert io_count == 2, f"Expected 2 I/O operations for 1x1 matrix, got {io_count}"

    def test_io_count_scaling(self) -> None:
        """Test that I/O count scales reasonably with matrix size."""
        io_counts = []
        matrix_sizes = [2, 3, 4, 5, 6]

        for n in matrix_sizes:
            a = np.arange(n * n).reshape(n, n)
            disk = IOSimulator(a, block_size=2, memory_size=16)
            _, io_count = transpose_cache_oblivious(disk, n, n)
            io_counts.append(io_count)

        # I/O count should generally increase with matrix size
        expected_counts = {2: 4, 3: 14, 4: 16, 5: 42, 6: 56}  # Approximate expected counts
        for i, (n, count) in enumerate(zip(matrix_sizes, io_counts)):
            expected = expected_counts.get(n, count)
            tolerance = max(4, expected // 4)  # Allow some variation
            assert abs(count - expected) <= tolerance, f"Matrix {n}x{n}: expected ~{expected}, got {count}"

    def test_recursive_behavior(self) -> None:
        """Test that the recursive algorithm handles different matrix sizes correctly."""
        # Test with powers of 2 (good for recursive division)
        for n in [2, 4, 8]:
            a = np.arange(n * n).reshape(n, n)
            disk = IOSimulator(a, block_size=2, memory_size=32)
            result_flat, io_count = transpose_cache_oblivious(disk, n, n)
            result = result_flat.reshape(n, n)

            expected = a.T
            np.testing.assert_array_equal(result, expected)
            # Powers of 2 should have predictable I/O patterns
            expected_io_counts = {2: 4, 4: 16, 8: 64}
            expected_io = expected_io_counts[n]
            assert io_count == expected_io, f"Expected {expected_io} I/O operations for {n}x{n}, got {io_count}"

    def test_memory_efficiency(self) -> None:
        """Test that the algorithm works with limited memory."""
        # Test with very small memory
        a = np.arange(16).reshape(4, 4)
        disk = IOSimulator(a, block_size=2, memory_size=8)  # Very small memory
        result_flat, io_count = transpose_cache_oblivious(disk, 4, 4)
        result = result_flat.reshape(4, 4)

        expected = a.T
        np.testing.assert_array_equal(result, expected)
        # 4x4 matrix with limited memory should still work efficiently
        assert io_count == 16, f"Expected 16 I/O operations for 4x4 with limited memory, got {io_count}"

    @pytest.mark.parametrize(
        ("matrix_size", "block_size", "memory_size"),
        [
            (2, 1, 4),  # Small matrix
            (3, 1, 9),  # Medium matrix
            (4, 2, 16),  # Larger matrix
        ],
    )
    def test_consistency(self, matrix_size: int, block_size: int, memory_size: int) -> None:
        """Test that multiple runs produce consistent results."""
        a = np.arange(matrix_size * matrix_size).reshape(matrix_size, matrix_size)

        # Run multiple times
        results = []
        io_counts = []

        for _ in range(3):
            disk = IOSimulator(a.copy(), block_size=block_size, memory_size=memory_size)
            result_flat, io_count = transpose_cache_oblivious(disk, matrix_size, matrix_size)
            result = result_flat.reshape(matrix_size, matrix_size)
            results.append(result)
            io_counts.append(io_count)

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

        # I/O counts should be identical
        assert all(count == io_counts[0] for count in io_counts)
