import sys
from pathlib import Path

import numpy as np
import pytest

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.transpose.cache_aware import transpose_cache_aware
from io_simulator.io_simulator import IOSimulator


class TestTransposeCacheAware:
    """Test cases for the transpose_cache_aware function."""

    @pytest.mark.parametrize(
        ("matrix_size", "block_size", "memory_size", "expected_io_count"),
        [
            (4, 2, 16, 16),  # 4x4 matrix, tile_size=2
            (4, 2, 32, 24),  # 4x4 matrix with large memory, tile_size=3
            (3, 1, 9, 18),  # 3x3 matrix, tile_size=2
            (5, 1, 16, 50),  # 5x5 matrix, tile_size=3
        ],
    )
    def test_successful_transpose(
        self, matrix_size: int, block_size: int, memory_size: int, expected_io_count: int
    ) -> None:
        """Test successful transpose operations with different configurations."""
        # Create test matrix
        if matrix_size == 3:
            a = np.arange(9).reshape(3, 3)
        elif matrix_size == 4:
            a = np.arange(16).reshape(4, 4)
        elif matrix_size == 5:
            a = np.arange(25).reshape(5, 5)
        else:
            a = np.arange(matrix_size * matrix_size).reshape(matrix_size, matrix_size)

        disk = IOSimulator(a, block_size=block_size, memory_size=memory_size)
        result_flat, io_count = transpose_cache_aware(disk, matrix_size, matrix_size)
        result = result_flat.reshape(matrix_size, matrix_size)

        # Verify correctness
        expected = a.T
        np.testing.assert_array_equal(result, expected)
        assert io_count == expected_io_count

    @pytest.mark.parametrize(
        ("matrix_size", "block_size", "memory_size"),
        [
            (2, 1, 4),  # 2x2 matrix, tile_size=1
            (2, 1, 0),  # 2x2 matrix, zero memory
            (2, 1, 1),  # 2x2 matrix, very small memory
            (2, 4, 16),  # 2x2 matrix, large block size
            (1, 1, 1),  # 1x1 matrix
            (8, 4, 32),  # 8x8 matrix, tile_size=1
            (3, 3, 9),  # 3x3 matrix, tile_size=1
        ],
    )
    def test_not_implemented_error(self, matrix_size: int, block_size: int, memory_size: int) -> None:
        """Test cases that should raise NotImplementedError for tile_size == 1."""
        # Create test matrix
        if matrix_size == 1:
            a = np.array([[42]])
        elif matrix_size == 2:
            a = np.array([[1, 2], [3, 4]])
        elif matrix_size == 3:
            a = np.arange(9).reshape(3, 3)
        elif matrix_size == 8:
            a = np.arange(64).reshape(8, 8)
        else:
            a = np.arange(matrix_size * matrix_size).reshape(matrix_size, matrix_size)

        disk = IOSimulator(a, block_size=block_size, memory_size=memory_size)

        with pytest.raises(NotImplementedError, match="Element-wise transpose not implemented"):
            transpose_cache_aware(disk, matrix_size, matrix_size)

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
            transpose_cache_aware(disk, rows, cols)

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
        result_flat, io_count = transpose_cache_aware(disk, 3, 3)
        result = result_flat.reshape(3, 3)

        expected = matrix_data.T
        np.testing.assert_array_equal(result, expected)
        assert io_count == 18

    def test_different_block_sizes(self) -> None:
        """Test transpose with different block sizes."""
        a = np.arange(9).reshape(3, 3)

        # Test with block_size=1 (should work)
        disk1 = IOSimulator(a.copy(), block_size=1, memory_size=9)
        result1_flat, io_count1 = transpose_cache_aware(disk1, 3, 3)
        result1 = result1_flat.reshape(3, 3)

        # Test with block_size=3 (should raise NotImplementedError)
        disk2 = IOSimulator(a.copy(), block_size=3, memory_size=9)
        with pytest.raises(NotImplementedError, match="Element-wise transpose not implemented"):
            transpose_cache_aware(disk2, 3, 3)

        expected = a.T
        np.testing.assert_array_equal(result1, expected)
        assert io_count1 == 18

    def test_different_memory_sizes(self) -> None:
        """Test transpose with different memory sizes."""
        a = np.arange(16).reshape(4, 4)

        # Test with small memory (should raise NotImplementedError)
        disk1 = IOSimulator(a.copy(), block_size=2, memory_size=8)
        with pytest.raises(NotImplementedError, match="Element-wise transpose not implemented"):
            transpose_cache_aware(disk1, 4, 4)

        # Test with large memory (should work)
        disk2 = IOSimulator(a.copy(), block_size=2, memory_size=32)
        result2_flat, io_count2 = transpose_cache_aware(disk2, 4, 4)
        result2 = result2_flat.reshape(4, 4)

        expected = a.T
        np.testing.assert_array_equal(result2, expected)
        assert io_count2 == 24

    @pytest.mark.parametrize(
        ("matrix_size", "block_size", "memory_size", "description"),
        [
            (4, 2, 8, "small_memory"),
            (4, 2, 16, "medium_memory"),
            (4, 2, 32, "large_memory"),
            (4, 2, 64, "very_large_memory"),
        ],
    )
    def test_memory_size_impact(self, matrix_size: int, block_size: int, memory_size: int, description: str) -> None:
        """Test how different memory sizes affect the algorithm."""
        a = np.arange(16).reshape(4, 4)
        disk = IOSimulator(a, block_size=block_size, memory_size=memory_size)

        # Calculate expected tile size
        tile_size = int(np.sqrt(memory_size)) - block_size
        if tile_size <= 0:
            tile_size = 1

        if tile_size == 1:
            # Should raise NotImplementedError
            with pytest.raises(NotImplementedError, match="Element-wise transpose not implemented"):
                transpose_cache_aware(disk, 4, 4)
        else:
            # Should work
            result_flat, io_count = transpose_cache_aware(disk, 4, 4)
            result = result_flat.reshape(4, 4)

            expected = a.T
            np.testing.assert_array_equal(result, expected)

            # Expected I/O counts for different memory configurations
            expected_io_counts = {
                (4, 2, 16): 16,  # medium_memory
                (4, 2, 32): 24,  # large_memory
                (4, 2, 64): 16,  # very_large_memory
            }
            expected_io = expected_io_counts.get((matrix_size, block_size, memory_size), io_count)
            assert io_count == expected_io, f"Expected {expected_io} I/O operations for {description}, got {io_count}"

    @pytest.mark.parametrize(
        ("matrix_size", "block_size", "memory_size", "expected_io_count"),
        [
            (3, 1, 9, 18),  # 3x3, tile_size=2
            (4, 2, 16, 16),  # 4x4, tile_size=2
            (4, 2, 32, 24),  # 4x4, tile_size=3
            (5, 1, 16, 50),  # 5x5, tile_size=3
        ],
    )
    def test_io_count_accuracy(
        self, matrix_size: int, block_size: int, memory_size: int, expected_io_count: int
    ) -> None:
        """Test that I/O count is accurate for known configurations."""
        # Create test matrix
        if matrix_size == 3:
            a = np.arange(9).reshape(3, 3)
        elif matrix_size == 4:
            a = np.arange(16).reshape(4, 4)
        elif matrix_size == 5:
            a = np.arange(25).reshape(5, 5)
        else:
            a = np.arange(matrix_size * matrix_size).reshape(matrix_size, matrix_size)

        disk = IOSimulator(a, block_size=block_size, memory_size=memory_size)
        result_flat, io_count = transpose_cache_aware(disk, matrix_size, matrix_size)
        result = result_flat.reshape(matrix_size, matrix_size)

        # Verify correctness
        expected = a.T
        np.testing.assert_array_equal(result, expected)
        assert io_count == expected_io_count
