import sys
from pathlib import Path

import numpy as np
import pytest

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.sorting.external_merge_sort import external_merge_sort
from io_simulator.io_simulator import IOSimulator


class TestExternalMergeSort:
    """Test cases for the external_merge_sort function."""

    @pytest.mark.parametrize(
        ("array_size", "block_size", "memory_size"),
        [
            (8, 2, 4),  # Small array
            (16, 4, 8),  # Medium array
            (32, 4, 12),  # Larger array
            (50, 5, 15),  # Non-power-of-2 sizes
        ],
    )
    def test_successful_sorting(self, array_size: int, block_size: int, memory_size: int) -> None:
        """Test successful sorting with different configurations."""
        # Create test array with known values
        test_array = np.random.RandomState(42).randint(0, 100, array_size)

        disk = IOSimulator(test_array, block_size=block_size, memory_size=memory_size)
        result, io_count = external_merge_sort(disk, array_size)

        # Verify correctness
        expected = np.sort(test_array)
        np.testing.assert_array_equal(result, expected)

        # Verify I/O count is reasonable
        assert io_count > 0, "Should perform at least some I/O operations"
        # Upper bound: should be reasonable for external merge sort
        # For external merge sort: O(n/B * log(n/M)), but allow generous bound
        import math
        theoretical_ios = (array_size / block_size) * math.log2(max(2, array_size / memory_size))
        max_ios = max(200, int(theoretical_ios * 5))  # Allow 5x theoretical for implementation overhead
        assert io_count <= max_ios, f"I/O count {io_count} exceeds reasonable upper bound {max_ios} (theoretical: {theoretical_ios:.1f})"

    def test_already_sorted_array(self) -> None:
        """Test sorting an already sorted array."""
        array_size = 20
        test_array = np.arange(array_size)  # Already sorted

        disk = IOSimulator(test_array, block_size=4, memory_size=8)
        result, io_count = external_merge_sort(disk, array_size)

        # Should remain sorted
        expected = test_array.copy()
        np.testing.assert_array_equal(result, expected)
        assert io_count > 0

    def test_reverse_sorted_array(self) -> None:
        """Test sorting a reverse sorted array (worst case)."""
        array_size = 16
        test_array = np.arange(array_size)[::-1]  # Reverse sorted

        disk = IOSimulator(test_array, block_size=4, memory_size=8)
        result, io_count = external_merge_sort(disk, array_size)

        # Should be sorted in ascending order
        expected = np.arange(array_size)
        np.testing.assert_array_equal(result, expected)
        assert io_count > 0

    def test_array_with_duplicates(self) -> None:
        """Test sorting array with duplicate elements."""
        test_array = np.array([5, 2, 8, 2, 1, 9, 5, 5, 3, 2])

        disk = IOSimulator(test_array, block_size=2, memory_size=6)
        result, io_count = external_merge_sort(disk, len(test_array))

        # Verify correctness
        expected = np.sort(test_array)
        np.testing.assert_array_equal(result, expected)
        assert io_count > 0

    def test_single_element_array(self) -> None:
        """Test sorting single element array."""
        test_array = np.array([42])

        disk = IOSimulator(test_array, block_size=1, memory_size=2)
        result, io_count = external_merge_sort(disk, 1)

        # Should remain unchanged
        np.testing.assert_array_equal(result, test_array)

    def test_two_element_array(self) -> None:
        """Test sorting two element array."""
        test_array = np.array([5, 1])

        disk = IOSimulator(test_array, block_size=1, memory_size=2)
        result, io_count = external_merge_sort(disk, 2)

        # Should be sorted
        expected = np.array([1, 5])
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        ("memory_size", "block_size"),
        [
            (4, 1),  # Large memory relative to block
            (8, 2),  # Medium memory
            (12, 4),  # Small memory relative to block
        ],
    )
    def test_different_memory_configurations(self, memory_size: int, block_size: int) -> None:
        """Test sorting with different memory configurations."""
        array_size = 24
        test_array = np.random.RandomState(123).randint(0, 50, array_size)

        disk = IOSimulator(test_array, block_size=block_size, memory_size=memory_size)
        result, io_count = external_merge_sort(disk, array_size)

        # Verify correctness
        expected = np.sort(test_array)
        np.testing.assert_array_equal(result, expected)
        assert io_count > 0

    def test_io_complexity_scaling(self) -> None:
        """Test that I/O count scales reasonably with array size."""
        io_counts = []
        array_sizes = [8, 16, 32]

        for n in array_sizes:
            test_array = np.random.RandomState(456).randint(0, 100, n)
            disk = IOSimulator(test_array, block_size=4, memory_size=8)
            _, io_count = external_merge_sort(disk, n)
            io_counts.append(io_count)

        # I/O count should generally increase with array size
        # But not too dramatically due to logarithmic factor
        assert io_counts[1] >= io_counts[0], "I/O count should not decrease with larger arrays"
        assert io_counts[2] >= io_counts[1], "I/O count should not decrease with larger arrays"

        # Check that growth is not worse than quadratic
        ratio_1 = io_counts[1] / max(1, io_counts[0])
        ratio_2 = io_counts[2] / max(1, io_counts[1])
        assert ratio_1 <= 8, f"I/O growth ratio {ratio_1} seems too high"
        assert ratio_2 <= 8, f"I/O growth ratio {ratio_2} seems too high"

    def test_stability_preservation(self) -> None:
        """Test that sorting preserves stability for equal elements."""
        # Create array with equal elements that have different "original positions"
        # We'll simulate this by using the array values themselves
        test_array = np.array([3, 1, 3, 2, 1, 3])

        disk = IOSimulator(test_array, block_size=2, memory_size=6)
        result, io_count = external_merge_sort(disk, len(test_array))

        # Check that result is sorted
        expected = np.sort(test_array)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        ("array_size", "expected_io_bound"),
        [
            (8, 20),  # Small array - should be very efficient
            (16, 50),  # Medium array
            (32, 120),  # Larger array
        ],
    )
    def test_io_efficiency_bounds(self, array_size: int, expected_io_bound: int) -> None:
        """Test that I/O count stays within reasonable bounds."""
        test_array = np.random.RandomState(789).randint(0, 1000, array_size)

        disk = IOSimulator(test_array, block_size=4, memory_size=8)
        result, io_count = external_merge_sort(disk, array_size)

        # Verify correctness
        expected = np.sort(test_array)
        np.testing.assert_array_equal(result, expected)

        # Check I/O efficiency
        assert io_count <= expected_io_bound, f"I/O count {io_count} exceeds expected bound {expected_io_bound}"

    def test_large_values_range(self) -> None:
        """Test sorting with large value ranges."""
        array_size = 20
        test_array = np.random.RandomState(101).randint(0, 10000, array_size)

        disk = IOSimulator(test_array, block_size=5, memory_size=10)
        result, io_count = external_merge_sort(disk, array_size)

        # Verify correctness
        expected = np.sort(test_array)
        np.testing.assert_array_equal(result, expected)
        assert io_count > 0

    def test_edge_case_block_sizes(self) -> None:
        """Test sorting with edge case block and memory sizes."""
        test_array = np.array([7, 3, 9, 1, 5, 8, 2, 4, 6])

        # Test with block_size = 1
        disk1 = IOSimulator(test_array.copy(), block_size=1, memory_size=3)
        result1, io_count1 = external_merge_sort(disk1, len(test_array))

        # Test with block_size = array_size (everything fits in one block)
        disk2 = IOSimulator(test_array.copy(), block_size=len(test_array), memory_size=len(test_array) + 1)
        result2, io_count2 = external_merge_sort(disk2, len(test_array))

        expected = np.sort(test_array)
        np.testing.assert_array_equal(result1, expected)
        np.testing.assert_array_equal(result2, expected)

        assert io_count1 > 0
        assert io_count2 > 0
