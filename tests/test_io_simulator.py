import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from io_simulator.io_simulator import IOSimulator


class TestIOSimulator:
    @pytest.fixture
    def sample_matrix_2d(self) -> np.ndarray:
        """Create a sample 4x4 matrix for testing."""
        return np.arange(16).reshape(4, 4)

    @pytest.fixture
    def sample_matrix_1d(self, sample_matrix_2d: np.ndarray) -> np.ndarray:
        """Create a flattened version of the sample matrix."""
        return sample_matrix_2d.flatten()

    @pytest.fixture
    def simulator(self, sample_matrix_2d: np.ndarray) -> IOSimulator:
        """Create a simulator instance for testing."""
        return IOSimulator(sample_matrix_2d, block_size=2, memory_size=8)

    def test_initialization(self, simulator: IOSimulator, sample_matrix_1d: np.ndarray) -> None:
        """Test IOSimulator initialization."""
        assert simulator.total_size == 16
        assert simulator.block_size == 2
        assert simulator.memory_size == 8
        assert simulator.io_count == 0
        assert simulator.memory_limit == 4  # memory_size // block_size
        np.testing.assert_array_equal(simulator.disk, sample_matrix_1d)

    @pytest.mark.parametrize(
        ("block_id", "expected_data"),
        [
            (0, [0, 1]),
            (1, [2, 3]),
            (2, [4, 5]),
            (3, [6, 7]),
            (4, [8, 9]),
            (5, [10, 11]),
            (6, [12, 13]),
            (7, [14, 15]),
        ],
    )
    def test_read_block(self, simulator: IOSimulator, block_id: int, expected_data: list[int]) -> None:
        """Test reading blocks from disk."""
        block = simulator._read_block(block_id)
        expected = np.array(expected_data)
        np.testing.assert_array_equal(block, expected)
        assert simulator.io_count == 1
        assert block_id in simulator.memory

        # Test reading same block again (should be from memory)
        block2 = simulator._read_block(block_id)
        np.testing.assert_array_equal(block2, expected)
        assert simulator.io_count == 1  # No additional I/O

    def test_write_block(self, simulator: IOSimulator) -> None:
        """Test writing blocks to disk."""
        # Read a block and modify it
        block = simulator._read_block(0)
        block[0] = 99
        simulator.memory[0] = block

        # Write the block back
        initial_io_count = simulator.io_count
        simulator._write_block(0)
        assert simulator.io_count == initial_io_count + 1

        # Verify the change was written to disk
        simulator.flush_memory()
        np.testing.assert_array_equal(simulator.disk[0:2], np.array([99, 1]))

    @pytest.mark.parametrize(
        ("row", "col", "expected_value"),
        [
            (0, 0, 0),  # row 0, col 0
            (0, 1, 1),  # row 0, col 1
            (0, 2, 2),  # row 0, col 2
            (0, 3, 3),  # row 0, col 3
            (1, 0, 4),  # row 1, col 0
            (3, 3, 15),  # row 3, col 3
        ],
    )
    def test_get_element(self, simulator: IOSimulator, row: int, col: int, expected_value: int) -> None:
        """Test getting individual elements."""
        assert simulator.get_element(row, col, 4) == expected_value

    def test_set_element(self, simulator: IOSimulator) -> None:
        """Test setting individual elements."""
        # Set an element and verify it's updated
        simulator.set_element(0, 0, 99, 4)
        assert simulator.get_element(0, 0, 4) == 99

        # Verify it's also updated in memory
        assert simulator.memory[0][0] == 99

    def test_get_submatrix(self, simulator: IOSimulator) -> None:
        """Test getting submatrices."""
        submatrix = simulator.get_submatrix(0, 2, 0, 2, 4)
        expected = np.array([[0, 1], [4, 5]])
        np.testing.assert_array_equal(submatrix, expected)

    def test_set_submatrix(self, simulator: IOSimulator) -> None:
        """Test setting submatrices."""
        new_submatrix = np.array([[99, 98], [97, 96]])
        simulator.set_submatrix(0, 0, new_submatrix, 4)

        # Verify the submatrix was set correctly
        assert simulator.get_element(0, 0, 4) == 99
        assert simulator.get_element(0, 1, 4) == 98
        assert simulator.get_element(1, 0, 4) == 97
        assert simulator.get_element(1, 1, 4) == 96

    def test_flush_memory(self, simulator: IOSimulator) -> None:
        """Test flushing memory to disk."""
        # Modify some elements
        simulator.set_element(0, 0, 99, 4)
        simulator.set_element(1, 1, 88, 4)

        # Flush memory
        simulator.flush_memory()

        # Verify memory is cleared
        assert len(simulator.memory) == 0

        # Verify changes were written to disk
        assert simulator.disk[0] == 99  # flat index 0
        assert simulator.disk[5] == 88  # flat index 1*4 + 1 = 5

    @pytest.mark.parametrize(
        ("block_size", "expected_memory_limit"),
        [
            (1, 16),
            (2, 8),
            (4, 4),
            (8, 2),
        ],
    )
    def test_different_block_sizes(
        self, sample_matrix_2d: np.ndarray, block_size: int, expected_memory_limit: int
    ) -> None:
        """Test with different block sizes."""
        sim = IOSimulator(sample_matrix_2d, block_size=block_size, memory_size=16)
        assert sim.block_size == block_size
        assert sim.memory_limit == expected_memory_limit

    def test_rectangular_matrix(self) -> None:
        """Test with rectangular matrices."""
        rect_matrix = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
        sim = IOSimulator(rect_matrix, block_size=2, memory_size=8)

        assert sim.total_size == 6  # 2x3 matrix = 6 elements
        assert sim.memory_limit == 4

        # Test reading elements (2x3 matrix, so n_cols=3)
        assert sim.get_element(0, 0, 3) == 1
        assert sim.get_element(0, 2, 3) == 3
        assert sim.get_element(1, 2, 3) == 6

    def test_single_element_matrix(self) -> None:
        """Test with 1x1 matrix."""
        single_matrix = np.array([[42]])
        sim = IOSimulator(single_matrix, block_size=1, memory_size=2)

        assert sim.total_size == 1
        assert sim.get_element(0, 0, 1) == 42

    def test_zero_memory_size(self) -> None:
        """Test with zero memory size."""
        matrix = np.array([[1, 2], [3, 4]])
        sim = IOSimulator(matrix, block_size=1, memory_size=0)

        # Should still work but with minimal memory
        assert sim.memory_limit == 1

        # Test that we can still read elements
        assert sim.get_element(0, 0, 2) == 1

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        # Test with very small matrix
        small_matrix = np.array([[1]])
        sim = IOSimulator(small_matrix, block_size=1, memory_size=2)
        assert sim.get_element(0, 0, 1) == 1

        # Test with large block size
        large_block_sim = IOSimulator(np.arange(16).reshape(4, 4), block_size=20, memory_size=100)
        assert large_block_sim.block_size == 20
        assert large_block_sim.memory_limit == 5

    def test_memory_limit(self) -> None:
        """Test memory limit functionality."""
        # Create a simulator with very small memory
        small_sim = IOSimulator(np.arange(16).reshape(4, 4), block_size=2, memory_size=4)

        # Fill memory beyond limit
        small_sim._read_block(0)  # Block 0
        small_sim._read_block(1)  # Block 1
        small_sim._read_block(2)  # Block 2 - should evict block 0

        # Verify first block was evicted
        assert 0 not in small_sim.memory
        assert 1 in small_sim.memory
        assert 2 in small_sim.memory

    def test_memory_eviction_order(self) -> None:
        """Test that memory eviction follows LRU order."""
        small_sim = IOSimulator(np.arange(16).reshape(4, 4), block_size=2, memory_size=4)

        # Read blocks in order
        small_sim._read_block(0)  # First block
        small_sim._read_block(1)  # Second block

        # Read third block - should evict first block (LRU)
        small_sim._read_block(2)  # Third block

        # First block should be evicted (least recently used)
        assert 0 not in small_sim.memory
        assert 1 in small_sim.memory
        assert 2 in small_sim.memory

    def test_io_count_tracking(self, simulator: IOSimulator) -> None:
        """Test that I/O count is tracked correctly."""
        initial_count = simulator.io_count

        # Perform some operations
        simulator._read_block(0)
        assert simulator.io_count == initial_count + 1

        simulator._read_block(1)
        assert simulator.io_count == initial_count + 2

        simulator._write_block(0)
        assert simulator.io_count == initial_count + 3

    @pytest.mark.parametrize(
        ("submatrix_shape", "description"),
        [
            ((1, 1), "1x1_submatrix"),
            ((2, 1), "2x1_submatrix"),
            ((1, 2), "1x2_submatrix"),
            ((2, 2), "2x2_submatrix"),
        ],
    )
    def test_set_submatrix_with_different_shapes(
        self, simulator: IOSimulator, submatrix_shape: tuple[int, int], description: str
    ) -> None:
        """Test set_submatrix with different submatrix shapes."""
        if submatrix_shape == (1, 1):
            # Test 1x1 submatrix
            simulator.set_submatrix(0, 0, np.array([[99]]), 4)
            assert simulator.get_element(0, 0, 4) == 99

        elif submatrix_shape == (2, 1):
            # Test 2x1 submatrix
            simulator.set_submatrix(0, 1, np.array([[98], [97]]), 4)
            assert simulator.get_element(0, 1, 4) == 98
            assert simulator.get_element(1, 1, 4) == 97

        elif submatrix_shape == (1, 2):
            # Test 1x2 submatrix
            simulator.set_submatrix(2, 0, np.array([[96, 95]]), 4)
            assert simulator.get_element(2, 0, 4) == 96
            assert simulator.get_element(2, 1, 4) == 95

        elif submatrix_shape == (2, 2):
            # Test 2x2 submatrix
            simulator.set_submatrix(0, 0, np.array([[99, 98], [97, 96]]), 4)
            assert simulator.get_element(0, 0, 4) == 99
            assert simulator.get_element(0, 1, 4) == 98
            assert simulator.get_element(1, 0, 4) == 97
            assert simulator.get_element(1, 1, 4) == 96


class TestIOSimulatorCacheEviction:
    """Tests for cache eviction fixes and data persistence."""

    def test_cache_eviction_preserves_dirty_blocks(self) -> None:
        """Test that modified blocks are written to disk when evicted from cache."""
        # Create simulator with very limited cache (only 2 blocks)
        simulator = IOSimulator(np.array([1, 2, 3, 4, 5, 6, 7, 8]), block_size=1, memory_size=2)

        # Modify element in block 0
        simulator.set_element(0, 0, 99, 8)  # Changes block 0
        assert 0 in simulator.dirty_blocks, "Block 0 should be marked as dirty"

        # Modify element in block 1
        simulator.set_element(0, 1, 88, 8)  # Changes block 1
        assert 1 in simulator.dirty_blocks, "Block 1 should be marked as dirty"

        # Force cache eviction by accessing block 2
        # This should automatically write dirty block 0 to disk before evicting it
        simulator.get_element(0, 2, 8)  # Forces eviction of block 0

        # Verify that block 0 was written to disk even though it was evicted
        assert simulator.disk[0] == 99, "Block 0 changes should be preserved on disk after eviction"

        # Block 0 should no longer be in cache but shouldn't be dirty anymore
        assert 0 not in simulator.memory, "Block 0 should be evicted from cache"
        assert 0 not in simulator.dirty_blocks, "Block 0 should no longer be marked as dirty"

    def test_multiple_cache_evictions_preserve_all_changes(self) -> None:
        """Test that multiple cache evictions preserve all dirty data."""
        # Create simulator with cache for only 2 blocks
        simulator = IOSimulator(np.arange(10), block_size=1, memory_size=2)

        # Modify multiple blocks in sequence, forcing evictions
        test_values = [99, 88, 77, 66, 55]

        for i, value in enumerate(test_values):
            simulator.set_element(0, i, value, 10)
            assert simulator.disk[i] != value, f"Value {value} should not be written to disk immediately"

        # Final flush to ensure all remaining dirty blocks are written
        simulator.flush_memory()

        # Verify all changes were preserved
        for i, expected_value in enumerate(test_values):
            assert simulator.disk[i] == expected_value, f"Changes at position {i} should be preserved"

    def test_dirty_blocks_tracking(self) -> None:
        """Test that dirty blocks are correctly tracked."""
        simulator = IOSimulator(np.array([1, 2, 3, 4]), block_size=1, memory_size=4)

        # Initially no dirty blocks
        assert len(simulator.dirty_blocks) == 0, "Initially should have no dirty blocks"

        # Modify element - should mark block as dirty
        simulator.set_element(0, 0, 99, 4)
        assert 0 in simulator.dirty_blocks, "Block 0 should be marked dirty after modification"

        # Read operation should not create dirty blocks
        simulator.get_element(0, 1, 4)
        assert 1 not in simulator.dirty_blocks, "Read operation should not mark block as dirty"

        # Flush should clear dirty blocks
        simulator.flush_memory()
        assert len(simulator.dirty_blocks) == 0, "Flush should clear all dirty block markers"

    def test_no_data_loss_with_limited_cache(self) -> None:
        """Test sorting-like operations don't lose data with limited cache."""
        # Simulate the problematic scenario from sorting algorithm
        original_data = np.array([5, 1, 3, 2, 8, 4, 7, 6])
        simulator = IOSimulator(original_data.copy(), block_size=2, memory_size=4)  # Cache for 2 blocks

        # Read all data (like sorting algorithm does)
        read_data = []
        for i in range(len(original_data)):
            read_data.append(simulator.get_element(0, i, len(original_data)))

        # Sort the data
        sorted_data = sorted(read_data)

        # Write back sorted data (this should trigger cache evictions)
        for i, value in enumerate(sorted_data):
            simulator.set_element(0, i, float(value), len(original_data))

        # Flush to ensure all changes are written
        simulator.flush_memory()

        # Verify final result matches expected sorted array
        expected_sorted = np.sort(original_data)
        np.testing.assert_array_equal(simulator.disk, expected_sorted)

    def test_io_count_excludes_automatic_eviction_writes(self) -> None:
        """Test that automatic writes during eviction don't count toward I/O statistics."""
        simulator = IOSimulator(np.array([1, 2, 3, 4, 5, 6]), block_size=1, memory_size=2)  # Cache for 2 blocks

        initial_io_count = simulator.io_count

        # Read two blocks into cache
        simulator.get_element(0, 0, 6)  # Read block 0
        simulator.get_element(0, 1, 6)  # Read block 1

        # Should have 2 read operations
        assert simulator.io_count == initial_io_count + 2

        # Modify block 0 (make it dirty)
        simulator.set_element(0, 0, 99, 6)

        # No additional I/O yet (just in-memory modification)
        assert simulator.io_count == initial_io_count + 2

        # Force eviction by reading block 2
        # This should automatically write dirty block 0 to disk, but NOT count in I/O statistics
        simulator.get_element(0, 2, 6)  # Read block 2, evicts block 0

        # Should only count the explicit read of block 2, not the automatic write of block 0
        assert simulator.io_count == initial_io_count + 3  # 2 initial reads + 1 explicit read

        # Explicit flush should count the writes
        simulator.flush_memory()
        # flush_memory writes remaining dirty blocks (block 1 if modified, block 2 if modified)
        # In this case, only explicit operations should be counted

    def test_cache_eviction_lru_with_dirty_blocks(self) -> None:
        """Test LRU eviction policy correctly handles dirty blocks."""
        simulator = IOSimulator(np.arange(8), block_size=1, memory_size=3)  # Cache for 3 blocks

        # Load 3 blocks into cache
        simulator.get_element(0, 0, 8)  # Block 0 (oldest)
        simulator.get_element(0, 1, 8)  # Block 1
        simulator.get_element(0, 2, 8)  # Block 2 (most recent)

        # Modify block 1 (middle in LRU order, will become most recent after modification)
        simulator.set_element(0, 1, 99, 8)

        # Cache should be full: [0, 2, 1*] (* indicates dirty, order is LRU)
        assert len(simulator.memory) == 3
        assert 1 in simulator.dirty_blocks

        # Access block 3 - should evict block 0 (least recently used)
        simulator.get_element(0, 3, 8)  # Forces eviction of block 0 (LRU)

        # Block 0 should be evicted, block 1 should remain (it was recently modified)
        assert 0 not in simulator.memory
        assert 1 in simulator.memory  # Block 1 should still be in cache (recently used)

        # The modified value should be in memory, not yet on disk until flush
        assert simulator.memory[1][0] == 99  # Changes should be in memory
        assert simulator.disk[1] == 1  # Original value still on disk

        # After flush, changes should be written to disk
        simulator.flush_memory()
        assert simulator.disk[1] == 99  # Changes should now be on disk

    def test_lru_dirty_block_automatic_writeback(self) -> None:
        """Test that dirty blocks are automatically written to disk when evicted in LRU order."""
        simulator = IOSimulator(np.arange(6), block_size=1, memory_size=2)  # Cache for only 2 blocks

        # Fill cache with blocks 0 and 1
        simulator.get_element(0, 0, 6)  # Block 0 (oldest)
        simulator.get_element(0, 1, 6)  # Block 1 (newest)

        # Modify block 0 (this makes block 0 the most recently used, block 1 becomes LRU)
        simulator.set_element(0, 0, 99, 6)  # Modify block 0, now LRU order is [1, 0]

        # Both blocks in cache, block 0 is dirty
        assert 0 in simulator.memory
        assert 1 in simulator.memory
        assert 0 in simulator.dirty_blocks

        # Access block 2 - should evict block 1 (LRU), not block 0
        simulator.get_element(0, 2, 6)  # Forces eviction of block 1

        # Block 1 should be evicted, block 0 should remain (it was recently modified)
        assert 1 not in simulator.memory  # Block 1 evicted
        assert 0 in simulator.memory  # Block 0 still in cache (recently used)
        assert 0 in simulator.dirty_blocks  # Block 0 still dirty

        # Now force eviction of dirty block 0 by accessing block 3
        simulator.get_element(0, 3, 6)  # Forces eviction of block 0 (dirty)

        # Block 0 should be evicted and its dirty changes automatically written to disk
        assert 0 not in simulator.memory
        assert 0 not in simulator.dirty_blocks  # No longer dirty since written to disk
        assert simulator.disk[0] == 99  # Changes should be automatically written to disk

    def test_large_dataset_cache_management(self) -> None:
        """Test cache management with larger dataset that exceeds cache multiple times."""
        # Create larger dataset (20 elements, cache for 4 blocks)
        large_data = np.arange(20)
        simulator = IOSimulator(large_data, block_size=2, memory_size=8)  # 4 blocks max

        # Modify every element (will cause multiple evictions)
        for i in range(20):
            new_value = 100 + i
            simulator.set_element(0, i, new_value, 20)

        # Flush remaining changes
        simulator.flush_memory()

        # Verify all changes were preserved despite cache evictions
        for i in range(20):
            expected_value = 100 + i
            assert simulator.disk[i] == expected_value, f"Element {i} should have value {expected_value}"

    def test_mixed_read_write_operations_preserve_data(self) -> None:
        """Test mixed read/write operations with cache evictions preserve all data."""
        simulator = IOSimulator(np.array([10, 20, 30, 40, 50, 60]), block_size=1, memory_size=2)

        # Mixed operations that will cause evictions
        simulator.set_element(0, 0, 11, 6)  # Modify position 0
        _ = simulator.get_element(0, 1, 6)  # Read position 1
        simulator.set_element(0, 2, 33, 6)  # Modify position 2 (may evict block 0)
        _ = simulator.get_element(0, 3, 6)  # Read position 3
        simulator.set_element(0, 4, 55, 6)  # Modify position 4 (may evict other blocks)

        # Flush all changes
        simulator.flush_memory()

        # Verify specific changes were preserved
        assert simulator.disk[0] == 11, "First modification should be preserved"
        assert simulator.disk[2] == 33, "Second modification should be preserved"
        assert simulator.disk[4] == 55, "Third modification should be preserved"

        # Verify unmodified elements remain unchanged
        assert simulator.disk[1] == 20, "Unmodified element should remain unchanged"
        assert simulator.disk[3] == 40, "Unmodified element should remain unchanged"
        assert simulator.disk[5] == 60, "Unmodified element should remain unchanged"

    def test_evict_lru_block(self) -> None:
        """Test _evict_lru_block method."""
        simulator = IOSimulator(np.arange(10), block_size=2, memory_size=4)  # 2 blocks max

        # Load blocks 0 and 1 into cache
        simulator._read_block(0)
        simulator._read_block(1)
        assert len(simulator.memory) == 2

        # Mark block 0 as dirty
        simulator.dirty_blocks.add(0)

        # Evict LRU block (should be block 0)
        simulator._evict_lru_block()

        # Block 0 should be evicted and no longer dirty
        assert 0 not in simulator.memory
        assert 0 not in simulator.dirty_blocks
        assert len(simulator.memory) == 1
        assert 1 in simulator.memory  # Block 1 should remain

    def test_load_block_from_disk(self) -> None:
        """Test _load_block_from_disk method."""
        data = np.array([10, 20, 30, 40, 50, 60])
        simulator = IOSimulator(data, block_size=2, memory_size=4)

        # Load block 0 (elements 0,1)
        block = simulator._load_block_from_disk(0)
        np.testing.assert_array_equal(block, [10, 20])

        # Load block 1 (elements 2,3)
        block = simulator._load_block_from_disk(1)
        np.testing.assert_array_equal(block, [30, 40])

        # Load block 2 (elements 4,5)
        block = simulator._load_block_from_disk(2)
        np.testing.assert_array_equal(block, [50, 60])

        # Load block beyond data (should return empty array)
        block = simulator._load_block_from_disk(10)
        assert len(block) == 0

    def test_write_block_to_disk(self) -> None:
        """Test _write_block_to_disk method."""
        data = np.array([10, 20, 30, 40, 50, 60])
        simulator = IOSimulator(data, block_size=2, memory_size=4)

        # Load block 0 and modify it
        block = simulator._load_block_from_disk(0)
        block[0] = 99
        block[1] = 88

        # Write modified block back to disk
        simulator._write_block_to_disk(0, block)

        # Verify changes were written to disk
        assert simulator.disk[0] == 99
        assert simulator.disk[1] == 88
        assert simulator.disk[2] == 30  # Other elements unchanged

    def test_write_block_to_disk_size_mismatch(self) -> None:
        """Test _write_block_to_disk with size mismatch."""
        data = np.array([10, 20, 30, 40, 50])
        simulator = IOSimulator(data, block_size=3, memory_size=6)

        # Load block 1 (elements 3,4) - only 2 elements
        block = simulator._load_block_from_disk(1)
        block[0] = 99
        block[1] = 88

        # Write back (should handle size mismatch)
        simulator._write_block_to_disk(1, block)

        # Verify changes were written correctly
        assert simulator.disk[3] == 99
        assert simulator.disk[4] == 88

    def test_get_block_id(self) -> None:
        """Test _get_block_id method."""
        simulator = IOSimulator(np.arange(20), block_size=3, memory_size=9)

        # Test various indices
        assert simulator._get_block_id(0) == 0  # First element in block 0
        assert simulator._get_block_id(2) == 0  # Last element in block 0
        assert simulator._get_block_id(3) == 1  # First element in block 1
        assert simulator._get_block_id(5) == 1  # Last element in block 1
        assert simulator._get_block_id(6) == 2  # First element in block 2
        assert simulator._get_block_id(19) == 6  # Last element in block 6

    def test_get_index_in_block(self) -> None:
        """Test _get_index_in_block method."""
        simulator = IOSimulator(np.arange(20), block_size=3, memory_size=9)

        # Test various indices
        assert simulator._get_index_in_block(0) == 0  # First element in block
        assert simulator._get_index_in_block(2) == 2  # Last element in block
        assert simulator._get_index_in_block(3) == 0  # First element in next block
        assert simulator._get_index_in_block(5) == 2  # Last element in next block
        assert simulator._get_index_in_block(6) == 0  # First element in next block
        assert simulator._get_index_in_block(19) == 1  # Second element in last block

    def test_is_valid_index(self) -> None:
        """Test _is_valid_index method."""
        simulator = IOSimulator(np.arange(10), block_size=3, memory_size=9)

        # Test valid indices
        assert simulator._is_valid_index(0)  # First element
        assert simulator._is_valid_index(5)  # Middle element
        assert simulator._is_valid_index(9)  # Last element

        # Test invalid indices
        assert not simulator._is_valid_index(-1)  # Negative index
        assert not simulator._is_valid_index(10)  # Beyond array
        assert not simulator._is_valid_index(100)  # Way beyond array

    def test_get_flat_index(self) -> None:
        """Test _get_flat_index method."""
        simulator = IOSimulator(np.arange(20), block_size=3, memory_size=9)

        # Test various matrix positions
        assert simulator._get_flat_index(0, 0, 4) == 0  # row 0, col 0 in 4-col matrix
        assert simulator._get_flat_index(0, 1, 4) == 1  # row 0, col 1 in 4-col matrix
        assert simulator._get_flat_index(1, 0, 4) == 4  # row 1, col 0 in 4-col matrix
        assert simulator._get_flat_index(1, 3, 4) == 7  # row 1, col 3 in 4-col matrix
        assert simulator._get_flat_index(2, 2, 4) == 10  # row 2, col 2 in 4-col matrix

        # Test with different column counts
        assert simulator._get_flat_index(0, 0, 5) == 0  # row 0, col 0 in 5-col matrix
        assert simulator._get_flat_index(1, 2, 5) == 7  # row 1, col 2 in 5-col matrix
