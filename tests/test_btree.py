import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.searching.btree import BTree
from io_simulator.io_simulator import IOSimulator


class TestBTree:
    """Test cases for B-tree implementation."""

    @pytest.fixture
    def disk_simulator(self) -> IOSimulator:
        """Create a disk simulator for B-tree storage."""
        # Large enough for B-tree nodes (each node uses ~100 positions)
        disk_data = np.zeros(10000)
        return IOSimulator(disk_data, block_size=50, memory_size=200)

    @pytest.fixture
    def btree(self, disk_simulator: IOSimulator) -> BTree:
        """Create a B-tree instance for testing."""
        return BTree(disk_simulator, d_min=3)  # Max 5 keys per node

    def test_empty_tree_operations(self, btree: BTree):
        """Test operations on empty tree."""
        # Search in empty tree
        assert not btree.search(10)

        # Min/max in empty tree
        btree.insert(1)  # Need at least one element
        btree.delete(1)
        assert btree.find_min() is None
        assert btree.find_max() is None

    def test_single_element(self, btree: BTree):
        """Test tree with single element."""
        btree.insert(42)

        assert btree.search(42)
        assert not btree.search(10)
        assert btree.find_min() == 42
        assert btree.find_max() == 42

        btree.delete(42)
        assert not btree.search(42)

    def test_basic_insertion_and_search(self, btree: BTree):
        """Test basic insertion and search operations."""
        keys = [10, 20, 5, 6, 12, 30, 7, 17]

        # Insert keys
        for key in keys:
            btree.insert(key)

        # Search for all inserted keys
        for key in keys:
            assert btree.search(key), f"Key {key} should be found"

        # Search for non-existent keys
        non_existent = [1, 15, 25, 100]
        for key in non_existent:
            assert not btree.search(key), f"Key {key} should not be found"

    def test_min_max_operations(self, btree: BTree):
        """Test minimum and maximum finding."""
        keys = [15, 10, 25, 5, 20, 30, 8]

        for key in keys:
            btree.insert(key)

        assert btree.find_min() == 5
        assert btree.find_max() == 30

    def test_deletion_leaf_nodes(self, btree: BTree):
        """Test deletion from leaf nodes."""
        keys = [10, 20, 5, 6, 12, 30, 7, 17]

        for key in keys:
            btree.insert(key)

        # Delete some keys
        btree.delete(5)
        assert not btree.search(5)
        assert btree.search(6)  # Other keys should remain

        btree.delete(30)
        assert not btree.search(30)
        assert btree.search(20)

    def test_large_dataset(self, btree: BTree):
        """Test B-tree with larger dataset."""
        # Insert 50 keys
        keys = list(range(1, 51))
        np.random.shuffle(keys)  # Random insertion order

        initial_io = btree.get_io_count()

        for key in keys:
            btree.insert(key)

        insertion_io = btree.get_io_count() - initial_io

        # Verify all keys are present
        for key in keys:
            assert btree.search(key), f"Key {key} should be found after insertion"

        # Test min and max
        assert btree.find_min() == 1
        assert btree.find_max() == 50

        # I/O should be logarithmic in the number of keys
        # For B-tree with d_min=3 and 50 keys, expect reasonable I/O count
        print(f"Large dataset insertion I/O: {insertion_io}")
        assert insertion_io < 100, "I/O count should be reasonable for 50 insertions"

    def test_io_complexity(self, btree: BTree):
        """Test that I/O operations scale logarithmically."""
        # Test different sizes
        sizes = [10, 20, 50]
        io_counts = []

        for size in sizes:
            # Create fresh B-tree
            disk_data = np.zeros(10000)
            disk = IOSimulator(disk_data, block_size=50, memory_size=200)
            test_btree = BTree(disk, d_min=3)

            keys = list(range(1, size + 1))
            np.random.shuffle(keys)

            initial_io = test_btree.get_io_count()

            for key in keys:
                test_btree.insert(key)

            # Perform some searches
            for i in range(min(10, size)):
                test_btree.search(keys[i])

            total_io = test_btree.get_io_count() - initial_io
            io_counts.append(total_io)

        print(f"I/O counts for sizes {sizes}: {io_counts}")

        # I/O should not grow too quickly (logarithmic growth)
        # Allow flexibility due to small dataset sizes and B-tree splitting overhead
        for i in range(1, len(io_counts)):
            growth_ratio = io_counts[i] / max(1, io_counts[i - 1])
            assert growth_ratio < 10, f"I/O growth ratio {growth_ratio} seems too high"

    def test_duplicate_insertion(self, btree: BTree):
        """Test inserting duplicate keys."""
        btree.insert(10)
        btree.insert(10)  # Duplicate

        assert btree.search(10)
        # Tree should handle duplicates gracefully (implementation dependent)

    def test_btree_properties(self, btree: BTree):
        """Test that B-tree maintains its structural properties."""
        keys = list(range(1, 21))  # 20 keys

        for key in keys:
            btree.insert(key)

        # Verify tree structure by checking that all operations work correctly
        # This implicitly tests that B-tree properties are maintained

        # All keys should be findable
        for key in keys:
            assert btree.search(key)

        # Min and max should be correct
        assert btree.find_min() == 1
        assert btree.find_max() == 20

        # Random access should work
        test_keys = [5, 10, 15, 18, 2]
        for key in test_keys:
            assert btree.search(key)

    def test_empty_tree_min_max(self, btree: BTree):
        """Test min/max on tree with only root."""
        # Insert and delete to leave empty root
        btree.insert(10)
        btree.delete(10)

        # Min/max should handle empty tree gracefully
        min_key = btree.find_min()
        max_key = btree.find_max()

        # Either None or some default value is acceptable
        assert min_key is None or isinstance(min_key, (int, float))
        assert max_key is None or isinstance(max_key, (int, float))

    @pytest.mark.parametrize("d_min", [2, 3, 4, 5])
    def test_different_branching_factors(self, disk_simulator: IOSimulator, d_min: int):
        """Test B-tree with different branching factors."""
        btree = BTree(disk_simulator, d_min=d_min)

        keys = [10, 5, 15, 3, 7, 12, 18, 1, 4, 6, 8, 11, 13, 16, 20]

        for key in keys:
            btree.insert(key)

        # All keys should be searchable regardless of branching factor
        for key in keys:
            assert btree.search(key)

        assert btree.find_min() == 1
        assert btree.find_max() == 20

    def test_stress_random_operations(self, btree: BTree):
        """Stress test with random insertions and searches."""
        np.random.seed(42)  # Reproducible test

        keys = list(range(1, 101))  # 100 keys
        np.random.shuffle(keys)

        # Insert first 50 keys
        for key in keys[:50]:
            btree.insert(key)

        # Verify they're all there
        for key in keys[:50]:
            assert btree.search(key)

        # Search for some keys not inserted yet
        for key in keys[50:60]:
            assert not btree.search(key)

        # Insert remaining keys
        for key in keys[50:]:
            btree.insert(key)

        # Final verification
        for key in keys:
            assert btree.search(key)

        assert btree.find_min() == 1
        assert btree.find_max() == 100
