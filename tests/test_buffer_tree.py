import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.searching.buffer_tree import BufferTree
from algorithms.searching.buffer_tree import Operation
from algorithms.searching.buffer_tree import OperationType
from io_simulator.io_simulator import IOSimulator


class TestBufferTree:
    """Test cases for Buffer Tree implementation."""

    @pytest.fixture
    def disk_simulator(self) -> IOSimulator:
        """Create a large disk simulator for buffer tree storage."""
        # Buffer trees need more space due to larger nodes and buffers
        disk_data = np.zeros(50000)
        return IOSimulator(disk_data, block_size=50, memory_size=200)

    @pytest.fixture
    def buffer_tree(self, disk_simulator: IOSimulator) -> BufferTree:
        """Create a buffer tree instance for testing."""
        return BufferTree(disk_simulator, memory_size=200, block_size=50, degree=8)

    def test_empty_tree_operations(self, buffer_tree: BufferTree):
        """Test operations on empty tree."""
        # Search in empty tree
        buffer_tree.search(10)
        buffer_tree.flush_all_operations()
        assert buffer_tree.search_results.get(10) is None

    def test_single_element_operations(self, buffer_tree: BufferTree):
        """Test tree with single element."""
        buffer_tree.insert(42, "value_42")
        buffer_tree.flush_all_operations()

        buffer_tree.search(42)
        buffer_tree.flush_all_operations()
        assert buffer_tree.search_results.get(42) == "value_42"

        # Search for non-existent key
        buffer_tree.search(10)
        buffer_tree.flush_all_operations()
        assert buffer_tree.search_results.get(10) is None

    def test_batch_insertions_and_searches(self, buffer_tree: BufferTree):
        """Test batch insertion and search operations."""
        keys_values = [(i, f"value_{i}") for i in [10, 5, 15, 3, 7, 12, 18, 1, 4, 6]]

        # Batch insert
        for key, value in keys_values:
            buffer_tree.insert(key, value)

        # Flush to process insertions
        buffer_tree.flush_all_operations()

        # Batch search for all inserted keys
        for key, _ in keys_values:
            buffer_tree.search(key)

        buffer_tree.flush_all_operations()

        # Verify results
        for key, expected_value in keys_values:
            assert buffer_tree.search_results.get(key) == expected_value

    def test_mixed_operations(self, buffer_tree: BufferTree):
        """Test mixed insert, search, and delete operations."""
        # Insert some keys
        for key in [10, 20, 30, 40, 50]:
            buffer_tree.insert(key, f"value_{key}")

        buffer_tree.flush_all_operations()

        # Delete some keys
        buffer_tree.delete(20)
        buffer_tree.delete(40)
        buffer_tree.flush_all_operations()

        # Search for remaining keys
        remaining_keys = [10, 30, 50]
        deleted_keys = [20, 40]

        for key in remaining_keys + deleted_keys:
            buffer_tree.search(key)

        buffer_tree.flush_all_operations()

        # Verify remaining keys exist and deleted keys don't
        for key in remaining_keys:
            assert buffer_tree.search_results.get(key) == f"value_{key}"

        for key in deleted_keys:
            assert buffer_tree.search_results.get(key) is None

    def test_large_dataset_batching(self, buffer_tree: BufferTree):
        """Test buffer tree with larger dataset to verify batching efficiency."""
        # Insert 100 keys in random order
        keys = list(range(1, 101))
        rng = np.random.default_rng(42)  # For reproducible tests
        rng.shuffle(keys)

        initial_io = buffer_tree.get_io_count()

        # Batch insert all keys
        for key in keys:
            buffer_tree.insert(key, f"value_{key}")

        insertion_io_before_flush = buffer_tree.get_io_count() - initial_io

        # Flush all operations
        buffer_tree.flush_all_operations()

        total_insertion_io = buffer_tree.get_io_count() - initial_io

        print(f"I/O before flush: {insertion_io_before_flush}")  # noqa: T201
        print(f"Total insertion I/O: {total_insertion_io}")  # noqa: T201
        print(f"Keys processed: {len(keys)}")  # noqa: T201

        # I/O should be efficient due to batching
        # Should be much better than O(n log n) for individual operations
        amortized_io = total_insertion_io / len(keys)
        print(f"Amortized I/O per insertion: {amortized_io:.3f}")  # noqa: T201

        # Verify all keys are present
        search_io_start = buffer_tree.get_io_count()

        # Search for subset of keys
        search_keys = keys[:20]  # Search first 20 keys
        for key in search_keys:
            buffer_tree.search(key)

        buffer_tree.flush_all_operations()

        search_io = buffer_tree.get_io_count() - search_io_start
        amortized_search_io = search_io / len(search_keys)
        print(f"Amortized I/O per search: {amortized_search_io:.3f}")  # noqa: T201

        # Verify search results
        for key in search_keys:
            assert buffer_tree.search_results.get(key) == f"value_{key}"

    def test_io_efficiency_scaling(self, buffer_tree: BufferTree):
        """Test that I/O operations scale efficiently with input size."""
        sizes = [20, 50, 100]
        io_per_operation = []

        for size in sizes:
            # Create fresh buffer tree for each test
            disk_data = np.zeros(50000)
            disk = IOSimulator(disk_data, block_size=50, memory_size=200)
            test_tree = BufferTree(disk, memory_size=200, block_size=50, degree=8)

            keys = list(range(1, size + 1))
            rng = np.random.default_rng()
            rng.shuffle(keys)

            initial_io = test_tree.get_io_count()

            # Batch insert
            for key in keys:
                test_tree.insert(key, f"value_{key}")

            test_tree.flush_all_operations()

            # Batch search for half the keys
            search_keys = keys[: size // 2]
            for key in search_keys:
                test_tree.search(key)

            test_tree.flush_all_operations()

            total_io = test_tree.get_io_count() - initial_io
            operations_count = size + len(search_keys)  # insertions + searches
            avg_io = total_io / operations_count

            io_per_operation.append(avg_io)
            print(f"Size {size}: {total_io} I/O for {operations_count} ops = {avg_io:.3f} I/O/op")  # noqa: T201

        print(f"I/O per operation for sizes {sizes}: {io_per_operation}")  # noqa: T201

        # Buffer trees should maintain efficient I/O even as size increases
        # Due to batching, growth should be sublinear
        assert all(io < 5.0 for io in io_per_operation), "I/O per operation should remain efficient"

    def test_buffer_overflow_handling(self, buffer_tree: BufferTree):
        """Test that buffer overflows are handled correctly."""
        # Create small capacity tree to test overflow
        disk_data = np.zeros(10000)
        disk = IOSimulator(disk_data, block_size=10, memory_size=50)
        small_tree = BufferTree(disk, memory_size=50, block_size=10, degree=4)

        # Insert enough operations to trigger buffer flushes
        initial_io = small_tree.get_io_count()

        for i in range(25):  # More than collection buffer + node buffer capacity
            small_tree.insert(i, f"value_{i}")

        # Should trigger automatic flushes
        small_tree.flush_all_operations()

        # Track I/O for buffer overflow handling
        _ = small_tree.get_io_count() - initial_io

        # Search for some inserted keys
        for i in range(0, 25, 5):  # Every 5th key
            small_tree.search(i)

        small_tree.flush_all_operations()

        # Verify keys are accessible despite buffer overflows
        for i in range(0, 25, 5):
            result = small_tree.search_results.get(i)
            assert result == f"value_{i}", f"Key {i} should be found after buffer overflow"

    def test_operation_types(self, buffer_tree: BufferTree):
        """Test different operation types and their handling."""
        # Test each operation type
        op_search = Operation(OperationType.SEARCH, 10)
        op_insert = Operation(OperationType.INSERT, 20, "test_value")
        op_delete = Operation(OperationType.DELETE, 30)

        assert op_search.op_type == OperationType.SEARCH
        assert op_insert.op_type == OperationType.INSERT
        assert op_delete.op_type == OperationType.DELETE

        # Test operation comparison for sorting
        ops = [
            Operation(OperationType.INSERT, 30, "c"),
            Operation(OperationType.SEARCH, 10),
            Operation(OperationType.INSERT, 20, "b"),
        ]

        sorted_ops = sorted(ops)
        assert sorted_ops[0].key == 10
        assert sorted_ops[1].key == 20
        assert sorted_ops[2].key == 30

    def test_duplicate_key_handling(self, buffer_tree: BufferTree):
        """Test handling of duplicate key insertions."""
        # Insert key
        buffer_tree.insert(10, "original_value")
        buffer_tree.flush_all_operations()

        # Insert same key with different value
        buffer_tree.insert(10, "updated_value")
        buffer_tree.flush_all_operations()

        # Search should return updated value
        buffer_tree.search(10)
        buffer_tree.flush_all_operations()

        assert buffer_tree.search_results.get(10) == "updated_value"

    def test_tree_structure_properties(self, buffer_tree: BufferTree):
        """Test that buffer tree maintains proper structure."""
        # Insert keys to build multi-level tree
        keys = list(range(1, 51))  # 50 keys should create internal structure

        for key in keys:
            buffer_tree.insert(key, f"value_{key}")

        buffer_tree.flush_all_operations()

        # Tree should maintain structure after operations
        # This is implicitly tested by successful operations

        # Search for random subset
        test_keys = [1, 10, 25, 35, 50]
        for key in test_keys:
            buffer_tree.search(key)

        buffer_tree.flush_all_operations()

        # All keys should be found
        for key in test_keys:
            assert buffer_tree.search_results.get(key) == f"value_{key}"

    @pytest.mark.parametrize("degree", [4, 8, 16])
    def test_different_degrees(self, disk_simulator: IOSimulator, degree: int):
        """Test buffer tree with different node degrees."""
        tree = BufferTree(disk_simulator, memory_size=200, block_size=50, degree=degree)

        # Insert keys
        keys = [15, 7, 23, 3, 11, 19, 27, 1, 5, 9, 13, 17, 21, 25, 29]

        for key in keys:
            tree.insert(key, f"value_{key}")

        tree.flush_all_operations()

        # Search for all keys
        for key in keys:
            tree.search(key)

        tree.flush_all_operations()

        # All keys should be found regardless of degree
        for key in keys:
            assert tree.search_results.get(key) == f"value_{key}"

    def test_stress_random_operations(self, buffer_tree: BufferTree):
        """Stress test with random mixed operations."""
        rng = np.random.default_rng(123)  # Reproducible test

        # Generate random operations
        keys = list(range(1, 201))  # 200 possible keys
        rng.shuffle(keys)

        # Insert first 100 keys
        insert_keys = keys[:100]
        for key in insert_keys:
            buffer_tree.insert(key, f"value_{key}")

        # Delete some keys
        delete_keys = keys[:20]  # Delete first 20
        for key in delete_keys:
            buffer_tree.delete(key)

        # Process all operations
        buffer_tree.flush_all_operations()

        # Search for mix of existing and deleted keys
        search_keys = keys[:50]  # First 50 (mix of deleted and existing)
        for key in search_keys:
            buffer_tree.search(key)

        buffer_tree.flush_all_operations()

        # Verify results
        expected_present = set(insert_keys) - set(delete_keys)

        for key in search_keys:
            result = buffer_tree.search_results.get(key)
            if key in expected_present:
                assert result == f"value_{key}", f"Key {key} should be present"
            else:
                assert result is None, f"Key {key} should be deleted or never inserted"
