import unittest

import numpy as np

from io_simulator.io_simulator import IOSimulator


class TestIOSimulator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        self.sim = IOSimulator(self.matrix, block_size=2, memory_size=8)

    def test_initialization(self):
        """Test IOSimulator initialization."""
        self.assertEqual(self.sim.n, 4)
        self.assertEqual(self.sim.block_size, 2)
        self.assertEqual(self.sim.memory_size, 8)
        self.assertEqual(self.sim.io_count, 0)
        self.assertEqual(self.sim.memory_limit, 4)  # memory_size // block_size
        np.testing.assert_array_equal(self.sim.disk, self.matrix)

    def test_read_block(self):
        """Test reading blocks from disk."""
        # Test reading first block of first row
        block = self.sim.read_block(0, 0)
        expected = np.array([1, 2])
        np.testing.assert_array_equal(block, expected)
        self.assertEqual(self.sim.io_count, 1)

        # Test reading same block again (should be from memory)
        initial_io_count = self.sim.io_count
        block = self.sim.read_block(0, 0)
        np.testing.assert_array_equal(block, expected)
        self.assertEqual(self.sim.io_count, initial_io_count)  # No new I/O

    def test_write_block(self):
        """Test writing blocks to disk."""
        # First read a block to get it in memory
        block = self.sim.read_block(0, 0)
        block[0] = 99  # Modify the block
        self.sim.memory[(0, 0)] = block

        initial_io_count = self.sim.io_count
        self.sim.write_block(0, 0)
        self.assertEqual(self.sim.io_count, initial_io_count + 1)

        # Verify the change was written to disk
        np.testing.assert_array_equal(self.sim.disk[0, 0:2], np.array([99, 2]))

    def test_get_element(self):
        """Test getting individual elements."""
        # Test elements in different blocks
        self.assertEqual(self.sim.get_element(0, 0), 1)
        self.assertEqual(self.sim.get_element(0, 1), 2)
        self.assertEqual(self.sim.get_element(0, 2), 3)
        self.assertEqual(self.sim.get_element(0, 3), 4)

        # Test elements in different rows
        self.assertEqual(self.sim.get_element(1, 0), 5)
        self.assertEqual(self.sim.get_element(3, 3), 16)

    def test_set_element(self):
        """Test setting individual elements."""
        # Set an element and verify it's updated
        self.sim.set_element(0, 0, 99)
        self.assertEqual(self.sim.get_element(0, 0), 99)

        # Verify it's also updated in memory
        self.assertEqual(self.sim.memory[(0, 0)][0], 99)

    def test_get_submatrix(self):
        """Test getting submatrices."""
        submatrix = self.sim.get_submatrix(0, 2, 0, 2)
        expected = np.array([[1, 2], [5, 6]])
        np.testing.assert_array_equal(submatrix, expected)

    def test_set_submatrix(self):
        """Test setting submatrices."""
        new_submatrix = np.array([[99, 98], [97, 96]])
        self.sim.set_submatrix(0, 0, new_submatrix)

        # Verify the submatrix was set correctly
        result = self.sim.get_submatrix(0, 2, 0, 2)
        np.testing.assert_array_equal(result, new_submatrix)

    def test_flush_memory(self):
        """Test flushing memory to disk."""
        # Modify some elements
        self.sim.set_element(0, 0, 99)
        self.sim.set_element(1, 1, 88)

        # Flush memory
        self.sim.flush_memory()

        # Verify memory is cleared
        self.assertEqual(len(self.sim.memory), 0)

        # Verify changes were written to disk
        self.assertEqual(self.sim.disk[0, 0], 99)
        self.assertEqual(self.sim.disk[1, 1], 88)

    def test_memory_limit(self):
        """Test memory limit functionality."""
        # Create a simulator with very small memory
        small_sim = IOSimulator(self.matrix, block_size=2, memory_size=4)

        # Fill memory beyond limit
        small_sim.read_block(0, 0)  # Block 1
        small_sim.read_block(0, 1)  # Block 2
        small_sim.read_block(1, 0)  # Block 3 - should evict block 1

        # Verify first block was evicted
        self.assertNotIn((0, 0), small_sim.memory)
        self.assertIn((0, 1), small_sim.memory)
        self.assertIn((1, 0), small_sim.memory)

    def test_different_block_sizes(self):
        """Test with different block sizes."""
        # Test with block_size=1
        sim1 = IOSimulator(self.matrix, block_size=1, memory_size=8)
        self.assertEqual(sim1.memory_limit, 8)

        # Test with block_size=4
        sim4 = IOSimulator(self.matrix, block_size=4, memory_size=8)
        self.assertEqual(sim4.memory_limit, 2)

        # Test with block_size larger than matrix
        sim_large = IOSimulator(self.matrix, block_size=10, memory_size=20)
        self.assertEqual(sim_large.memory_limit, 2)

    def test_rectangular_matrix(self):
        """Test with rectangular matrices."""
        rect_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        sim = IOSimulator(rect_matrix, block_size=2, memory_size=8)

        self.assertEqual(sim.n, 2)
        self.assertEqual(sim.disk.shape[1], 3)
        self.assertEqual(sim.memory_limit, 4)

        # Test reading elements
        self.assertEqual(sim.get_element(0, 0), 1)
        self.assertEqual(sim.get_element(0, 2), 3)
        self.assertEqual(sim.get_element(1, 2), 6)

    def test_single_element_matrix(self):
        """Test with 1x1 matrix."""
        single_matrix = np.array([[42]])
        sim = IOSimulator(single_matrix, block_size=1, memory_size=2)

        self.assertEqual(sim.n, 1)
        self.assertEqual(sim.memory_limit, 2)

        # Test reading and writing
        self.assertEqual(sim.get_element(0, 0), 42)
        sim.set_element(0, 0, 99)
        self.assertEqual(sim.get_element(0, 0), 99)

    def test_zero_memory_size(self):
        """Test with zero memory size."""
        sim = IOSimulator(self.matrix, block_size=2, memory_size=0)

        # Should still work but with minimal memory
        self.assertEqual(sim.memory_limit, 1)

        # Test that it can still read elements
        self.assertEqual(sim.get_element(0, 0), 1)

    def test_io_count_tracking(self):
        """Test that I/O count is tracked correctly."""
        initial_count = self.sim.io_count

        # Read a new block
        self.sim.read_block(0, 0)
        self.assertEqual(self.sim.io_count, initial_count + 1)

        # Read same block again (should not increment)
        self.sim.read_block(0, 0)
        self.assertEqual(self.sim.io_count, initial_count + 1)

        # Read a different block
        self.sim.read_block(0, 1)
        self.assertEqual(self.sim.io_count, initial_count + 2)

        # Write a block
        self.sim.write_block(0, 0)
        self.assertEqual(self.sim.io_count, initial_count + 3)

    def test_memory_eviction_order(self):
        """Test that memory eviction follows FIFO order."""
        small_sim = IOSimulator(self.matrix, block_size=2, memory_size=4)

        # Fill memory
        small_sim.read_block(0, 0)  # First block
        small_sim.read_block(0, 1)  # Second block

        # Read third block - should evict first block
        small_sim.read_block(1, 0)  # Third block

        # First block should be evicted
        self.assertNotIn((0, 0), small_sim.memory)
        self.assertIn((0, 1), small_sim.memory)
        self.assertIn((1, 0), small_sim.memory)

    def test_set_submatrix_with_different_shapes(self):
        """Test set_submatrix with different submatrix shapes."""
        # Test 1x1 submatrix
        self.sim.set_submatrix(0, 0, np.array([[99]]))
        self.assertEqual(self.sim.get_element(0, 0), 99)

        # Test 2x1 submatrix
        self.sim.set_submatrix(0, 1, np.array([[98], [97]]))
        self.assertEqual(self.sim.get_element(0, 1), 98)
        self.assertEqual(self.sim.get_element(1, 1), 97)

        # Test 1x2 submatrix
        self.sim.set_submatrix(2, 0, np.array([[96, 95]]))
        self.assertEqual(self.sim.get_element(2, 0), 96)
        self.assertEqual(self.sim.get_element(2, 1), 95)

    def test_edge_cases(self):
        """Test edge cases."""
        # Test with very small matrix
        tiny_matrix = np.array([[1, 2]])
        sim = IOSimulator(tiny_matrix, block_size=1, memory_size=2)

        self.assertEqual(sim.get_element(0, 0), 1)
        self.assertEqual(sim.get_element(0, 1), 2)

        # Test setting elements
        sim.set_element(0, 0, 99)
        self.assertEqual(sim.get_element(0, 0), 99)

        # Test flush
        sim.flush_memory()
        self.assertEqual(sim.disk[0, 0], 99)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
