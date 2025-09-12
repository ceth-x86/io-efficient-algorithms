from collections import OrderedDict

import numpy as np


class IOSimulator:
    """
    Simulates disk I/O operations with memory caching.

    This class simulates reading and writing data to/from disk in blocks,
    with a limited memory cache. It's designed to work with flat 1D arrays
    while providing matrix-like access patterns through get_element/set_element.

    Attributes:
        disk (np.ndarray): The flat 1D array representing disk storage
        total_size (int): Total number of elements in the array
        block_size (int): Size of each I/O block
        memory_size (int): Maximum memory available for caching
        io_count (int): Number of I/O operations performed
        memory (dict): Cache storing blocks in memory
        memory_limit (int): Maximum number of blocks that can be cached

    Example:
        >>> import numpy as np
        >>> matrix = np.array([[1, 2], [3, 4]])
        >>> sim = IOSimulator(matrix, block_size=2, memory_size=8)
        >>> value = sim.get_element(0, 1, 2)  # Get element at row 0, col 1
        >>> sim.set_element(0, 1, 99, 2)     # Set element at row 0, col 1
    """

    def __init__(self, data: np.ndarray, block_size: int, memory_size: int) -> None:
        """
        Initialize the I/O simulator.

        Args:
            data (np.ndarray): Input data (will be flattened to 1D)
            block_size (int): Size of each I/O block
            memory_size (int): Maximum memory available for caching
        """
        # Flatten the data to 1D array
        self.disk = data.flatten() if data.ndim > 1 else data.copy()
        self.total_size = len(self.disk)
        self.block_size = block_size
        self.memory_size = memory_size
        self.io_count = 0
        self.memory = OrderedDict()  # Use OrderedDict for LRU tracking
        self.dirty_blocks = set()  # Track which blocks have been modified
        self.memory_limit = max(1, memory_size // block_size)

    def _read_block(self, block_id: int) -> np.ndarray:
        # LRU: If block is already in cache, move it to end (most recently used)
        if block_id in self.memory:
            block = self.memory[block_id]
            self.memory.move_to_end(block_id)  # Mark as recently used
            return block

        # Cache is full - evict least recently used (first in OrderedDict)
        if len(self.memory) >= self.memory_limit:
            # Get LRU block (first item in OrderedDict)
            lru_block_id, lru_block_data = self.memory.popitem(last=False)  # Remove LRU

            # Write LRU block to disk if it's dirty
            if lru_block_id in self.dirty_blocks:
                # Write the block data directly since it's no longer in memory
                start = lru_block_id * self.block_size
                end = min(start + self.block_size, self.total_size)

                # Handle size mismatch between memory block and disk slice
                disk_slice_size = end - start

                if len(lru_block_data) >= disk_slice_size:
                    # Block in memory is larger or equal - take only what we need
                    self.disk[start:end] = lru_block_data[:disk_slice_size]
                else:
                    # Block in memory is smaller - write only what we have
                    self.disk[start : start + len(lru_block_data)] = lru_block_data

                self.dirty_blocks.remove(lru_block_id)

        # Read new block from disk
        start = block_id * self.block_size
        end = min(start + self.block_size, self.total_size)

        # Handle case where start is beyond the data
        block = np.zeros(0, dtype=self.disk.dtype) if start >= self.total_size else self.disk[start:end].copy()

        # Add to cache (automatically goes to end as most recently used)
        self.memory[block_id] = block
        self.io_count += 1
        return block

    def _write_block_to_disk_only(self, block_id: int) -> None:
        """Write block to disk without updating I/O count (used for eviction)."""
        if block_id not in self.memory:
            return
        start = block_id * self.block_size
        end = min(start + self.block_size, self.total_size)

        # Handle size mismatch between memory block and disk slice
        block_data = self.memory[block_id]
        disk_slice_size = end - start

        if len(block_data) >= disk_slice_size:
            # Block in memory is larger or equal - take only what we need
            self.disk[start:end] = block_data[:disk_slice_size]
        else:
            # Block in memory is smaller - write only what we have
            self.disk[start : start + len(block_data)] = block_data

    def _write_block(self, block_id: int) -> None:
        if block_id not in self.memory:
            return

        self._write_block_to_disk_only(block_id)
        self.io_count += 1

    def get_element(self, i: int, j: int, n_cols: int) -> float:
        """
        Get element at position (i, j) in a matrix with n_cols columns.

        This method simulates reading a single element from disk, handling
        block-based I/O operations and memory management.

        Args:
            i (int): Row index in the matrix
            j (int): Column index in the matrix
            n_cols (int): Number of columns in the matrix

        Returns:
            float: The value at position (i, j)

        Note:
            For out-of-bounds access, returns 0.0
        """
        flat_index = i * n_cols + j
        block_id = flat_index // self.block_size
        block = self._read_block(block_id)

        # Handle case where block is empty (beyond data)
        if len(block) == 0:
            return 0.0  # Return 0 for out-of-bounds access

        # Handle case where the index is beyond the block size
        index_in_block = flat_index % self.block_size
        if index_in_block >= len(block):
            return 0.0  # Return 0 for out-of-bounds access

        return block[index_in_block]

    def set_element(self, i: int, j: int, value: float, n_cols: int) -> None:
        """
        Set element at position (i, j) in a matrix with n_cols columns.

        This method simulates writing a single element to disk, handling
        block-based I/O operations and memory management.

        Args:
            i (int): Row index in the matrix
            j (int): Column index in the matrix
            value (float): Value to set at position (i, j)
            n_cols (int): Number of columns in the matrix

        Note:
            Out-of-bounds writes are silently ignored
        """
        flat_index = i * n_cols + j

        # Check bounds before proceeding
        if flat_index >= self.total_size or flat_index < 0:
            return  # Skip out-of-bounds writes

        block_id = flat_index // self.block_size
        block = self._read_block(block_id)

        # Handle case where the index is beyond the block size
        index_in_block = flat_index % self.block_size
        if index_in_block >= len(block):
            # Need to extend block to accommodate the element
            if flat_index < self.total_size:
                # Create a properly sized block
                new_size = min(self.block_size, self.total_size - block_id * self.block_size)
                new_block = np.zeros(new_size, dtype=self.disk.dtype)
                new_block[: len(block)] = block
                block = new_block
            else:
                return  # Skip setting out-of-bounds elements

        block[index_in_block] = value
        self.memory[block_id] = block
        self.dirty_blocks.add(block_id)  # Mark block as modified

        # LRU: Mark this block as recently used
        self.memory.move_to_end(block_id)

    def get_submatrix(self, i_start: int, i_end: int, j_start: int, j_end: int, n_cols: int) -> np.ndarray:
        """
        Read submatrix into numpy array (with I/O).

        Args:
            i_start (int): Starting row index
            i_end (int): Ending row index (exclusive)
            j_start (int): Starting column index
            j_end (int): Ending column index (exclusive)
            n_cols (int): Number of columns in the matrix

        Returns:
            np.ndarray: The submatrix as a 2D numpy array
        """
        rows = i_end - i_start
        cols = j_end - j_start
        result = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                result[i, j] = self.get_element(i_start + i, j_start + j, n_cols)

        return result

    def set_submatrix(self, i_start: int, j_start: int, submat: np.ndarray, n_cols: int) -> None:
        """
        Write numpy array back (with I/O).

        Args:
            i_start (int): Starting row index
            j_start (int): Starting column index
            submat (np.ndarray): The submatrix to write
            n_cols (int): Number of columns in the matrix
        """
        rows, cols = submat.shape

        for i in range(rows):
            for j in range(cols):
                self.set_element(i_start + i, j_start + j, submat[i, j], n_cols)

    def flush_memory(self) -> None:
        """
        Flush all cached blocks to disk and clear memory.

        This method writes all modified blocks back to disk and clears
        the memory cache. Should be called to ensure data persistence.
        """
        for block_id in list(self.memory.keys()):
            self._write_block(block_id)
        self.memory.clear()
        self.dirty_blocks.clear()
