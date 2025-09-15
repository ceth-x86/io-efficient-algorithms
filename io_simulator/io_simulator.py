from collections import OrderedDict

import numpy as np


class IOSimulator:
    """
    Simulates disk I/O operations with memory caching for external memory algorithms.

    This class simulates reading and writing data to/from disk in blocks,
    with a limited memory cache using LRU (Least Recently Used) eviction policy.
    It's designed to work with flat 1D arrays while providing matrix-like access
    patterns through get_element/set_element methods.

    The simulator is particularly useful for:
    - Testing external memory algorithms
    - Measuring I/O complexity of algorithms
    - Simulating memory-constrained environments

    Attributes:
        disk (np.ndarray): The flat 1D array representing disk storage
        total_size (int): Total number of elements in the array
        block_size (int): Size of each I/O block (B)
        memory_size (int): Maximum memory available for caching (M)
        io_count (int): Number of I/O operations performed
        memory (OrderedDict): LRU cache storing blocks in memory
        dirty_blocks (set): Set of block IDs that have been modified
        memory_limit (int): Maximum number of blocks that can be cached (M/B)

    Example:
        >>> import numpy as np
        >>> # Create a 4x4 matrix
        >>> matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        >>> sim = IOSimulator(matrix, block_size=2, memory_size=8)
        >>>
        >>> # Access elements (triggers I/O operations)
        >>> value = sim.get_element(0, 1, 4)  # Get element at row 0, col 1
        >>> sim.set_element(0, 1, 99, 4)     # Set element at row 0, col 1
        >>>
        >>> # Check I/O operations performed
        >>> print(f"I/O operations: {sim.io_count}")
        >>>
        >>> # Flush all changes to disk
        >>> sim.flush_memory()
    """

    def __init__(self, data: np.ndarray, block_size: int, memory_size: int) -> None:
        """
        Initialize the I/O simulator.

        Args:
            data (np.ndarray): Input data (will be flattened to 1D)
            block_size (int): Size of each I/O block
            memory_size (int): Maximum memory available for caching
        """
        # Store data as 1D array
        self.disk = data.flatten() if data.ndim > 1 else data.copy()
        self.total_size = len(self.disk)

        # I/O configuration
        self.block_size = block_size
        self.memory_size = memory_size
        self.memory_limit = max(1, memory_size // block_size)

        # Runtime state
        self.io_count = 0
        self.memory = OrderedDict()  # LRU cache: block_id -> block_data
        self.dirty_blocks = set()  # Track modified blocks

    def _get_block_id(self, flat_index: int) -> int:
        """Get block ID for a given flat index."""
        return flat_index // self.block_size

    def _get_index_in_block(self, flat_index: int) -> int:
        """Get index within block for a given flat index."""
        return flat_index % self.block_size

    def _is_valid_index(self, flat_index: int) -> bool:
        """Check if flat index is valid (within bounds)."""
        return 0 <= flat_index < self.total_size

    def _get_flat_index(self, i: int, j: int, n_cols: int) -> int:
        """Get flat index for matrix position (i, j) with n_cols columns."""
        return i * n_cols + j

    def _read_block(self, block_id: int) -> np.ndarray:
        """Read a block from cache or disk."""
        # Check if block is already in cache
        if block_id in self.memory:
            self.memory.move_to_end(block_id)  # Mark as recently used
            return self.memory[block_id]

        # Cache is full - need to evict a block
        if len(self.memory) >= self.memory_limit:
            self._evict_lru_block()

        # Read block from disk
        block = self._load_block_from_disk(block_id)

        # Add to cache
        self.memory[block_id] = block
        self.io_count += 1
        return block

    def _evict_lru_block(self) -> None:
        """Evict the least recently used block from cache."""
        if not self.memory:
            return

        lru_block_id, lru_block_data = self.memory.popitem(last=False)

        # Write to disk if block was modified
        if lru_block_id in self.dirty_blocks:
            self._write_block_to_disk(lru_block_id, lru_block_data)
            self.dirty_blocks.remove(lru_block_id)

    def _load_block_from_disk(self, block_id: int) -> np.ndarray:
        """Load a block from disk into memory."""
        start = block_id * self.block_size
        end = min(start + self.block_size, self.total_size)

        if start >= self.total_size:
            return np.zeros(0, dtype=self.disk.dtype)

        return self.disk[start:end].copy()

    def _write_block_to_disk(self, block_id: int, block_data: np.ndarray) -> None:
        """Write block data to disk."""
        start = block_id * self.block_size
        end = min(start + self.block_size, self.total_size)
        disk_slice_size = end - start

        if len(block_data) >= disk_slice_size:
            self.disk[start:end] = block_data[:disk_slice_size]
        else:
            self.disk[start : start + len(block_data)] = block_data

    def _write_block(self, block_id: int) -> None:
        """Write block to disk and update I/O count."""
        if block_id not in self.memory:
            return

        block_data = self.memory[block_id]
        self._write_block_to_disk(block_id, block_data)
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
        flat_index = self._get_flat_index(i, j, n_cols)
        block_id = self._get_block_id(flat_index)
        block = self._read_block(block_id)

        # Handle case where block is empty (beyond data)
        if len(block) == 0:
            return 0.0  # Return 0 for out-of-bounds access

        # Handle case where the index is beyond the block size
        index_in_block = self._get_index_in_block(flat_index)
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
        flat_index = self._get_flat_index(i, j, n_cols)

        # Check bounds before proceeding
        if not self._is_valid_index(flat_index):
            return  # Skip out-of-bounds writes

        block_id = self._get_block_id(flat_index)
        block = self._read_block(block_id)

        index_in_block = self._get_index_in_block(flat_index)

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
