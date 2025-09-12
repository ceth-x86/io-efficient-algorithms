# I/O Simulator Module

This module provides a simulator for disk I/O operations with memory caching, designed to work with flat 1D arrays while providing matrix-like access patterns.

## Overview

The `IOSimulator` class simulates reading and writing data to/from disk in blocks, with a limited memory cache. It's designed to work with flat 1D arrays while providing matrix-like access patterns through `get_element`/`set_element` methods.

## Features

- **Block-based I/O**: Data is read and written in configurable block sizes
- **Memory caching**: Limited memory cache with LRU eviction policy
- **Matrix-like access**: Provides `get_element`/`set_element` for matrix operations
- **Submatrix operations**: Efficient `get_submatrix`/`set_submatrix` methods
- **I/O tracking**: Counts all disk I/O operations for performance analysis
- **Data persistence**: Automatic write-back of modified blocks during cache eviction
- **Dirty block tracking**: Ensures no data loss during cache management

## Usage

### Basic Usage

```python
import numpy as np
from io_simulator import IOSimulator

# Create a matrix
matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

# Initialize simulator
sim = IOSimulator(matrix, block_size=2, memory_size=8)

# Read elements
value = sim.get_element(0, 1, 4)  # Get element at row 0, col 1
print(f"Value: {value}")  # Output: 2

# Write elements
sim.set_element(0, 1, 99, 4)  # Set element at row 0, col 1 to 99

# Read submatrix
submatrix = sim.get_submatrix(0, 2, 0, 2, 4)  # Get 2x2 submatrix
print(f"Submatrix:\n{submatrix}")

# Write submatrix
new_submatrix = np.array([[100, 101], [102, 103]])
sim.set_submatrix(0, 0, new_submatrix, 4)

# Flush changes to disk
sim.flush_memory()

# Check I/O count
print(f"I/O operations: {sim.io_count}")
```

### Advanced Usage

```python
# Work with different data types
float_matrix = matrix.astype(np.float32)
sim = IOSimulator(float_matrix, block_size=4, memory_size=16)

# Handle rectangular matrices
rect_matrix = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
sim = IOSimulator(rect_matrix, block_size=2, memory_size=8)

# Access elements (note: n_cols=3 for 2x3 matrix)
value = sim.get_element(0, 2, 3)  # Get element at row 0, col 2
```

## API Reference

### IOSimulator Class

#### Constructor

```python
IOSimulator(data: np.ndarray, block_size: int, memory_size: int)
```

- `data`: Input data (will be flattened to 1D)
- `block_size`: Size of each I/O block
- `memory_size`: Maximum memory available for caching

#### Public Methods

##### Element Access

- `get_element(i: int, j: int, n_cols: int) -> float`
  - Get element at position (i, j) in a matrix with n_cols columns
  - Returns 0.0 for out-of-bounds access

- `set_element(i: int, j: int, value: float, n_cols: int) -> None`
  - Set element at position (i, j) in a matrix with n_cols columns
  - Silently ignores out-of-bounds writes

##### Submatrix Operations

- `get_submatrix(i_start: int, i_end: int, j_start: int, j_end: int, n_cols: int) -> np.ndarray`
  - Read submatrix into numpy array (with I/O)
  - Returns 2D numpy array

- `set_submatrix(i_start: int, j_start: int, submat: np.ndarray, n_cols: int) -> None`
  - Write numpy array back (with I/O)

##### Memory Management

- `flush_memory() -> None`
  - Flush all cached blocks to disk and clear memory
  - Should be called to ensure data persistence

#### Attributes

- `disk`: The flat 1D array representing disk storage
- `total_size`: Total number of elements in the array
- `block_size`: Size of each I/O block
- `memory_size`: Maximum memory available for caching
- `io_count`: Number of I/O operations performed
- `memory`: Cache storing blocks in memory
- `dirty_blocks`: Set of block IDs that have been modified
- `memory_limit`: Maximum number of blocks that can be cached

## Performance Considerations

### Block Size Selection

- **Small blocks**: More I/O operations, but better cache utilization
- **Large blocks**: Fewer I/O operations, but may waste memory
- **Optimal**: Usually `block_size = sqrt(memory_size)` or similar

### Memory Management

- Memory is managed using LRU (Least Recently Used) eviction
- **Automatic write-back**: Modified blocks are automatically written to disk during eviction
- **Dirty block tracking**: Prevents data loss by tracking which blocks need saving
- `flush_memory()` should be called to ensure all changes are persistent
- Memory limit is calculated as `memory_size // block_size`

### I/O Counting

- **Explicit I/O operations** are counted in `io_count` (reads via `get_element`, writes via `flush_memory`)
- **Automatic write-back during eviction** does NOT count toward I/O statistics
- This ensures fair comparison of algorithms independent of cache size
- Useful for performance analysis and algorithm optimization
- Reset to 0 on initialization

## Important Changes and Fixes

### Cache Eviction Data Loss Prevention (v2.0)

**Problem Fixed**: Previously, when the cache became full and blocks were evicted, any modifications to those blocks were **lost forever**.

```python
# Before fix - DATA LOSS! 😱
sim = IOSimulator(data, block_size=1, memory_size=2)  # Cache for 2 blocks
sim.set_element(0, 0, 99, 4)  # Modify block 0 → stored in cache
sim.set_element(0, 4, 88, 4)  # Need block 2 → block 0 EVICTED WITHOUT SAVING!
# Value 99 is permanently lost!
```

**Solution Implemented**:
- **Dirty block tracking**: `dirty_blocks` set tracks which blocks have unsaved changes
- **Automatic write-back**: Before evicting a block, check if it's dirty and save it to disk
- **Data integrity**: Guaranteed no data loss during cache management

```python
# After fix - DATA SAFE! ✅
sim = IOSimulator(data, block_size=1, memory_size=2)  # Cache for 2 blocks
sim.set_element(0, 0, 99, 4)  # Modify block 0 → marked as dirty
sim.set_element(0, 4, 88, 4)  # Need block 2 → block 0 automatically saved before eviction
# Value 99 is preserved on disk!
```

### I/O Statistics Accuracy

**Problem Fixed**: Automatic cache management operations were incorrectly counted in I/O statistics.

**Solution**: 
- **Separation of concerns**: `_write_block()` (explicit) vs `_write_block_to_disk_only()` (automatic)
- **Fair algorithm comparison**: Only algorithm-requested operations count toward I/O complexity
- **Cache-size independence**: Algorithm I/O measurements don't vary with cache size

### Comprehensive Testing

**New test suite** `TestIOSimulatorCacheEviction` covers:
- ✅ Data persistence during cache evictions  
- ✅ Dirty block tracking accuracy
- ✅ I/O count correctness
- ✅ Large dataset cache management
- ✅ Mixed read/write operation safety
- ✅ LRU eviction policy with dirty blocks

**Run cache-specific tests**:
```bash
python -m pytest tests/test_io_simulator.py::TestIOSimulatorCacheEviction -v
```

### Compatibility

All existing code remains **100% compatible**. The fixes are internal improvements that enhance reliability without changing the public API.

## Examples

### Cache-Aware Matrix Transpose

```python
def transpose_with_simulator(matrix):
    sim = IOSimulator(matrix, block_size=2, memory_size=8)
    n_rows, n_cols = matrix.shape
    
    # Perform transpose using submatrix operations
    for i in range(0, n_rows, 2):
        for j in range(0, n_cols, 2):
            # Read submatrix
            submatrix = sim.get_submatrix(i, i+2, j, j+2, n_cols)
            
            # Transpose submatrix
            submatrix = submatrix.T
            
            # Write back
            sim.set_submatrix(i, j, submatrix, n_cols)
    
    # Flush changes
    sim.flush_memory()
    
    return sim.disk.reshape(n_cols, n_rows), sim.io_count
```

### Memory Usage Analysis

```python
# Test different memory sizes
for memory_size in [4, 8, 16, 32]:
    sim = IOSimulator(matrix, block_size=2, memory_size=memory_size)
    
    # Perform some operations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            sim.get_element(i, j, matrix.shape[1])
    
    print(f"Memory: {memory_size}, I/O count: {sim.io_count}")
```

## Testing

### Run All Tests

```bash
# Using pytest (recommended)
python -m pytest tests/test_io_simulator.py -v

# Using Makefile
make test-io
```

### Run Specific Test Suites

```bash
# Test basic functionality
python -m pytest tests/test_io_simulator.py::TestIOSimulator -v

# Test cache eviction fixes (NEW)
python -m pytest tests/test_io_simulator.py::TestIOSimulatorCacheEviction -v

# Test specific cache functionality
python -m pytest tests/test_io_simulator.py::TestIOSimulatorCacheEviction::test_cache_eviction_preserves_dirty_blocks -v
```

### Performance Testing

```bash
# Test with different memory configurations
python -c "
import numpy as np
from io_simulator import IOSimulator

data = np.arange(100)
for mem_size in [4, 8, 16, 32]:
    sim = IOSimulator(data, block_size=2, memory_size=mem_size)
    # Perform operations that cause evictions
    for i in range(100):
        sim.set_element(0, i, i+100, 100)
    sim.flush_memory()
    print(f'Memory {mem_size}: {sim.io_count} I/O ops, result correct: {np.array_equal(sim.disk, np.arange(100,200))}')
"
```

## Dependencies

- `numpy`: For array operations and data handling
