# Cache-Aware Matrix Transposition

Implementation of a cache-aware matrix transposition algorithm with I/O operation simulation for large matrices that don't fit in memory.

## Problem Description

### Matrix Transposition Task

- Need to transform a square matrix `m x m`: element `(i, j)` is swapped with element `(j, i)`
- Input: large matrix that doesn't fit in memory; stored in _row-major order_ (row-wise)
- Condition: even a single row or column doesn't fit entirely in internal memory

### Naive Algorithm Problem

- **Straightforward algorithm**: traverse matrix row-wise, for each element `(i, j)` find `(j, i)` and swap them
- **I/O behavior**: 
  - First half: reading in storage order → `n/B` I/O operations
  - Second half: accessing `(j, i)` requires column-wise scanning → `n` I/O operations
- **Conclusion**: naive algorithm requires about `n` I/O operations and is inefficient

## Cache-Aware Algorithm

### Main Idea

Break the matrix into square tiles of size `t x t`, choosing `t` so that **two tiles fit completely in memory**.

### Tile Size Selection

```
t = sqrt(M) - B
```

Where:
- `M` - available memory size
- `B` - disk block size
- Condition: `2*(t^2 + 2*B*t) <= M` (accounting for "protruding" blocks in row-major storage)

### Algorithm

1. **Tile partitioning**: matrix is divided into square tiles of size `t x t`
2. **Tile pair processing**: for each pair of tiles `(i,j)` and `(j,i)`:
   - Read both tiles into memory
   - Perform transposition in memory
   - Write result back to disk
3. **In-place transposition**: 
   - Diagonal tiles: transpose in-place
   - Symmetric pairs: swap and transpose

### I/O Complexity Analysis

- **Number of tile pairs**: `n / (t^2)`, where `n = m * m`
- **Operations per pair**: `4t * (t/B + 2)`
- **Final complexity**: `O(n/B + n/t)`
- **With tall-cache condition**: `O(n/B)` - optimal bound!

## Project Structure

```
.
├── algorithms/
│   ├── __init__.py
│   └── transpose_cache_aware.py    # Main algorithm
├── io_simulator/
│   ├── __init__.py
│   └── io_simulator.py             # I/O operations simulator
├── tests/
│   ├── test_io_simulator.py        # Tests for IOSimulator
│   └── test_transpose.py           # Tests for transposition algorithm
├── Makefile                        # Task automation
├── algorithm_analysis.md           # Detailed algorithm analysis
└── README.md                       # This file
```

## Installation and Usage

### Requirements

- Python 3.7+
- NumPy

### Installing Dependencies

```bash
make install
# or
pip install numpy
```

### Quick Start

```bash
# Run built-in example
make example

# Run all tests
make test

# Run additional examples
make examples
```

### Programmatic Usage

```python
import numpy as np
from io_simulator import IOSimulator
from algorithms import transpose_cache_aware

# Create matrix
A = np.arange(16).reshape(4, 4)

# Create I/O simulator with parameters:
# - block_size=2: disk block size
# - memory_size=8: available memory size
sim = IOSimulator(A, block_size=2, memory_size=8)

# Perform transposition
result, io_count = transpose_cache_aware(sim)

print(f"Transposition result:")
print(result)
print(f"Number of I/O operations: {io_count}")
```

## API

### IOSimulator

Class for simulating I/O operations with caching:

```python
sim = IOSimulator(matrix, block_size, memory_size)
```

**Parameters:**
- `matrix`: NumPy array (matrix)
- `block_size`: disk block size
- `memory_size`: available memory size

**Methods:**
- `get_submatrix(i_start, i_end, j_start, j_end)`: get submatrix
- `set_submatrix(i, j, submatrix)`: set submatrix
- `flush_memory()`: flush cache to disk

### transpose_cache_aware

Cache-aware transposition function:

```python
result, io_count = transpose_cache_aware(sim)
```

**Parameters:**
- `sim`: IOSimulator instance

**Returns:**
- `result`: transposed matrix
- `io_count`: number of I/O operations performed

## Testing

### Running Tests

```bash
# All tests
make test

# IOSimulator tests only
make test-io

# Transposition tests only
make test-transpose

# Quick check
make quick-test
```

### Test Coverage

```bash
# Install coverage (if needed)
pip install coverage

# Run with coverage
make test-coverage
```

## Usage Examples

### Different Matrix Sizes

```bash
# Small matrix (2x2)
make example-small

# Large matrix (8x8)
make example-large

# All examples
make examples
```

### Performance Analysis

```python
import numpy as np
from io_simulator import IOSimulator
from algorithms import transpose_cache_aware
import math

# Test different memory parameters
test_cases = [
    (16, 2, 'Small memory'),
    (32, 2, 'Medium memory'),
    (64, 4, 'Large memory'),
    (128, 8, 'Very large memory')
]

for memory_size, block_size, description in test_cases:
    print(f'{description}: M={memory_size}, B={block_size}')
    
    # Calculate tile size
    tile_size = max(1, int(math.sqrt(memory_size)) - block_size)
    print(f'  Tile size: t = sqrt({memory_size}) - {block_size} = {tile_size}')
    
    # Create test matrix
    A = np.arange(16).reshape(4, 4)
    sim = IOSimulator(A, block_size=block_size, memory_size=memory_size)
    
    result, io_count = transpose_cache_aware(sim)
    
    print(f'  I/O operations: {io_count}')
    print(f'  Theoretical O(n/B) bound: ~{A.shape[0] * A.shape[1] // block_size}')
    print()
```

## Performance Results

| Memory (M) | Block (B) | Tile (t) | I/O Operations | Matrix | O(n/B) Bound |
|------------|-----------|----------|----------------|--------|--------------|
| 16         | 2         | 2        | 16             | 4x4    | ~8            |
| 32         | 2         | 3        | 24             | 4x4    | ~8            |
| 64         | 4         | 4        | 8              | 4x4    | ~4            |
| 128        | 8         | 3        | 16             | 4x4    | ~2            |

## Theoretical Foundation

### Cache-Aware Algorithm Advantages

1. **Optimal I/O complexity**: `O(n/B)` instead of `O(n)` for naive algorithm
2. **Efficient memory usage**: two tiles fit in memory simultaneously
3. **Architecture awareness**: algorithm adapts to system parameters (M, B)

### Limitations

1. **Requires parameter knowledge**: need to know memory size M and block size B
2. **Square matrices only**: algorithm works only with square matrices
3. **In-place operations**: modifies the original matrix

## Additional Materials

- [Detailed algorithm analysis](algorithm_analysis.md)
- [Algorithm source code](algorithms/transpose_cache_aware.py)
- [Tests and examples](tests/)

## License

MIT License

## Author

Implementation of cache-aware matrix transposition algorithm for external memory algorithms course.
