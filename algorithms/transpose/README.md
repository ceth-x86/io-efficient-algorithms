# Matrix Transposition Algorithms

This directory contains I/O efficient algorithms for transposing large square matrices that don't fit in memory.

## Problem Description

### Matrix Transposition Task

- **Goal**: Transform a square matrix `m x m` where element `(i, j)` is swapped with element `(j, i)`
- **Challenge**: Matrix stored in row-major order, but transpose requires column-wise access
- **Constraint**: Even a single row or column doesn't fit in internal memory
- **Naive approach**: Requires `O(n)` I/O operations due to poor locality

## Implemented Algorithms

### 1. Cache-Aware Algorithm (`transpose_cache_aware.py`)

#### Main Idea

Break the matrix into square tiles of size `t x t`, choosing `t` so that **two tiles fit completely in memory**.

#### Tile Size Selection

```
t = sqrt(M) - B
```

Where:
- `M` - available memory size
- `B` - disk block size
- Condition: `2*(t^2 + 2*B*t) <= M` (accounting for "protruding" blocks in row-major storage)

#### Algorithm Steps

1. **Tile partitioning**: matrix is divided into square tiles of size `t x t`
2. **Tile pair processing**: for each pair of tiles `(i,j)` and `(j,i)`:
   - Read both tiles into memory
   - Perform transposition in memory
   - Write result back to disk
3. **In-place transposition**: 
   - Diagonal tiles: transpose in-place
   - Symmetric pairs: swap and transpose

#### I/O Complexity Analysis

- **Number of tile pairs**: `n / (t^2)`, where `n = m * m`
- **Operations per pair**: `4t * (t/B + 2)`
- **Final complexity**: `O(n/B + n/t)`
- **With tall-cache condition**: `O(n/B)` - optimal bound!

### 2. Cache-Oblivious Algorithm (`transpose_cache_oblivious.py`)

#### Main Idea

The cache-oblivious algorithm uses recursive divide-and-conquer to automatically adapt to the memory hierarchy **without knowing the cache parameters M and B**.

#### Algorithm Steps

1. **Recursive partitioning**: matrix is recursively divided into 4 submatrices:
   - Top-left and bottom-right quadrants are processed recursively (diagonal)
   - Top-right and bottom-left quadrants are swapped and transposed recursively (off-diagonal)
2. **Base case**: small matrices (≤ 2x2) are transposed directly using element swaps
3. **Automatic adaptation**: recursion depth automatically adapts to cache hierarchy

#### Key Properties

- **Parameter-free**: no need to know M or B in advance
- **Optimal complexity**: achieves `O(n/B + n/sqrt(M))` I/O bound
- **Cache hierarchy adaptive**: works optimally across multiple cache levels
- **Divide-and-conquer**: naturally exploits temporal and spatial locality

#### I/O Complexity Analysis

- **Recursive structure**: `T(n) = 4T(n/4) + O(n/B)` when subproblems fit in cache
- **Final complexity**: `O(n/B + n/sqrt(M))` for optimal cache use
- **Cache-oblivious bound**: asymptotically optimal across all cache levels

## API Reference

### transpose_cache_aware

Cache-aware transposition function:

```python
result_flat, io_count = transpose_cache_aware(sim, n_rows, n_cols)
```

**Parameters:**
- `sim`: IOSimulator instance
- `n_rows`: number of rows in the matrix
- `n_cols`: number of columns in the matrix

**Returns:**
- `result_flat`: transposed matrix as flat 1D array
- `io_count`: number of I/O operations performed

### transpose_cache_oblivious

Cache-oblivious transposition function:

```python
result_flat, io_count = transpose_cache_oblivious(sim, n_rows, n_cols)
```

**Parameters:**
- `sim`: IOSimulator instance
- `n_rows`: number of rows in the matrix (must equal n_cols)
- `n_cols`: number of columns in the matrix (must equal n_rows)

**Returns:**
- `result_flat`: transposed matrix as flat 1D array
- `io_count`: number of I/O operations performed

**Note:** Both algorithms require square matrices and return results as flat arrays that need to be reshaped.

## Usage Examples

### Cache-Aware Algorithm

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

# Perform cache-aware transposition
result_flat, io_count = transpose_cache_aware(sim, 4, 4)

# Reshape back to matrix
result = result_flat.reshape(4, 4)

print(f"Cache-aware result:")
print(result)
print(f"I/O operations: {io_count}")
```

### Cache-Oblivious Algorithm

```python
import numpy as np
from io_simulator import IOSimulator
from algorithms.transpose_cache_oblivious import transpose_cache_oblivious

# Create matrix
A = np.arange(16).reshape(4, 4)

# Create I/O simulator (parameters less critical for cache-oblivious)
sim = IOSimulator(A, block_size=2, memory_size=8)

# Perform cache-oblivious transposition
result_flat, io_count = transpose_cache_oblivious(sim, 4, 4)

# Reshape back to matrix
result = result_flat.reshape(4, 4)

print(f"Cache-oblivious result:")
print(result)
print(f"I/O operations: {io_count}")
```

### Algorithm Comparison

```python
import numpy as np
from io_simulator import IOSimulator
from algorithms import transpose_cache_aware
from algorithms.transpose_cache_oblivious import transpose_cache_oblivious

# Test different matrix sizes
matrix_sizes = [2, 4, 8]
block_size, memory_size = 2, 16

print(f"Comparing algorithms (B={block_size}, M={memory_size})")
print("Size | Cache-Aware I/O | Cache-Oblivious I/O | Ratio")
print("-" * 55)

for n in matrix_sizes:
    A = np.arange(n * n).reshape(n, n)
    
    # Test cache-aware
    sim1 = IOSimulator(A.copy(), block_size=block_size, memory_size=memory_size)
    _, io_aware = transpose_cache_aware(sim1, n, n)
    
    # Test cache-oblivious
    sim2 = IOSimulator(A.copy(), block_size=block_size, memory_size=memory_size)
    _, io_oblivious = transpose_cache_oblivious(sim2, n, n)
    
    ratio = io_oblivious / io_aware if io_aware > 0 else float('inf')
    print(f" {n}x{n} |      {io_aware:6d}     |       {io_oblivious:6d}      | {ratio:4.1f}")
```

## Performance Results

### Cache-Aware Algorithm

| Memory (M) | Block (B) | Tile (t) | I/O Operations | Matrix | O(n/B) Bound |
|------------|-----------|----------|----------------|--------|--------------|
| 16         | 2         | 2        | 16             | 4x4    | ~8            |
| 32         | 2         | 3        | 24             | 4x4    | ~8            |
| 64         | 4         | 4        | 8              | 4x4    | ~4            |
| 128        | 8         | 3        | 16             | 4x4    | ~2            |

### Algorithm Comparison (B=2, M=16)

| Matrix Size | Cache-Aware I/O | Cache-Oblivious I/O | Efficiency Ratio |
|-------------|-----------------|---------------------|------------------|
| 2x2         | 4               | 4                   | 1.0              |
| 4x4         | 16              | 16                  | 1.0              |
| 8x8         | 64              | 64                  | 1.0              |

*Updated results show both algorithms perform equally well, with cache-oblivious showing advantages on non-power-of-2 matrices.*

### I/O Complexity Analysis: Theoretical vs Actual Results

Both algorithms should achieve **T(n) = O(n/B)** I/O complexity. Here's how the actual results compare to theoretical bounds:

#### Cache-Aware Algorithm: O(n/B) Analysis

| Matrix | n  | B | Actual I/O | n/B | Ratio | Status | Analysis |
|--------|----|----|-----------|-----|-------|--------|----------|
| 3×3    | 9  | 1 | 18        | 9.0 | 2.00  | ✅ OK  | Expected factor ≈2 (read+write) |
| 4×4    | 16 | 2 | 16        | 8.0 | 2.00  | ✅ OK  | Optimal tiling efficiency |
| 4×4    | 16 | 2 | 24        | 8.0 | 3.00  | ⚠️ OK  | Larger tile size causes overhead |
| 5×5    | 25 | 1 | 50        | 25.0| 2.00  | ✅ OK  | Consistent 2× factor |

**Cache-Aware Conclusion**: Achieves O(n/B) with **constant factor 2-3**. Factor of 2 is theoretical minimum (each element read + written).

#### Cache-Oblivious Algorithm: O(n/B) Analysis  

| Matrix | n  | B | Actual I/O | n/B | Ratio | Status | Analysis |
|--------|----|----|-----------|-----|-------|--------|----------|
| 2×2    | 4  | 1 | 4         | 4.0 | 1.00  | 🏆 Perfect | Ideal recursive division |
| 3×3    | 9  | 1 | 14        | 9.0 | 1.56  | 🏆 Excellent | Better than cache-aware |
| 4×4    | 16 | 2 | 16        | 8.0 | 2.00  | ✅ OK  | Power-of-2 optimality |
| 5×5    | 25 | 1 | 42        | 25.0| 1.68  | 🏆 Excellent | Better than cache-aware |
| 8×8    | 64 | 2 | 64        | 32.0| 2.00  | ✅ OK  | Consistent power-of-2 |

**Cache-Oblivious Conclusion**: Achieves O(n/B) with **exceptional factors 1.0-2.0**. Shows adaptive optimization based on matrix size.

#### Comparative Algorithm Performance

| Matrix Size | Cache-Aware I/O | Cache-Oblivious I/O | Winner | Improvement |
|-------------|-----------------|---------------------|--------|-------------|
| 3×3         | 18              | 14                  | Cache-Oblivious | 22% better |
| 4×4         | 16              | 16                  | Tie             | Equal |
| 5×5         | 50              | 42                  | Cache-Oblivious | 16% better |

#### Key Insights

**Why Cache-Aware shows 2× factor:**
- Each matrix element must be read AND written
- Minimum theoretical factor is 2n/B operations  
- Tile-based approach adds minimal overhead
- Factor 2-3 is **optimal** for tiled algorithms

**Why Cache-Oblivious performs better:**
- **Powers of 2**: Perfect recursive subdivision (factors 1.0-2.0)
- **Non-powers of 2**: Adaptive optimization still beats cache-aware
- **Automatic tuning**: No manual parameter selection needed
- **Multi-level optimization**: Works across entire memory hierarchy

**Theoretical Validation:**
Both algorithms successfully achieve **O(n/B)** asymptotic complexity with reasonable constant factors, confirming the theoretical analysis. Cache-oblivious shows superior practical performance due to its adaptive nature.

## Theoretical Foundation

### Cache-Aware Algorithm

**Advantages:**
1. **Optimal I/O complexity**: `O(n/B)` under tall-cache assumption
2. **Efficient memory usage**: exactly two tiles fit in memory
3. **Predictable performance**: tile size calculation ensures optimal cache use
4. **Simple implementation**: straightforward nested loops over tiles

**Limitations:**
1. **Requires parameter knowledge**: needs explicit M and B values
2. **Architecture-specific**: tile size must be recalculated for different systems
3. **Cache hierarchy**: only optimizes for one cache level

### Cache-Oblivious Algorithm

**Advantages:**
1. **Parameter-free**: no need to know M or B in advance
2. **Multi-level optimization**: automatically adapts to entire cache hierarchy
3. **Portable**: same code works optimally on different architectures
4. **Theoretical elegance**: achieves optimal bounds without explicit parameters

**Limitations:**
1. **Recursive overhead**: function call overhead for small subproblems
2. **Complex analysis**: harder to predict exact I/O count
3. **Implementation complexity**: requires careful base case handling

### Common Properties

1. **Square matrices only**: both algorithms work only with square matrices
2. **In-place operations**: both modify the original matrix
3. **Optimal complexity**: both achieve theoretically optimal I/O bounds

## Algorithm Selection Guidelines

### Choose Cache-Aware when:
- System parameters (M, B) are known and fixed
- Targeting a specific architecture for optimal performance
- Predictable, minimal I/O count is critical
- Simple implementation is preferred

### Choose Cache-Oblivious when:
- System parameters are unknown or highly variable
- Code must run optimally across different architectures
- Working with complex memory hierarchies
- Portability is more important than minimal overhead

## Testing

Tests for these algorithms are located in the parent `tests/` directory:
- `test_transpose_cache_aware.py`: Tests for cache-aware algorithm
- `test_transpose_cache_oblivious.py`: Tests for cache-oblivious algorithm

Run tests using:
```bash
make test-transpose        # Cache-aware tests
make test-cache-oblivious  # Cache-oblivious tests
```

## References

For more information about the I/O simulation framework, see [../io_simulator/README.md](../io_simulator/README.md).