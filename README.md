# I/O Efficient Algorithms

Implementation of various I/O efficient algorithms for external memory computation with simulation framework for analyzing performance on large datasets that don't fit in memory.

## Currently Implemented Algorithms

- **Matrix Transposition**: Cache-aware and cache-oblivious algorithms for transposing large square matrices
- **External Memory Sorting**: External merge sort algorithm for sorting datasets larger than memory
- **B-Trees**: External memory search trees for efficient dictionary operations
- **Buffer Trees**: Advanced batched processing trees achieving optimal sorting bound
- *More algorithms coming soon...*

📖 **Algorithm Documentation:**
- **[Matrix Transpose Algorithms →](algorithms/transpose/README.md)**
- **[External Memory Sorting →](algorithms/sorting/README.md)**
- **[B-Trees for External Memory →](algorithms/searching/README.md)**

## I/O Model and Framework

### External Memory Model

- **Large datasets**: Data structures too large to fit in internal memory
- **Two-level hierarchy**: Fast internal memory (size M) and slow external storage
- **Block-based I/O**: Data transferred in blocks of size B between levels
- **Goal**: Minimize the number of I/O operations to achieve optimal performance

### Performance Metrics

- **I/O complexity**: Number of block transfers between memory and storage
- **Space complexity**: Amount of internal memory used
- **Optimal bounds**: Theoretical lower bounds for specific problems
- **Cache-oblivious**: Algorithms that work optimally without knowing M and B

## Project Structure

```
.
├── algorithms/
│   ├── __init__.py
│   ├── [future README files...]      # Algorithm category documentation
│   ├── searching/                    # External memory search structures
│   │   ├── __init__.py
│   │   ├── README.md                 # Search structures overview
│   │   ├── btree/                    # B-tree implementation
│   │   │   ├── __init__.py
│   │   │   ├── README.md             # B-tree documentation
│   │   │   └── btree.py              # B-tree implementation
│   │   ├── buffer_tree/              # Buffer tree implementation
│   │   │   ├── __init__.py
│   │   │   ├── README.md             # Buffer tree documentation
│   │   │   └── buffer_tree.py        # Buffer tree implementation
│   │   ├── b-trees.py                # Algorithm description (Russian)
│   │   └── buffer_trees.py           # Algorithm description (Russian)
│   ├── sorting/                      # External memory sorting algorithms
│   │   ├── __init__.py
│   │   ├── README.md                 # Sorting algorithm documentation  
│   │   └── external_merge_sort.py    # External merge sort implementation
│   ├── transpose/                    # Matrix transpose algorithms
│   │   ├── __init__.py
│   │   ├── README.md                 # Matrix transpose documentation
│   │   ├── cache_aware.py            # Cache-aware matrix transpose
│   │   └── cache_oblivious.py        # Cache-oblivious matrix transpose
│   └── [future algorithms...]        # Additional I/O efficient algorithms
├── io_simulator/
│   ├── __init__.py
│   ├── README.md                     # IOSimulator documentation
│   └── io_simulator.py               # Core I/O operations simulator
├── tests/
│   ├── test_io_simulator.py          # Tests for I/O simulator
│   ├── test_btree.py                 # Tests for B-tree implementation
│   ├── test_buffer_tree.py           # Tests for Buffer tree implementation
│   ├── test_external_merge_sort.py   # Tests for external merge sort
│   ├── test_transpose_cache_aware.py # Tests for cache-aware transpose
│   ├── test_transpose_cache_oblivious.py # Tests for cache-oblivious transpose
│   └── [future test files...]        # Tests for additional algorithms
├── Makefile                          # Build and test automation
├── CLAUDE.md                         # AI assistant guidance
├── pyproject.toml                    # Python project configuration
└── README.md                         # This documentation
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

# For testing (optional)
pip install pytest
```

### Quick Start

```bash
# Run matrix transpose examples
make example                    # Cache-aware algorithm
make example-cache-oblivious    # Cache-oblivious algorithm

# Run all tests
make test                       # All algorithms and simulator

# Run specific test suites
make test-transpose            # Matrix transpose tests
make test-io                   # I/O simulator tests

# Additional examples
make examples                   # All available examples
```

### Basic Usage

```python
import numpy as np
from io_simulator import IOSimulator

# Create a large matrix that doesn't fit in memory
matrix = np.arange(16).reshape(4, 4)

# Initialize I/O simulator
simulator = IOSimulator(matrix, block_size=2, memory_size=8)

# Use with algorithms (see algorithms/README.md for details)
```

## Documentation

### Core Framework
- **[I/O Simulator Framework](io_simulator/README.md)** - Detailed IOSimulator documentation with LRU cache management

### Algorithms
- **[Matrix Transpose Algorithms](algorithms/transpose/README.md)** - Cache-aware and cache-oblivious transpose implementations
- **[External Memory Sorting](algorithms/sorting/README.md)** - External merge sort algorithm and analysis
- **[B-Trees for External Memory](algorithms/searching/README.md)** - External memory search trees and dictionary operations

## Testing

### Running Tests

```bash
# All tests
make test

# Specific test categories
make test-io                   # I/O simulator tests
make test-transpose           # Matrix transpose tests
make test-sorting             # External merge sort tests

# Quick verification
make quick-test
```

### Test Coverage

```bash
# Install coverage (if needed)
pip install coverage

# Run with coverage
make test-coverage
```

## Examples

### Matrix Transposition Examples

```bash
# Cache-aware algorithm examples
make example          # Default 4x4 matrix
make example-small    # Small 2x2 matrix
make example-large    # Large 8x8 matrix

# Cache-oblivious algorithm examples  
make example-cache-oblivious  # Recursive approach

# Run all examples
make examples         # All implemented algorithms
```

### External Sorting Examples

```bash
# Test external merge sort on different dataset sizes
python -c "
import numpy as np
from io_simulator import IOSimulator
from algorithms.sorting import external_merge_sort

# Test with different configurations
test_array = np.random.randint(0, 100, 20)
print(f'Original: {test_array}')

sim = IOSimulator(test_array, block_size=4, memory_size=8)
sorted_result, io_count = external_merge_sort(sim, len(test_array))

print(f'Sorted: {sorted_result}')
print(f'I/O operations: {io_count}')
"
```

### Future Algorithm Examples

*As more algorithms are added, corresponding examples will be available through make commands.*

## License

MIT License
