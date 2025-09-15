# I/O Efficient Algorithms

Implementation of various I/O efficient algorithms for external memory computation with simulation framework for analyzing performance on large datasets that don't fit in memory.

## Currently Implemented Algorithms

- **Matrix Transposition**: Cache-aware and cache-oblivious algorithms for transposing large square matrices
- **External Memory Sorting**: External merge sort algorithm for sorting datasets larger than memory
- **B-Trees**: External memory search trees for efficient dictionary operations
- **Buffer Trees**: Advanced batched processing trees achieving optimal sorting bound
- **Priority Queues**: Phase-based priority queues achieving optimal I/O complexity for batch operations
- **Time-Forward Processing**: I/O-efficient algorithms for computing local functions on DAGs
- **Maximal Independent Sets**: Graph algorithms using time-forward processing for optimal I/O complexity
- *More algorithms coming soon...*

📖 **Algorithm Documentation:**
- **[Matrix Transpose Algorithms →](algorithms/transpose/README.md)**
- **[External Memory Sorting →](algorithms/sorting/README.md)**
- **[External Memory Search Structures →](algorithms/searching/README.md)**
  - **[B-Trees →](algorithms/searching/btree/README.md)**
  - **[Buffer Trees →](algorithms/searching/buffer_tree/README.md)**
  - **[Priority Queues →](algorithms/searching/priority_queue/README.md)**
- **[Time-Forward Processing →](algorithms/time_forward_processing/README.md)**

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
│   │   ├── priority_queue/           # Priority queue implementation
│   │   │   ├── __init__.py
│   │   │   ├── README.md             # Priority queue documentation
│   │   │   └── priority_queue.py     # Priority queue implementation
│   │   ├── b-trees.py                # Algorithm description (Russian)
│   │   └── buffer_trees.py           # Algorithm description (Russian)
│   ├── time_forward_processing/      # Time-forward processing algorithms
│   │   ├── __init__.py
│   │   ├── README.md                 # Time-forward processing documentation
│   │   └── maximal_independent_sets.py # Maximal independent sets implementation
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
│   ├── test_priority_queue.py        # Tests for Priority queue implementation
│   ├── test_maximal_independent_sets.py # Tests for time-forward processing
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
# Run examples
make example-transpose-cache-aware      # Matrix transpose (cache-aware)
make example-transpose-cache-oblivious  # Matrix transpose (cache-oblivious)
make example-sorting            # External memory sorting
make example-btree              # B-tree operations
make example-buffer-tree        # Buffer tree batch operations
make example-priority-queue     # Priority queue phase-based operations
make example-time-forward       # Time-forward processing
make example-maximal-independent-sets # Maximal independent sets

# Run all tests
make test                       # All algorithms and simulator

# Run specific test suites
make test-transpose            # Matrix transpose tests
make test-sorting              # External memory sorting tests
make test-btree                # B-tree tests
make test-buffer-tree          # Buffer tree tests
make test-priority-queue       # Priority queue tests
make test-time-forward         # Time-forward processing tests
make test-maximal-independent-sets # Maximal independent sets tests
make test-io                   # I/O simulator tests

# Run all examples
make examples                   # All available examples
```

### Basic Usage Examples

#### Matrix Transposition
```python
import numpy as np
from io_simulator import IOSimulator
from algorithms.transpose import transpose_cache_aware

# Create a large matrix that doesn't fit in memory
matrix = np.arange(16).reshape(4, 4)

# Initialize I/O simulator
simulator = IOSimulator(matrix, block_size=2, memory_size=8)

# Perform cache-aware transpose
result, io_count = transpose_cache_aware(simulator, 4, 4)
print(f"Transposed with {io_count} I/O operations")
```

#### B-Tree Operations
```python
import numpy as np
from io_simulator import IOSimulator
from algorithms.searching import BTree

# Setup external storage
disk_data = np.zeros(10000)
disk = IOSimulator(disk_data, block_size=50, memory_size=200)

# Create B-tree
btree = BTree(disk, d_min=3)

# Dictionary operations
btree.insert(10)
btree.insert(20)
found = btree.search(10)  # True
min_key = btree.find_min()  # 10
```

#### Buffer Tree Batch Operations
```python
import numpy as np
from io_simulator import IOSimulator
from algorithms.searching import BufferTree

# Setup larger external storage
disk_data = np.zeros(50000)
disk = IOSimulator(disk_data, block_size=50, memory_size=200)

# Create buffer tree for batch processing
buffer_tree = BufferTree(disk, degree=8)

# Batch operations (accumulated in memory)
for i in [10, 5, 15, 3, 7]:
    buffer_tree.insert(i, f"value_{i}")

# Process all operations together
buffer_tree.flush_all_operations()

# Search results
for i in [5, 10, 15]:
    buffer_tree.search(i)
buffer_tree.flush_all_operations()

# Check results - demonstrates batching efficiency
print(f"Total I/O operations: {buffer_tree.get_io_count()}")
```

#### Priority Queue Phase-Based Operations
```python
import numpy as np
from io_simulator import IOSimulator
from algorithms.searching.priority_queue import ExternalPriorityQueue

# Setup external storage
disk_data = np.zeros(50000)
disk = IOSimulator(disk_data, block_size=50, memory_size=200)

# Create priority queue with phase size M/4 = 50
pq = ExternalPriorityQueue(disk, memory_size=200, block_size=50, degree=8)

# Insert elements (goes to buffer tree when no active phase)
priorities = [10, 5, 15, 3, 7, 12, 18, 1, 4, 6]
for priority in priorities:
    pq.insert(priority, f"value_{priority}")

# First extract starts a phase - loads M/4 minimum elements
min_elem = pq.extract_min()  # Gets (1, "value_1")
print(f"First extract: {min_elem}")

# Subsequent extracts work from in-memory set
while not pq.is_empty():
    elem = pq.extract_min()
    if elem:
        print(f"Extracted: {elem}")

print(f"Total I/O operations: {pq.get_io_count()}")
print("Phase-based processing demonstrates I/O efficiency!")
```

#### Time-Forward Processing and Maximal Independent Sets
```python
import numpy as np
from io_simulator import IOSimulator
from algorithms.time_forward_processing import Graph, MaximalIndependentSetSolver

# Setup external memory
disk_data = np.zeros(10000)
disk = IOSimulator(disk_data, block_size=50, memory_size=200)
solver = MaximalIndependentSetSolver(disk)

# Create path graph: 0-1-2-3-4
graph = Graph(5)
for i in range(4):
    graph.add_edge(i, i + 1)

# Compute maximal independent set using time-forward processing
independent_set, values = solver.solve(graph)
print(f"Graph: Path 0-1-2-3-4")
print(f"Maximal independent set: {sorted(independent_set)}")
print(f"Is independent: {solver.verify_independence(graph, independent_set)}")
print(f"Is maximal: {solver.verify_maximality(graph, independent_set)}")
print(f"I/O operations: {solver.get_io_count()}")
print("Time-forward processing achieves O(sort(V+E)) I/O complexity!")
```

## Documentation

### Core Framework
- **[I/O Simulator Framework](io_simulator/README.md)** - Detailed IOSimulator documentation with LRU cache management

### Algorithms
- **[Matrix Transpose Algorithms](algorithms/transpose/README.md)** - Cache-aware and cache-oblivious transpose implementations
- **[External Memory Sorting](algorithms/sorting/README.md)** - External merge sort algorithm and analysis
- **[External Memory Search Structures](algorithms/searching/README.md)** - Overview of search data structures
  - **[B-Trees](algorithms/searching/btree/README.md)** - Classic balanced trees for interactive operations
  - **[Buffer Trees](algorithms/searching/buffer_tree/README.md)** - Advanced batched processing achieving sorting bound
  - **[Priority Queues](algorithms/searching/priority_queue/README.md)** - Phase-based priority queues for batch operations
- **[Time-Forward Processing](algorithms/time_forward_processing/README.md)** - I/O-efficient algorithms for computing local functions on DAGs with maximal independent sets

## Testing

### Running Tests

```bash
# All tests
make test

# Specific test categories
make test-io                   # I/O simulator tests
make test-transpose           # Matrix transpose tests
make test-sorting             # External merge sort tests  
make test-btree               # B-tree tests
make test-buffer-tree         # Buffer tree tests
make test-time-forward        # Time-forward processing tests
make test-maximal-independent-sets # Maximal independent sets tests

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
make example-transpose-cache-aware  # Default 4x4 matrix
make example-small                  # Small 2x2 matrix
make example-large                  # Large 8x8 matrix

# Cache-oblivious algorithm examples  
make example-transpose-cache-oblivious  # Recursive approach
```

### External Memory Algorithms Examples

```bash
# External sorting
make example-sorting        # External merge sort demonstration

# B-tree operations (individual operations)
make example-btree         # Dictionary operations with O(log_B n) per operation

# Buffer tree operations (batch processing)  
make example-buffer-tree   # Batch operations achieving sorting bound

# Priority queue operations (phase-based processing)
make example-priority-queue # Phase-based operations achieving sorting bound

# Time-forward processing algorithms
make example-time-forward  # Time-forward processing demonstration
make example-maximal-independent-sets # Maximal independent sets computation

# Run all examples
make examples              # All implemented algorithms
```

### Custom Usage Examples

```bash
# External Sorting
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

# B-tree vs Buffer Tree vs Priority Queue I/O comparison
python -c "
import numpy as np
from io_simulator import IOSimulator
from algorithms.searching import BTree, BufferTree
from algorithms.searching.priority_queue import ExternalPriorityQueue

# B-tree: individual operations
disk1 = IOSimulator(np.zeros(10000), block_size=50, memory_size=200)
btree = BTree(disk1, d_min=3)
for i in range(20):
    btree.insert(i)
print(f'B-tree (20 inserts): {btree.get_io_count()} I/O operations')

# Buffer tree: batch operations  
disk2 = IOSimulator(np.zeros(50000), block_size=50, memory_size=200)
buffer_tree = BufferTree(disk2, degree=8)
for i in range(20):
    buffer_tree.insert(i, f'value_{i}')
buffer_tree.flush_all_operations()
print(f'Buffer tree (20 batch inserts): {buffer_tree.get_io_count()} I/O operations')

# Priority queue: phase-based operations
disk3 = IOSimulator(np.zeros(50000), block_size=50, memory_size=200)
pq = ExternalPriorityQueue(disk3, memory_size=200, block_size=50, degree=8)
for i in range(20):
    pq.insert(i, f'item_{i}')
extracted = []
while not pq.is_empty():
    elem = pq.extract_min()
    if elem: extracted.append(elem[0])
print(f'Priority queue (20 inserts + extracts): {pq.get_io_count()} I/O operations')
print('Phase-based processing demonstrates batching advantage!')
"
```

## License

MIT License
