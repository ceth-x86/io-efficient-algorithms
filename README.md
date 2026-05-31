# I/O Efficient Algorithms

Implementation of various I/O efficient algorithms for external memory computation with a simulation framework for analyzing performance on large datasets that don't fit in memory.

This repository contains implementations of tasks and algorithms from the Coursera course [I/O-Efficient Algorithms](https://www.coursera.org/learn/io-efficient-algorithms) and the Yandex School of Data Analysis (YSDA / ШАД) course "Algorithms in External Memory" (Алгоритмы во внешней памяти).

## Currently Implemented Algorithms

- **External Memory List Ranking**: Computes the ranks of nodes in a linked list using independent set reduction and recursive compaction.
- **Matrix Transposition**: Cache-aware and cache-oblivious algorithms for transposing large square matrices.
- **External Memory Sorting**: External merge sort algorithm for sorting datasets larger than memory.
- **B-Trees**: External memory search trees for efficient dictionary operations.
- **Buffer Trees**: Advanced batched processing trees achieving the optimal sorting bound.
- **Priority Queues**: Phase-based priority queues achieving optimal I/O complexity for batch operations.
- **Time-Forward Processing**: I/O-efficient algorithms for computing local functions on DAGs.
- **Maximal Independent Sets**: Graph algorithms using time-forward processing for optimal I/O complexity.
- **External Memory Stack**: A LIFO data structure utilizing a 2B RAM hysteresis buffer to achieve O(1/B) amortized I/O complexity.

## Algorithm Documentation

- **[External Memory List Ranking](algorithms/list_ranking/)**
- **[Matrix Transpose Algorithms](algorithms/transpose/README.md)**
- **[External Memory Sorting](algorithms/sorting/README.md)**
- **[External Memory Search Structures](algorithms/searching/README.md)**
  - **[B-Trees](algorithms/searching/btree/README.md)**
  - **[Buffer Trees](algorithms/searching/buffer_tree/README.md)**
  - **[Priority Queues](algorithms/searching/priority_queue/README.md)**
- **[Time-Forward Processing](algorithms/time_forward_processing/README.md)**
- **[External Memory Primitives](external_memory_primitives/)**: Root package containing:
  - **[External Merge Sort](external_memory_primitives/external_sort.py)**
  - **[Merge Join](external_memory_primitives/merge_join.py)**
- **[External Memory Data Structures](data_structures/)**:
  - **[External Stack](data_structures/stack/README.md)**

## I/O Model and Framework

### External Memory Model

- **Large datasets**: Data structures too large to fit in internal memory.
- **Two-level hierarchy**: Fast internal memory (size M) and slow external storage.
- **Block-based I/O**: Data transferred in blocks of size B between levels.
- **Goal**: Minimize the number of I/O operations to achieve optimal performance.

### Performance Metrics and IOSimulator

The core simulation framework, `IOSimulator`, models this hierarchy:
- **I/O complexity**: Counts the number of block transfers (reads/writes) between memory and storage.
- **Cache Management**: Simulates internal memory using an LRU cache of blocks.
- **Virtual Devices**: Operates on abstract structures (`VirtualDisk`, `VirtualFile`, `VirtualMatrix`) mapping logical addresses to block indexes.

## Project Structure

```
.
├── algorithms/                       # Implementations of I/O efficient algorithms
│   ├── list_ranking/                 # List ranking via independent set reduction
│   ├── searching/                    # External memory search data structures
│   │   ├── btree/                    # Classic external B-tree
│   │   ├── buffer_tree/              # Buffer tree for batch operations
│   │   └── priority_queue/           # External phase-based priority queue
│   ├── sorting/                      # External memory sorting algorithms
│   ├── time_forward_processing/      # Time-forward graph processing framework
│   └── transpose/                    # Matrix transposition algorithms
├── data_structures/                  # Fundamental external memory structures
│   └── stack/                        # External stack with hysteresis buffering
├── external_memory_primitives/       # Root package with external sort & join primitives
└── io_simulator/                     # Core simulation and caching framework
```

## Installation and Usage

### Requirements

- Python 3.7+
- NumPy

### Installing Dependencies

```bash
pip install numpy

# For testing (optional)
pip install pytest
```

### Basic Usage Examples

#### External Memory List Ranking

```python
import random
from io_simulator import VirtualDisk, IOSimulator, VirtualFile
from algorithms.list_ranking.main import generate_random_linked_list
from algorithms.list_ranking.list_ranking import list_ranking

# Generate random linked list of size 100
records, expected_ranks = generate_random_linked_list(100)

# Initialize simulator with VirtualDisk
vd = VirtualDisk(size=10000)
sim = IOSimulator(vd, block_size=9, cache_memory_size=150)

# Write input list as VirtualFile records
vf_in = VirtualFile(sim, 100, record_size=3)
for i, r in enumerate(records):
    vf_in.write_record(i, r)

sim.flush_memory()
sim.io_count = 0

# Run list ranking algorithm
vf_out = list_ranking(sim, vd, vf_in, M=50)
sim.flush_memory()

print(f"Total Block I/O Operations: {sim.io_count}")
```

#### Matrix Transposition

```python
import numpy as np
from io_simulator import IOSimulator, VirtualDisk
from algorithms.transpose import transpose_cache_aware

# Initialize I/O simulator and flat storage representing a 4x4 matrix
vd = VirtualDisk(size=16)
vd.disk = list(np.arange(16))
simulator = IOSimulator(vd, block_size=2, cache_memory_size=8)

# Perform cache-aware transpose
result, io_count = transpose_cache_aware(simulator, 4, 4)
print(f"Transposed with {io_count} I/O operations")
```

#### B-Tree Operations

```python
from io_simulator import IOSimulator, VirtualDisk
from algorithms.searching import BTree

# Setup external storage
vd = VirtualDisk(size=10000)
disk = IOSimulator(vd, block_size=50, cache_memory_size=200)

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
from io_simulator import IOSimulator, VirtualDisk
from algorithms.searching import BufferTree

# Setup larger external storage
vd = VirtualDisk(size=50000)
disk = IOSimulator(vd, block_size=50, cache_memory_size=200)

# Create buffer tree for batch processing
buffer_tree = BufferTree(disk, memory_size=200, block_size=50, degree=8)

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
from io_simulator import IOSimulator, VirtualDisk
from algorithms.searching import ExternalPriorityQueue

# Setup external storage
vd = VirtualDisk(size=50000)
disk = IOSimulator(vd, block_size=50, cache_memory_size=200)

# Create priority queue
pq = ExternalPriorityQueue(disk, memory_size=200, block_size=50, degree=8)

# Insert elements
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
```

#### Time-Forward Processing and Maximal Independent Sets

```python
from io_simulator import IOSimulator, VirtualDisk
from algorithms.time_forward_processing import Graph, MaximalIndependentSetSolver

# Setup external memory
vd = VirtualDisk(size=10000)
disk = IOSimulator(vd, block_size=50, cache_memory_size=200)
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
```

#### External Memory Stack

```python
from io_simulator import IOSimulator, VirtualDisk
from data_structures.stack import ExternalStack

# Setup external storage
vd = VirtualDisk(size=1000)
sim = IOSimulator(vd, block_size=3, cache_memory_size=30)

# Create stack
stack = ExternalStack(sim)

# Push and pop
stack.push(10)
stack.push(20)
val = stack.pop()  # 20
stack.close()
```

## Running Example Demonstrations

You can execute each algorithm implementation file directly to see its demo run:

```bash
# Run External List Ranking simulation
.venv/bin/python -m algorithms.list_ranking.main

# Run Matrix Transpose (Cache-Aware) example
.venv/bin/python algorithms/transpose/cache_aware.py

# Run Matrix Transpose (Cache-Oblivious) example
.venv/bin/python algorithms/transpose/cache_oblivious.py

# Run External Memory Sorting example
.venv/bin/python algorithms/sorting/external_merge_sort.py

# Run Priority Queue example
.venv/bin/python algorithms/searching/priority_queue/priority_queue.py

# Run Maximal Independent Sets example
.venv/bin/python algorithms/time_forward_processing/maximal_independent_sets.py

# Run External Memory Stack example
.venv/bin/python data_structures/stack/examples/stack_example.py

# Run External Merge Sort Primitives example
.venv/bin/python external_memory_primitives/examples/sorting_example.py

# Run Merge Join Primitives example
.venv/bin/python external_memory_primitives/examples/joining_example.py
```

## Testing

To run the complete test suite, use pytest:

```bash
# Run all tests
.venv/bin/pytest

# Run specific test suites
.venv/bin/pytest algorithms/list_ranking/test_list_ranking.py
.venv/bin/pytest algorithms/searching/btree/test_btree.py
.venv/bin/pytest algorithms/searching/buffer_tree/test_buffer_tree.py
.venv/bin/pytest algorithms/searching/priority_queue/test_priority_queue.py
.venv/bin/pytest algorithms/sorting/test_external_merge_sort.py
.venv/bin/pytest algorithms/time_forward_processing/test_maximal_independent_sets.py
.venv/bin/pytest algorithms/transpose/test_transpose_cache_aware.py
.venv/bin/pytest algorithms/transpose/test_transpose_cache_oblivious.py
.venv/bin/pytest io_simulator/test_simulator.py
.venv/bin/pytest data_structures/stack/test_stack.py
.venv/bin/pytest external_memory_primitives/test_external_sort.py
.venv/bin/pytest external_memory_primitives/test_merge_join.py
```

## License

MIT License
