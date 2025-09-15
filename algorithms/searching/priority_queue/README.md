# External Memory Priority Queue

Phase-based external memory priority queue implementation achieving optimal I/O complexity O(n/B·log_{M/B}(n/M)) for n operations.

## Algorithm Overview

The external memory priority queue solves the fundamental problem that `extract-min` operations cannot be delayed or batched like other dictionary operations. The solution uses a clever phase-based approach:

### Key Innovation: Phase-Based Processing

1. **Phases of M/4 Operations**: Operations are organized into phases of M/4 operations each
2. **In-Memory Minimum Set (S*)**: At the start of each phase, load the M/4 smallest elements into memory
3. **Memory Operations**: During an active phase, all `insert` and `extract-min` operations work entirely in memory
4. **Batch Flush**: At phase end, remaining elements are flushed back to the buffer tree

### Why This Works

- **extract-min availability**: Minimum elements are always available in memory during active phases
- **Batching efficiency**: Bulk operations on buffer tree achieve O(n/B·log_{M/B}(n/M)) sorting bound
- **Amortized analysis**: The I/O cost of loading M/4 elements is amortized over M/4 operations

## Complexity Analysis

- **Per-operation I/O**: O(1/B·log_{M/B}(n/M)) amortized
- **Total I/O for n operations**: O(n/B·log_{M/B}(n/M)) - achieves sorting bound
- **Memory usage**: O(M) internal memory
- **Space on disk**: O(n) elements

## Implementation Details

### Data Structure Components

```python
class ExternalPriorityQueue:
    def __init__(self, disk: IOSimulator, memory_size: int, block_size: int, degree: int):
        self.buffer_tree = BufferTree(...)    # Bulk storage
        self.min_set = []                     # In-memory priority queue (heap)
        self.phase_size = memory_size // 4    # M/4 operations per phase
        self.phase_active = False             # Phase state tracking
```

### Phase Management

- **Phase Start**: Triggered by first `extract-min` when no active phase
  - Extract M/4 minimum elements from buffer tree
  - Load into in-memory heap (`min_set`)
  
- **Phase Operations**: 
  - `insert(priority, value)`: Add to in-memory heap if phase active, otherwise buffer tree
  - `extract_min()`: Pop from in-memory heap, decrement phase counter
  
- **Phase End**: When M/4 operations completed
  - Flush remaining in-memory elements back to buffer tree
  - Reset phase state

### Duplicate Priority Handling

The implementation uses encoded keys to handle duplicate priorities:
- **Encoding**: `priority * scale + sequence_counter`
- **Uniqueness**: Sequence counter ensures all elements get unique buffer tree keys
- **Ordering**: Encoded keys preserve priority ordering with insertion order as tiebreaker

## Usage Examples

### Basic Operations

```python
import numpy as np
from io_simulator import IOSimulator
from algorithms.searching.priority_queue import ExternalPriorityQueue

# Setup external storage
disk_data = np.zeros(50000)
disk = IOSimulator(disk_data, block_size=50, memory_size=200)

# Create priority queue with M/4 = 50 phase size
pq = ExternalPriorityQueue(disk, memory_size=200, block_size=50, degree=8)

# Insert elements
priorities = [10, 5, 15, 3, 7, 12, 18, 1, 4, 6]
for priority in priorities:
    pq.insert(priority, f"value_{priority}")

# Extract minimum elements
while not pq.is_empty():
    min_elem = pq.extract_min()
    if min_elem:
        print(f"Extracted: {min_elem}")
```

### Phase-Based Processing Example

```python
# Demonstrate phase efficiency
initial_io = pq.get_io_count()

# Batch insert (no active phase - goes to buffer tree)
for i in range(20):
    pq.insert(i, f"item_{i}")

# First extract starts a phase - loads minimum elements
first = pq.extract_min()  # Phase starts here, loads M/4 elements
print(f"First extract: {first}")

# Subsequent extracts work from memory
for _ in range(10):
    elem = pq.extract_min()
    print(f"From memory: {elem}")

total_io = pq.get_io_count() - initial_io
print(f"Total I/O for 20 inserts + 11 extracts: {total_io}")
print(f"Amortized I/O per operation: {total_io / 31:.3f}")
```

### Mixed Operations

```python
# Mixed insert/extract during active phases
pq.insert(100, "high")
pq.insert(1, "low")

# Start phase
first = pq.extract_min()  # Gets minimum, starts phase

# Insert during active phase (goes to in-memory set)
pq.insert(50, "medium")

# Extract from in-memory set
second = pq.extract_min()  # Next minimum from memory
```

## Performance Characteristics

### I/O Efficiency

The priority queue achieves the optimal sorting bound through:

1. **Batched Buffer Tree Operations**: Each phase start involves one bulk extraction
2. **Memory Operations**: M/4 operations per phase work entirely in memory  
3. **Amortized Cost**: Bulk I/O cost spread over multiple operations

### Comparison with B-Trees

| Operation | B-Tree | Priority Queue |
|-----------|---------|----------------|
| Single insert | O(log_B n) | O(1/B·log_{M/B}(n/M)) amortized |
| Single extract-min | O(log_B n) | O(1/B·log_{M/B}(n/M)) amortized |
| n operations | O(n log_B n) | O(n/B·log_{M/B}(n/M)) |

The priority queue achieves better asymptotic complexity for batch workloads.

## Implementation Notes

### Strengths
- **Optimal I/O complexity** for batch operations
- **Handles duplicate priorities** correctly
- **Phase-based design** provides predictable performance
- **Memory efficient** with bounded working set

### Limitations
- **Phase boundaries** can cause minor ordering variations in mixed workloads
- **Not optimal** for single operation workloads (use B-tree instead)
- **Requires external storage** setup and management

### Best Use Cases
- **Batch processing** workflows
- **External sorting** as a subroutine
- **Large-scale priority queues** that don't fit in memory
- **Scientific computing** with predictable access patterns

## Testing

The implementation includes comprehensive tests:

```bash
# Run priority queue tests
make test-priority-queue

# Run priority queue example
make example-priority-queue

# Performance testing
python -c "
from algorithms.searching.priority_queue import ExternalPriorityQueue
# ... test large datasets
"
```

## Algorithm References

This implementation is based on the theoretical foundation of external memory priority queues that achieve optimal sorting bounds through phase-based processing and in-memory minimum set management.

The key insight is that by organizing operations into phases and maintaining minimum elements in memory, we can provide immediate access to the smallest elements while still achieving optimal I/O complexity through batched buffer tree operations.