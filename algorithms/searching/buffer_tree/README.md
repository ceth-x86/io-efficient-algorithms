# Buffer Trees for External Memory

Advanced data structure for batched external memory operations that achieves the optimal sorting bound through intelligent operation buffering and batch processing.

## Problem Statement

**Limitations of B-trees:**
- For one operation (search, insert, delete) B-tree requires Θ(log₍B₎(n/B)) I/O
- If need to execute n operations, total time will be n·log₍B₎(n/B)
- This is worse than sorting bound, which equals (n/B)log₍M/B₎(n/B)
- To achieve sorting bound, need to spend less than 1 I/O per operation on average

**Buffer tree solution:** If can't afford ≥1 I/O per each operation, need to **accumulate (buffer)** them and execute in batches.

## Overview

Buffer trees overcome the limitations of B-trees by processing operations in batches rather than individually, achieving O(n/B·log₍M/B₎(n/M)) I/O complexity for n operations - the theoretical optimum for sequence-based operations.

## Key Innovation

**Problem with B-trees**: For n operations, B-trees require n·O(log₍B₎(n/B)) I/O, which exceeds the sorting bound.

**Buffer tree solution**: Accumulate operations in buffers and process them in batches, amortizing I/O cost across multiple operations.

## Core Features

- **Very High Degree**: Θ(M/B) children per internal node
- **Operation Buffering**: Each internal node buffers Θ(M/B) operations
- **Batch Processing**: Operations processed together to amortize I/O
- **Sorting Bound**: Achieves O(n/B·log₍M/B₎(n/M)) for n operations

## Algorithm Mechanism

### 1. Collection Phase
```
Operations → Collection Buffer (size B) → Root Buffer
```
Operations accumulate in memory until buffer fills.

### 2. Distribution Phase  
```
Root Buffer → Sort → Distribute to Children → Child Buffers
```
When buffers overflow, operations are sorted by key and distributed.

### 3. Batch Processing
```
Multiple Operations → Single I/O Burst → Amortized Cost
```
Process M/B operations together, paying only 1/B I/O per operation.

### 4. Lazy Evaluation
```
Operations → Push Down Tree → Process at Leaves
```
Operations flow down tree levels until reaching actual data.

## Implementation

**File**: `buffer_tree.py`

**Node Structure**:
```python
class BufferTreeNode:
    keys: List[Any]              # Routing keys for child selection
    children: List[int]          # Child node IDs (Θ(M/B) children)
    buffer: List[Operation]      # Operation buffer (size Θ(M/B))
    data: List[Tuple]           # Actual data (leaves only)
    is_leaf: bool               # Node type indicator
```

**Operation Types**:
```python
class Operation:
    op_type: OperationType      # SEARCH, INSERT, DELETE
    key: Any                    # Operation key
    value: Any                  # Value (for INSERT operations)
```

## Operations

### Batched Operations
```python
# Operations are buffered, not executed immediately
buffer_tree.insert(key, value)
buffer_tree.search(key) 
buffer_tree.delete(key)

# Process all accumulated operations
buffer_tree.flush_all_operations()

# Check search results
result = buffer_tree.search_results.get(key)
```

### Flush Algorithm
1. **Read** all buffered operations (M/B I/O operations)
2. **Sort** operations by key for efficient distribution  
3. **Distribute** to appropriate children based on routing keys
4. **Recursively flush** children when their buffers overflow
5. **Process at leaves** when flush reaches bottom level

## Performance Analysis

### Theoretical Complexity
- **Tree height**: O(log₍M/B₎(n/M)) levels (much shorter than B-trees)
- **n operations**: O(n/B·log₍M/B₎(n/M)) total I/O
- **Amortized per operation**: O((1/B)·log₍M/B₎(n/M)) I/O

### Comparison with B-trees
| Structure | Single Op | n Operations | Advantage |
|-----------|-----------|-------------|-----------|
| B-tree | O(log₍B₎(n/B)) | n·O(log₍B₎(n/B)) | Individual ops |
| Buffer tree | N/A* | O(n/B·log₍M/B₎(n/M)) | Batch efficiency |

*Buffer trees are designed for batch processing, not individual operations.

### Example Performance
For n=1M keys, B=100, M=10,000:
- **B-tree**: ~3 I/O per operation × 1M = 3M I/O
- **Buffer tree**: ~0.1 I/O per operation × 1M = 100K I/O
- **Improvement**: 30× fewer I/O operations

## Usage Example

```python
import numpy as np
from io_simulator import IOSimulator  
from algorithms.searching.buffer_tree import BufferTree

# Create external storage (larger for buffer trees)
disk_data = np.zeros(50000)
disk = IOSimulator(disk_data, block_size=50, memory_size=200)

# Create buffer tree with high degree (Θ(M/B))
buffer_tree = BufferTree(disk, memory_size=200, block_size=50, degree=8)

# Batch operations - accumulated in memory
keys = [10, 5, 15, 3, 7, 12, 18, 1, 4, 6, 8, 11, 13, 16, 20]

# Operations are buffered, not executed immediately
for key in keys:
    buffer_tree.insert(key, f"value_{key}")

# Process all accumulated operations  
buffer_tree.flush_all_operations()

# Batch searches
search_keys = [1, 5, 10, 15, 20, 25]
for key in search_keys:
    buffer_tree.search(key)

# Process search operations
buffer_tree.flush_all_operations()

# Check results
for key in search_keys:
    result = buffer_tree.search_results.get(key)
    print(f"Key {key}: {'Found' if result else 'Not found'}")

# Performance metrics
total_ops = len(keys) + len(search_keys)
total_io = buffer_tree.get_io_count()
print(f"Amortized I/O per operation: {total_io / total_ops:.3f}")
```

## Performance Results

Our implementation demonstrates the batching advantage:

```
Testing Buffer Tree implementation...
Inserting 20 keys...
I/O operations for batch insertions: 1
Searching for 6 keys...
I/O operations for batch searches: 0
Total I/O operations: 1
Amortized I/O per operation: 0.05
```

**Key insight**: 26 operations completed with only 1 I/O operation!

## Integration

- **IOSimulator**: Block-based I/O simulation with realistic constraints
- **Batch Processing**: Operations accumulated and processed together
- **Memory Management**: Respects M/B parameter relationships
- **I/O Tracking**: Precise measurement of batching efficiency

## Testing

Comprehensive test suite validates:
- ✅ Correctness of batched operations
- ✅ Buffer overflow handling and flush operations
- ✅ I/O efficiency scaling with dataset size
- ✅ Mixed operation scenarios (insert/search/delete)
- ✅ Different node degrees and configurations
- ✅ Stress testing with random operation sequences

```bash
# Run buffer tree tests
make test-buffer-tree

# Run buffer tree example  
make example-buffer-tree
```

## Use Cases

**Ideal for**:
- Bulk data loading and batch updates
- Data warehouse and ETL operations
- Large-scale data processing pipelines
- Scenarios where I/O efficiency is critical
- Applications that can defer operation results

**Choose buffer trees when**:
- Processing many operations together
- I/O efficiency is more important than individual response time
- Working with large datasets in batch mode
- Have sufficient memory for operation buffering
- Can tolerate lazy/deferred execution

**Avoid buffer trees when**:
- Need immediate results for individual operations
- Interactive applications requiring real-time responses
- Memory is severely constrained
- Working with small numbers of operations

## Theoretical Background

### Sorting Bound
The sorting bound Ω(n/B·log₍M/B₎(n/M)) represents the theoretical minimum I/O complexity for processing n operations in external memory. Buffer trees achieve this bound through:

1. **High fan-out**: Θ(M/B) children reduce tree height
2. **Operation batching**: Amortize I/O across multiple operations  
3. **Lazy processing**: Defer work until batches are full
4. **Optimal scheduling**: Process operations in key-sorted order

### Amortized Analysis
Each operation "pays" for:
- 1/B portion of each flush operation it participates in
- log₍M/B₎(n/M) flush operations (tree height)
- Total: O((1/B)·log₍M/B₎(n/M)) amortized I/O

This achieves the sorting bound and demonstrates the power of batching in external memory algorithms.