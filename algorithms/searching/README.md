# External Memory Search Structures

This directory contains implementations of search data structures optimized for external memory access, where I/O operations are the primary performance bottleneck.

## Implemented Data Structures

- **[B-Trees](btree/)**: Classic balanced trees with O(log₍B₎(n/B)) per operation
- **[Buffer Trees](buffer_tree/)**: Advanced batched processing achieving sorting bound O(n/B·log₍M/B₎(n/M))

## Problem Statement

**External Memory Dictionary Challenge:**
- Implement dictionary operations (search, insert, delete, find_min, find_max) for datasets larger than internal memory
- Traditional binary search trees require O(log₂ n) I/O operations per operation
- Goal: Develop data structures that minimize I/O operations for external memory scenarios

## Algorithm Comparison

| Data Structure | Single Operation | n Operations | Best Use Case |
|---------------|------------------|-------------|---------------|
| **Binary tree** | O(log₂ n) | n·O(log₂ n) | Simple, small datasets |
| **[B-tree](btree/)** | O(log₍B₎ n) | n·O(log₍B₎ n) | Interactive queries, traditional databases |
| **[Buffer tree](buffer_tree/)** | N/A* | O(n/B·log₍M/B₎(n/M)) | Batch processing, data warehouses |

*Buffer trees are designed for batch operation processing, not individual operations.

## Key Concepts

### External Memory Model
- **Memory Hierarchy**: Fast internal memory (size M) ↔ Slow external storage
- **Block Transfer**: Data moved in blocks of size B between levels
- **I/O Cost**: Minimize number of block transfers
- **Parameters**: Typically M ≫ B and n ≫ M

### Performance Metrics
- **I/O complexity**: Number of block transfers between memory and storage
- **Space complexity**: Amount of internal memory used
- **Optimal bounds**: Theoretical lower bounds for specific problems
- **Cache-oblivious**: Algorithms that work optimally without knowing M and B

## When to Use Which Structure

### Choose B-Trees when:
- Individual operations need immediate results
- Working with traditional database workloads
- Logarithmic guarantees per operation are sufficient
- System has limited memory for batching
- Interactive applications requiring predictable response time

### Choose Buffer Trees when:
- Processing many operations together
- I/O efficiency is more important than individual response time
- Working with large datasets in batch mode
- Have sufficient memory for operation buffering
- Can tolerate lazy/deferred execution

## Quick Start

### B-Tree Example
```python
from algorithms.searching import BTree
from io_simulator import IOSimulator
import numpy as np

# Setup
disk = IOSimulator(np.zeros(10000), block_size=50, memory_size=200)
btree = BTree(disk, d_min=3)

# Operations
btree.insert(10)
found = btree.search(10)  # True
```

### Buffer Tree Example  
```python
from algorithms.searching import BufferTree
from io_simulator import IOSimulator
import numpy as np

# Setup
disk = IOSimulator(np.zeros(50000), block_size=50, memory_size=200)
buffer_tree = BufferTree(disk, degree=8)

# Batch operations
buffer_tree.insert(10, "value_10")
buffer_tree.insert(20, "value_20")
buffer_tree.flush_all_operations()  # Process batch

# Check results
result = buffer_tree.search_results.get(10)
```

## Testing and Examples

```bash
# Run tests
make test-btree           # B-tree tests  
make test-buffer-tree     # Buffer tree tests

# Run examples
make example-btree        # B-tree example
make example-buffer-tree  # Buffer tree example

# Run all searching examples
make examples            # All examples including search structures
```

## Integration with IOSimulator

Both implementations integrate seamlessly with the project's IOSimulator framework:

- **Block-Based I/O**: All disk access through simulator's block interface
- **LRU Cache**: Automatic caching of frequently accessed blocks  
- **I/O Tracking**: Precise measurement of I/O operations for performance analysis
- **Memory Management**: Realistic simulation of memory constraints

This allows for accurate performance analysis and comparison of different external memory algorithms under realistic conditions.

## Detailed Documentation

For comprehensive information about each data structure, including implementation details, algorithms, performance analysis, and usage examples, see the individual documentation:

- **[B-Trees Documentation →](btree/README.md)**
- **[Buffer Trees Documentation →](buffer_tree/README.md)**

## References

- **External Memory Model**: Aggarwal, A. & Vitter, J. S. (1988). "The input/output complexity of sorting and related problems"
- **B-trees**: Bayer, R. & McCreight, E. (1972). "Organization and maintenance of large ordered indexes"
- **Buffer Trees**: Arge, L. (1995). "The buffer tree: A new technique for optimal I/O-algorithms"