# Time-Forward Processing for Graph Algorithms

I/O-efficient implementation of time-forward processing for computing local functions on directed acyclic graphs (DAGs) using external memory priority queues.

## Algorithm Overview

Time-forward processing solves the problem of computing local functions on DAGs in external memory with optimal I/O complexity. The key insight is to use a priority queue to "forward" computed values through time to future vertices in topological order.

### Core Concept

1. **Local Functions on DAGs**: Functions where f(v) depends only on vertex label λ(v) and values f(u) of incoming neighbors
2. **Time-Forward Processing**: Instead of storing incoming values in arrays, use priority queue with vertex index as priority
3. **I/O Efficiency**: Achieves O(sort(V+E)) I/O complexity instead of O(E) for naive approach

### Algorithm Steps

```
For each vertex v in topological order (0, 1, 2, ...):
1. Extract all values from priority queue with priority = v
2. Compute f(v) using local function and extracted values  
3. For each outgoing neighbor u of v:
   - Insert (u, f(v)) into priority queue
```

## Applications

### Maximal Independent Sets

The main application implemented is computing maximal independent sets on undirected graphs:

1. **Graph Conversion**: Convert undirected graph to DAG by directing edges from lower to higher vertex indices
2. **Local Function**: f(v) = 1 if vertex is in independent set, 0 otherwise
   - If no incoming neighbors: f(v) = 1
   - If any incoming neighbor has f = 1: f(v) = 0  
   - If all incoming neighbors have f = 0: f(v) = 1
3. **Result**: Vertices with f(v) = 1 form maximal independent set

## Complexity Analysis

- **I/O Complexity**: O(sort(V+E)) = O((V+E)/B × log_{M/B}((V+E)/B))
- **Time Complexity**: O(V+E) operations on priority queue
- **Space Complexity**: O(M) internal memory

This is optimal for the sorting-based model and much better than the naive O(E) I/O approach.

## Implementation Classes

### Graph
Represents undirected graphs with conversion to DAG representation:
```python
graph = Graph(num_vertices)
graph.add_edge(u, v)  # Add undirected edge
outgoing = graph.get_outgoing_neighbors(v)  # DAG outgoing edges
incoming = graph.get_incoming_neighbors(v)  # DAG incoming edges
```

### TimeForwardProcessor  
Core time-forward processing engine:
```python
processor = TimeForwardProcessor(disk, memory_size, block_size)
results = processor.process_dag(graph, local_function)
```

### MaximalIndependentSetSolver
Specialized solver for maximal independent sets:
```python
solver = MaximalIndependentSetSolver(disk, memory_size, block_size)
independent_set, vertex_values = solver.solve(graph)
```

## Usage Examples

### Basic Maximal Independent Set

```python
import numpy as np
from io_simulator import IOSimulator
from algorithms.time_forward_processing import Graph, MaximalIndependentSetSolver

# Setup external memory
disk_data = np.zeros(10000)
disk = IOSimulator(disk_data, block_size=50, memory_size=200)
solver = MaximalIndependentSetSolver(disk)

# Create graph (path: 0-1-2-3-4)
graph = Graph(5)
for i in range(4):
    graph.add_edge(i, i + 1)

# Compute maximal independent set
independent_set, values = solver.solve(graph)
print(f"Independent set: {sorted(independent_set)}")  # [0, 2, 4]

# Verify correctness
assert solver.verify_independence(graph, independent_set)
assert solver.verify_maximality(graph, independent_set)
```

### Custom Local Function Processing

```python
from algorithms.time_forward_processing import TimeForwardProcessor

processor = TimeForwardProcessor(disk)

# Define custom local function
def my_function(vertex, label, incoming_values):
    return sum(incoming_values) + 1

# Process DAG
results = processor.process_dag(graph, my_function)
```

### I/O Efficiency Demonstration

```python
# Large cycle graph
n = 100
graph = Graph(n)
for i in range(n):
    graph.add_edge(i, (i + 1) % n)

initial_io = solver.get_io_count()
independent_set, values = solver.solve(graph)
io_used = solver.get_io_count() - initial_io

print(f"I/O operations: {io_used}")
print(f"I/O per vertex+edge: {io_used / (2*n):.3f}")
# Should be much less than 1.0 due to batching efficiency
```

## Graph Types and Expected Results

### Path Graph (0-1-2-...-n)
- **Independent Set Size**: ⌈n/2⌉ 
- **Pattern**: Alternating vertices {0, 2, 4, ...}

### Cycle Graph (0-1-2-...-n-0)
- **Independent Set Size**: ⌊n/2⌋
- **Pattern**: Alternating vertices with gap

### Complete Graph K_n
- **Independent Set Size**: 1
- **Pattern**: Single vertex (lowest index)

### Star Graph (center connected to leaves)
- **Independent Set Size**: 1 or (n-1)
- **Pattern**: Either center alone or all leaves

### Tree/Forest
- **Independent Set Size**: Variable
- **Pattern**: Greedy selection based on index ordering

## Performance Characteristics

### I/O Efficiency
The algorithm achieves optimal I/O complexity by:
1. **Batched Processing**: Priority queue operations are batched efficiently
2. **Sequential Access**: Vertices processed in topological order
3. **Minimal Random Access**: Only priority queue operations, no direct graph traversal

### Memory Usage
- **External Storage**: Graph adjacency lists
- **Internal Memory**: Priority queue (M/4 elements per phase)
- **Working Set**: O(M) bounded

### Scalability
Tested on graphs with:
- **Vertices**: Up to 1000+ vertices
- **Edges**: Up to 10000+ edges  
- **I/O Efficiency**: Maintains <1.0 I/O operations per vertex+edge

## Implementation Notes

### Design Decisions
- **Vertex Ordering**: Natural integer ordering (0, 1, 2, ...)
- **Edge Direction**: Lower index → higher index for DAG conversion
- **Priority Queue**: External memory implementation with phase-based processing
- **Error Handling**: Boundary checks for vertex indices and edge validation

### Limitations
- **Graph Size**: Limited by available external storage
- **Memory Requirements**: Requires sufficient internal memory for priority queue phases
- **Ordering Dependence**: Results depend on vertex ordering (different orderings give different valid maximal independent sets)

### Extensions
The time-forward processing framework can be extended to other graph algorithms:
- **Shortest Paths**: Single-source shortest paths on DAGs
- **Dynamic Programming**: DP problems with DAG structure  
- **Expression Evaluation**: Arithmetic/boolean expression trees
- **Dependency Resolution**: Task scheduling with dependencies

## Testing

Comprehensive test suite covers:
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end algorithm verification
- **Property Tests**: Independence and maximality verification
- **Performance Tests**: I/O efficiency measurement
- **Edge Cases**: Empty graphs, single vertices, complete graphs

```bash
# Run tests
python -m pytest tests/test_maximal_independent_sets.py -v

# Run example
python algorithms/time_forward_processing/maximal_independent_sets.py
```

## References

This implementation is based on the theoretical foundations of:
- **External Memory Algorithms**: I/O-efficient graph processing
- **Time-Forward Processing**: Priority queue-based DAG computation
- **Maximal Independent Sets**: Greedy algorithms with ordering dependencies

The key innovation is applying external memory priority queues to achieve optimal I/O complexity for local function computation on graphs.