# I/O Simulator for External Memory Algorithms

The **primary purpose** of this simulator is to measure and analyze the **I/O complexity** (Block I/O Operations) of algorithms designed for external memory. 

In external memory models, CPU computation time is negligible compared to the latency of reading and writing data to disk. Therefore, the efficiency of an algorithm is measured by how many blocks of size $B$ are transferred between the disk and the RAM of size $M$. This simulator acts as a testing bed to count these transfers precisely.

## How I/O Counting Works

The simulator monitors all memory accesses and maintains a strict count of block transfers in `sim.io_count`:
* **Block Reads (+1 I/O)**: Triggered whenever a requested element is in a block that is not currently cached in memory (a cache miss). The simulator loads the entire block of size `block_size` from the disk into the LRU cache.
* **Block Writes (+1 I/O)**: Triggered in two cases:
  1. When a cached block is modified (marked as "dirty") and gets evicted from the cache to make room for a new block (using Least Recently Used policy).
  2. When `flush_memory()` is explicitly called to persist all remaining modified blocks back to the virtual disk.

By tracking `sim.io_count` before and after running an algorithm, you can measure its exact empirical I/O cost and verify if it matches theoretical complexity bounds (such as $O(\frac{N}{B} \log_{M/B} \frac{N}{B})$ for sorting or $O(\log_B N)$ for B-Trees).

---

## Class Overview

The simulator divides responsibilities into five classes:

1. **`VirtualDisk`** (`virtual_disk.py`)
   Simulates physical disk space as a flat array. It contains a memory allocator (`allocate`/`free`) to request continuous regions on the disk for virtual files or B-Tree nodes, and performs raw block reading/writing.

2. **`IOSimulator`** (`io_simulator.py`)
   The central cache manager. It maintains an LRU block cache of size $M$ (max $M/B$ blocks) and increments `io_count` on cache misses and dirty block evictions. All high-level data structures access elements through this class.

3. **`VirtualFile`** (`virtual_file.py`)
   An abstraction for sequential vectors of fixed-size records (tuples). It provides high-level `read_record(idx)` and `write_record(idx, record)` APIs, which internally invoke the simulator's cached element operations.

4. **`VirtualMatrix`** (`virtual_matrix.py`)
   An abstraction for 2D matrices stored on disk in Row-Major layout. It translates 2D coordinates into flat indices and supports submatrix reading/writing.

5. **`DiskBTree`** (`disk_btree.py`)
   An indexing B+ Tree where nodes are serialized to block-sized records on disk. Allows measuring B-Tree search and insertion I/O costs.

---

## Measuring I/O Complexity: Code Examples

### 1. Evaluating Sequential I/O Cache Efficiency
If we read elements sequentially, the simulator will load a block once, and subsequent elements in that block will be read from the cache with 0 I/O cost.

```python
from io_simulator import VirtualDisk, IOSimulator, VirtualFile

vd = VirtualDisk(size=1000)
# Block size B=10, Memory size M=30 (Cache limit = 3 blocks)
sim = IOSimulator(vd, block_size=10, memory_size=30)

# Create file with 10 records (each record is 3 integers = 30 elements total)
vf = VirtualFile(sim, size=10, record_size=3)

# Reset I/O counter
sim.io_count = 0

# Write records sequentially
for i in range(10):
    vf.write_record(i, [i, i*2, i*3])

# Evict/flush cache to disk
sim.flush_memory()

# 30 elements sequentially written with B=10 means 3 blocks are written.
# Total write I/Os = 3.
print(f"I/Os performed during sequential write: {sim.io_count}") # Output: 3
```

### 2. Measuring B+ Tree Lookup I/O Complexity
You can measure the number of block reads needed to find a key in a B+ Tree, which theoretically scales as $O(\log_B N)$.

```python
from io_simulator import VirtualDisk, IOSimulator, DiskBTree

vd = VirtualDisk(size=5000)
sim = IOSimulator(vd, block_size=9, memory_size=90)
btree = DiskBTree(sim)

# Insert keys
for k in range(1, 100):
    btree.insert(key=k, value=k*10)

sim.flush_memory()

# Reset I/O count before search
sim.io_count = 0

# Search key
btree.search(45)

# B+ Tree search traverses nodes from root to leaf. 
# Each node load from disk increments I/O by 1.
print(f"Block I/Os performed during B+ Tree search: {sim.io_count}")
```

---

## Running Unit Tests

To run the unit tests verifying caching, allocations, and operations:

```bash
python3 -m unittest io_simulator.test_simulator
```
