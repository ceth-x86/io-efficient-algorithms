# External Memory Primitives

This package contains fundamental low-level primitives upon which almost all efficient external memory algorithms (I/O-efficient algorithms) are constructed.

## Why are these primitives important?

In external memory, random disk access is extremely expensive due to physical latency in block transfers. The standard RAM model (which assumes O(1) random access) is not applicable here. To optimize disk I/O operations, algorithms are built on two key building blocks:

### 1. External Sort (external_sort)
Sorting replaces random data access with sequential scans. Instead of following pointers arbitrarily in memory, external memory algorithms represent relations as flat records, sort them by key fields, and process them linearly (scanning in a single pass over the disk with an I/O complexity of O(N/B)).

The `external_sort` implementation is an External Merge Sort algorithm. It splits the input file into sorted runs that fit within internal memory of size M, and then performs a multi-way merge on these runs using a min-heap (heapq).

### 2. Merge Join (merge_join)
A merge join in external memory replaces pointer dereferencing or lookup table operations.
When we need to match related data (e.g., a node and its parent, or a node and its successor in a linked list), following address pointers directly on disk would result in random disk reads.
Instead, `merge_join` accepts two pre-sorted files and performs a linear merge join over them in a single sequential pass over the disk, allowing millions of links to be resolved simultaneously.

---

## Package Structure

- `external_sort.py` — External Merge Sort algorithm implementation.
- `merge_join.py` — Merge Join algorithm implementation (supports both `inner` and `left_outer` join types).
- `test_external_sort.py` — Unit tests for the external sort primitive.
- `test_merge_join.py` — Unit tests for the merge join primitive.
