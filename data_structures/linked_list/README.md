# External Memory Linked List

This module implements a block-grouped **External Memory Linked List** optimized for the Aggarwal-Vitter I/O model and integrated with the project's **IO Simulator**.

## Core Concepts

In a naive external memory linked list, each node is stored in a separate block on disk. Although point updates (`insert`, `remove`, `lookup`) take $O(1)$ I/Os, traversing $K$ elements sequentially costs $O(K)$ I/Os because the nodes are scattered across the disk, causing a cache miss on almost every pointer jump.

This implementation resolves the traversal overhead by **grouping consecutive elements into blocks of variable size**.

### 1. Block Grouping Invariant
Each physical block on disk contains between $B/2$ and $3B/2$ consecutive logical elements of the list (where $B$ is the system block size).
- Storing elements consecutively inside blocks ensures that a sequential traversal of $K$ elements only reads $\approx \lceil K / (B/2) \rceil$ blocks, reducing the I/O complexity to **$O(K/B)$ block I/Os**.

### 2. Balancing (Splitting, Merging, and Redistribution)
To maintain the grouping invariants during modifications:
- **Split**: When an `insert` causes a block's size to exceed $3B/2$, the block is split into two blocks of size $\approx 3B/4$ each.
- **Merge**: When a `remove` causes a block's size to drop below $B/2$, we check its sibling. If their combined size is small ($\le 3B/2$), we merge them into a single block.
- **Redistribute**: If a merge is not possible because the sibling has too many elements, we redistribute elements between the two blocks so both have size $\approx B$.

Since block splits and merges occur at most once every $O(B)$ modifications, the amortized cost of block balancing is **$O(1/B)$ I/Os**.

### 3. Stable Pointers & External Hash Table
Because splits and merges physically relocate elements on disk, direct disk addresses cannot serve as stable pointers.
- We use an external hash table mapping a stable unique `pointer_id` to its physical location:
  `pointer_id -> (block_address, index_within_block)`
- A `lookup(pointer)` queries this hash table and fetches the block, costing $O(1)$ I/O.
- Block splits and merges update the locations of $O(B)$ elements in the hash table, costing $O(B)$ I/Os. Since this happens once every $O(B)$ modifications, the amortized hash table update cost is **$O(1)$ I/O**.

## Complexity Summary

| Operation | Naive Complexity | Block-Grouped Complexity |
| :--- | :--- | :--- |
| **Lookup** | $O(1)$ I/O | $O(1)$ I/O |
| **Insert** | $O(1)$ I/O | $O(1)$ I/O amortized |
| **Remove** | $O(1)$ I/O | $O(1)$ I/O amortized |
| **Traverse** (size $K$) | $O(K)$ I/Os | $O(K/B)$ I/Os |

## Running the Example

To run the interactive demonstration of the external linked list:
```bash
.venv/bin/python data_structures/linked_list/examples/linked_list_example.py
```
