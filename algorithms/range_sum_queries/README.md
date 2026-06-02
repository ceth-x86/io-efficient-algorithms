# Range Sum Queries in External Memory

This module implements algorithms for the **Range Sum Query (RSQ)** problem on a sequence of values $a_1, a_2, \dots, a_n$, optimizing the number of I/O operations according to the Aggarwal-Vitter model.

## Implemented Algorithms

We support three major variants as described in the YSDA lecture on External Memory algorithms:

### 1. Static Online RSQ
Answering query $\text{sum}(i, j) = \sum_{k=i}^j a_k$ on a static array.
* **Naive Version ($O(N)$ disk storage)**:
  - Precomputes prefix sum array $S[i] = \sum_{k=1}^i a_k$ on disk.
  - Resolves each query in exactly 2 read I/Os: $S[j] - S[i-1]$.
* **Block-Grouped Version ($O(N/B)$ disk storage)**:
  - Precomputes block-level prefix sums and keeps them in RAM (takes $N/B$ entries).
  - To answer a query $(i, j)$:
    * Fully covered intermediate blocks are summed using the RAM prefix sums (0 I/O).
    * Partially covered boundary blocks (at the start index $i$ and end index $j$) are fetched from the disk.
    * Since elements within each block are cached, reading them costs at most 2 Read I/Os.

### 2. Static Offline RSQ
Answering a batch of $K$ queries of the form $(query\_id, i, j)$ without maintaining a persistent index on disk.
* **Algorithm**:
  1. Scan the input file to compute prefix sums $S$ on disk.
  2. Sort queries by the right boundary $j$, and merge-join them with $S$ to append $S[j]$ to each query record.
  3. Map records to use index $i-1$, sort them by $i-1$, and left-outer join with $S$ (defaulting to 0 when $i-1 = -1$) to append $S[i-1]$.
  4. Compute $S[j] - S[i-1]$ for each query and sort back by `query_id`.
* **Complexity**: $O(\text{sort}(N + K))$ I/O operations.

### 3. Dynamic Offline RSQ
Answering queries mixed with updates (relative updates $a_i += x$ or absolute updates $a_i = x$).
* **Absolute to Relative Conversion**:
  - Filter and sort all update operations by $(index, time\_id)$.
  - Track current absolute value sequentially per index, rewriting absolute updates $a_i = x$ as relative updates $a_i += x - a_{prev}$.
* **Segment Tree Event Decomposition**:
  - A virtual segment tree is built over the array.
  - Each relative update $a_i += x$ is decomposed into $O(\log N)$ updates on all ancestor nodes.
  - Each query $\text{sum}(i, j)$ is decomposed into $O(\log N)$ queries on canonical covering segment tree nodes.
  - Elementary events are written to a temporary file, sorted by $(node\_id, time\_id)$ using a compound key, and scanned sequentially.
  - Subquery answers are written to disk, sorted by `query_id`, and aggregated.
* **Complexity**: $O(\frac{K \log N}{B} \log_{M/B} \frac{K \log N}{B})$ I/O operations.

---

## Complexity Summary

| Variant | Disk Space | Setup Cost (I/O) | Query Cost (I/O) |
| :--- | :--- | :--- | :--- |
| **Static Online (Naive)** | $O(N)$ | $O(N/B)$ | $2$ Read I/Os |
| **Static Online (Block)** | $O(N/B)$ (RAM) | $O(N/B)$ | $\le 2$ Read I/Os |
| **Static Offline** | $O(N + K)$ | — | $O(\text{sort}(N + K))$ (batch) |
| **Dynamic Offline** | $O(K \log N)$ | — | $O(\text{sort}(K \log N))$ (batch) |

---

## Running the Example

An interactive example showing the I/O operations and outputs can be run via:
```bash
.venv/bin/python algorithms/range_sum_queries/examples/range_sum_example.py
```
