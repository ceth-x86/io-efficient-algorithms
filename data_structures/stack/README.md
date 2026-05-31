# External Memory Stack

This module contains a Python implementation of an **External Memory Stack** optimized for the Aggarwal-Vitter I/O model, integrated with the project's **IO Simulator**.

## Core Concept: Hysteresis Buffer

To achieve optimal performance, we must minimize physical disk block reads and writes. A naive stack implementation might write to disk on every `push` or read on every `pop`, resulting in $\Theta(1)$ I/Os per operation.

To achieve $\Theta(1/B)$ amortized I/O complexity (where $B$ is the block size in elements):
1. **RAM Buffer (up to $2B$ elements)**: We keep a buffer of up to $2B$ elements in internal memory (RAM). All `push` and `pop` operations are performed locally in this buffer as long as possible.
2. **Hysteresis on Flush**: When the RAM buffer size reaches $2B$ elements during a `push`, we flush the oldest $B$ elements from the bottom of the buffer as a single block onto the virtual disk, leaving exactly $B$ elements in the RAM buffer.
3. **Hysteresis on Load**: When the RAM buffer becomes completely empty (size 0) during a `pop`, we read the latest block of $B$ elements from the virtual disk into the RAM buffer.

This hysteresis (lag) ensures that after any disk I/O operation (flush or load), the RAM buffer contains exactly $B$ elements. Consequently, at least $B$ subsequent `push` or `pop` operations must occur before the next disk access is needed.

## Performance Analysis
- **Time/IO Complexity**: $O(1/B)$ amortized I/O operations per `push` or `pop`.
- **Disk Space**: Space is allocated dynamically in blocks of size $B$ on the virtual disk using `sim.disk.allocate(B)` and freed immediately when popped using `sim.disk.free(addr, B)`.

## Running the Example

To run the interactive demonstration of the external stack:
```bash
.venv/bin/python data_structures/stack/examples/stack_example.py
```
