import sys
from pathlib import Path
import math

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

def in_place_partition_sort(sim, start_idx: int, N: int, L: int, R: int, M: int):
    """
    In-place Partition Sort in External Memory.
    
    Theoretical Context:
    - Traditional Merge Sort requires O(N) additional disk space.
    - In-place Partition Sort sorts the array without using any extra disk space.
    - Complexity: O((N/B) log_{M/B} (K/B)) I/Os, where K is the range size R - L + 1.
    
    Algorithm workflow:
    1. Base Case 1: If subarray size N <= M, load the entire subarray into RAM,
       sort it in memory, and write it back to disk.
    2. Base Case 2: If the value range R - L + 1 <= 1, all elements are identical
       and already sorted, so return immediately.
    3. Multiway Partitioning:
       - Divide the value range [L, R] into d = M/B subranges.
       - Scan the subarray to count how many elements fall in each subrange.
       - Compute prefix sums of counts to find the partition boundaries on disk.
       - Load the first block of size B for each active partition into RAM buffers.
       - Permute elements in RAM buffers. If a buffer becomes full of elements belonging
         to its own subrange, flush it to disk and load the next block of that partition.
       - Recursively sort each of the d partitions.
    """
    if N <= 1 or L >= R:
        return
        
    B = sim.block_size
    
    # -------------------------------------------------------------------------
    # Base Case 1: Subarray fits entirely in RAM
    # -------------------------------------------------------------------------
    if N <= M:
        # Load the whole subarray
        data = []
        for i in range(N):
            data.append(sim.read_element(start_idx + i))
            
        data.sort()
        
        # Write back
        for i in range(N):
            sim.write_element(start_idx + i, data[i])
        return

    # -------------------------------------------------------------------------
    # Step 1: Partitioning range [L, R] into d subranges
    # -------------------------------------------------------------------------
    d = max(2, M // B)
    range_len = R - L + 1
    if range_len < d:
        d = range_len
        
    subrange_width = math.ceil(range_len / d)
    
    # Define subranges: [L_i, R_i] for i in 0..d-1
    subranges = []
    for i in range(d):
        sub_L = L + i * subrange_width
        sub_R = min(R, L + (i + 1) * subrange_width - 1)
        subranges.append((sub_L, sub_R))
        
    # Helper to get subrange index for a given value
    def get_subrange_idx(val):
        idx = (val - L) // subrange_width
        return min(d - 1, max(0, idx))

    # -------------------------------------------------------------------------
    # Step 2: Scan and count partition sizes
    # -------------------------------------------------------------------------
    subrange_counts = [0] * d
    for i in range(N):
        val = sim.read_element(start_idx + i)
        idx = get_subrange_idx(val)
        subrange_counts[idx] += 1
        
    # Compute partition start boundaries (logical offsets from start_idx)
    p = [0] * d
    for i in range(1, d):
        p[i] = p[i - 1] + subrange_counts[i - 1]
        
    # -------------------------------------------------------------------------
    # Step 3: Initialize buffers and state
    # -------------------------------------------------------------------------
    write_ptr = [p_i for p_i in p]
    buffer = [[] for _ in range(d)]
    curr_idx = [0] * d
    unsorted_end = [0] * d
    done = [False] * d
    
    def load_next_block(part_idx):
        p_start = p[part_idx]
        p_count = subrange_counts[part_idx]
        w_ptr = write_ptr[part_idx]
        
        if w_ptr >= p_start + p_count:
            done[part_idx] = True
            buffer[part_idx] = []
            curr_idx[part_idx] = 0
            unsorted_end[part_idx] = 0
            return
            
        block_len = min(B, p_start + p_count - w_ptr)
        blk = []
        for i in range(block_len):
            blk.append(sim.read_element(start_idx + w_ptr + i))
            
        buffer[part_idx] = blk
        curr_idx[part_idx] = 0
        unsorted_end[part_idx] = block_len

    def flush_block(part_idx):
        blk = buffer[part_idx]
        w_ptr = write_ptr[part_idx]
        for i in range(len(blk)):
            sim.write_element(start_idx + w_ptr + i, blk[i])

    # Initial load of first block for all non-empty partitions
    for i in range(d):
        if subrange_counts[i] > 0:
            load_next_block(i)
        else:
            done[i] = True

    # -------------------------------------------------------------------------
    # Step 4: In-place Partitioning loop
    # -------------------------------------------------------------------------
    while not all(done):
        # Find first active partition
        i = -1
        for idx in range(d):
            if not done[idx]:
                i = idx
                break
                
        if i == -1:
            break
            
        # Check if the buffer for partition i is exhausted (all elements checked)
        if curr_idx[i] == unsorted_end[i]:
            flush_block(i)
            write_ptr[i] += B
            load_next_block(i)
            continue
            
        x = buffer[i][curr_idx[i]]
        j = get_subrange_idx(x)
        
        if j == i:
            curr_idx[i] += 1
        else:
            # Swap x into partition j
            # First ensure partition j's buffer has room and is loaded
            if curr_idx[j] == unsorted_end[j]:
                flush_block(j)
                write_ptr[j] += B
                load_next_block(j)
                
            # Perform swap in memory
            buffer[i][curr_idx[i]], buffer[j][curr_idx[j]] = buffer[j][curr_idx[j]], buffer[i][curr_idx[i]]
            
            # Since the element swapped into buffer[j] is now correct, advance curr_idx[j]
            curr_idx[j] += 1

    # -------------------------------------------------------------------------
    # Step 5: Recursive call on each partition
    # -------------------------------------------------------------------------
    for i in range(d):
        if subrange_counts[i] > 0:
            sub_L, sub_R = subranges[i]
            in_place_partition_sort(sim, start_idx + p[i], subrange_counts[i], sub_L, sub_R, M)

def in_place_sort(sim, start_idx: int, N: int, K: int, M: int):
    """
    Wrapper for sorting elements in range [1, K] in-place.
    """
    in_place_partition_sort(sim, start_idx, N, 1, K, M)
