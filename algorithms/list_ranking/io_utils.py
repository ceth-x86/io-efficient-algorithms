import heapq
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from io_simulator import VirtualFile

def external_sort(sim, dm, vf_in, key_index, M):
    """
    Sorts a VirtualFile using External Merge Sort in the simulator.

    Args:
        sim (IOSimulator): The simulator handling I/O operations and caching.
        dm (VirtualDisk): The virtual disk used for temp file allocations.
        vf_in (VirtualFile): The input virtual file containing unsorted records.
        key_index (int): The index in the record to sort by.
        M (int): Maximum number of records allowed in memory.

    Returns:
        VirtualFile: A new virtual file containing the sorted records.
    """
    if vf_in.size == 0:
        return VirtualFile(sim, 0, vf_in.record_size)

    chunks = []
    # Phase 1: Split into sorted runs
    for start_rec in range(0, vf_in.size, M):
        chunk_size = min(M, vf_in.size - start_rec)
        run = []
        for i in range(chunk_size):
            run.append(vf_in.read_record(start_rec + i))
            
        run.sort(key=lambda x: x[key_index])
        
        chunk_vf = VirtualFile(sim, chunk_size, vf_in.record_size)
        for i, rec in enumerate(run):
            chunk_vf.write_record(i, rec)
        chunks.append(chunk_vf)

    # Phase 2: Merge sorted runs
    out_vf = VirtualFile(sim, vf_in.size, vf_in.record_size)
    heap = []
    chunk_ptrs = [0] * len(chunks)

    # Read first record from each chunk
    for i, chunk in enumerate(chunks):
        if chunk.size > 0:
            rec = chunk.read_record(0)
            heapq.heappush(heap, (rec[key_index], i, rec))
            chunk_ptrs[i] = 1

    out_ptr = 0
    while heap:
        key, chunk_idx, rec = heapq.heappop(heap)
        out_vf.write_record(out_ptr, rec)
        out_ptr += 1

        chunk = chunks[chunk_idx]
        ptr = chunk_ptrs[chunk_idx]
        if ptr < chunk.size:
            next_rec = chunk.read_record(ptr)
            heapq.heappush(heap, (next_rec[key_index], chunk_idx, next_rec))
            chunk_ptrs[chunk_idx] += 1

    # Close and free all chunks
    for chunk in chunks:
        chunk.close()

    return out_vf

def merge_join(sim, dm, vf1, key1_index, vf2, key2_index, join_type='inner', default_val=1):
    """
    Performs a Merge Join on two pre-sorted VirtualFiles.
    """
    if join_type == 'left_outer':
        max_size = vf1.size
    else:
        max_size = min(vf1.size, vf2.size)

    # output record size = size of rec1 + size of rec2 (excluding the join key)
    out_record_size = vf1.record_size + vf2.record_size - 1

    # Temporarily allocate a virtual file of maximum possible size
    temp_out_vf = VirtualFile(sim, max_size, out_record_size)

    ptr1 = 0
    ptr2 = 0
    out_count = 0

    while ptr1 < vf1.size:
        rec1 = vf1.read_record(ptr1)
        if ptr2 >= vf2.size:
            if join_type == 'left_outer':
                joined = rec1 + [default_val] * (vf2.record_size - 1)
                temp_out_vf.write_record(out_count, joined)
                out_count += 1
            ptr1 += 1
            continue

        rec2 = vf2.read_record(ptr2)
        k1 = rec1[key1_index]
        k2 = rec2[key2_index]

        if k1 < k2:
            if join_type == 'left_outer':
                joined = rec1 + [default_val] * (vf2.record_size - 1)
                temp_out_vf.write_record(out_count, joined)
                out_count += 1
            ptr1 += 1
        elif k1 > k2:
            ptr2 += 1
        else:
            # Match found
            extra = [val for idx, val in enumerate(rec2) if idx != key2_index]
            joined = rec1 + extra
            temp_out_vf.write_record(out_count, joined)
            out_count += 1
            
            ptr1 += 1
            ptr2 += 1

    # Shrink the file to the exact matched size
    out_vf = VirtualFile(sim, out_count, out_record_size)
    for i in range(out_count):
        rec = temp_out_vf.read_record(i)
        out_vf.write_record(i, rec)

    temp_out_vf.close()
    return out_vf
