import heapq
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from io_simulator
sys.path.append(str(Path(__file__).parent.parent))

from io_simulator import VirtualFile


def external_sort(sim, vf_in, key_index, M):
    """
    Sorts a VirtualFile using External Merge Sort in the simulator.

    Args:
        sim (IOSimulator): The simulator handling I/O operations and caching.
        vf_in (VirtualFile): The input virtual file containing unsorted records.
        key_index (int): The index in the record to sort by.
        M (int): Maximum number of records allowed in memory.

    Returns:
        VirtualFile: A new virtual file containing the sorted records.
    """
    # Base case: an empty file does not require sorting
    if vf_in.size == 0:
        return VirtualFile(sim, 0, vf_in.record_size)

    # List to store descriptors of temporary files (runs)
    chunks = []
    
    # -------------------------------------------------------------------------
    # PHASE 1: Split the input file into sorted runs (chunks)
    # -------------------------------------------------------------------------
    # We read the input file sequentially in chunks of size M.
    # Each chunk fits completely in internal memory, gets sorted in RAM,
    # and is written back to disk as a separate temporary sorted file.
    for start_rec in range(0, vf_in.size, M):
        # Calculate the size of the current chunk (the last chunk may be smaller than M)
        chunk_size = min(M, vf_in.size - start_rec)
        
        # Read chunk_size records from the virtual disk into memory
        run = []
        for i in range(chunk_size):
            run.append(vf_in.read_record(start_rec + i))
            
        # Sort the chunk in memory by the specified key_index
        run.sort(key=lambda x: x[key_index])
        
        # Create a temporary file on disk to save the sorted run
        chunk_vf = VirtualFile(sim, chunk_size, vf_in.record_size)
        for i, rec in enumerate(run):
            chunk_vf.write_record(i, rec)
            
        # Store the reference to the temporary file for the next phase
        chunks.append(chunk_vf)

    # -------------------------------------------------------------------------
    # PHASE 2: Multi-way Merge of the sorted runs
    # -------------------------------------------------------------------------
    # We now have multiple sorted files (chunk files).
    # We want to merge them into a single sorted output file out_vf.
    #
    # LIMITATION: This is a single-pass merge implementation.
    # It assumes that the number of chunks k = N/M <= M (which means N <= M^2).
    # If N > M^2, the first records from all chunks would not fit in memory simultaneously.
    # To handle N > M^2, a multi-pass merge sort would be required: recursively
    # merging groups of (M-1) chunks into larger runs until the total number of runs
    # is <= M-1.
    #
    # Since we assume N <= M^2, we can perform the merge in a single pass.
    # We keep exactly one record from each chunk in memory using a min-heap.
    
    # Create the final output file of the same size as the input
    out_vf = VirtualFile(sim, vf_in.size, vf_in.record_size)
    
    # Initialize the heap. It will store tuples of the form:
    # (sort_key_value, source_chunk_index, record)
    heap = []
    
    # Array of current read pointers for each chunk
    chunk_ptrs = [0] * len(chunks)

    # Read the first record from each chunk and push it into the heap.
    # This sets up the initial state for the merge.
    for i, chunk in enumerate(chunks):
        if chunk.size > 0:
            rec = chunk.read_record(0)
            # Push sort key, chunk index, and the record tuple
            heapq.heappush(heap, (rec[key_index], i, rec))
            chunk_ptrs[i] = 1

    # Main merge loop
    out_ptr = 0
    while heap:
        # Pop the record with the minimum sorting key from the heap
        key, chunk_idx, rec = heapq.heappop(heap)
        
        # Write the minimum record to the final sorted output file
        out_vf.write_record(out_ptr, rec)
        out_ptr += 1

        # Advance the pointer in the chunk from which we just popped the element,
        # and read its next record into the heap if it has elements left.
        chunk = chunks[chunk_idx]
        ptr = chunk_ptrs[chunk_idx]
        if ptr < chunk.size:
            next_rec = chunk.read_record(ptr)
            heapq.heappush(heap, (next_rec[key_index], chunk_idx, next_rec))
            chunk_ptrs[chunk_idx] += 1

    # Clean up resources: close and release all temporary chunk files
    for chunk in chunks:
        chunk.close()

    return out_vf
