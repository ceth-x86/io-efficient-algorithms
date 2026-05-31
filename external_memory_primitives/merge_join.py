import sys
from pathlib import Path

# Add project root to sys.path to allow importing from io_simulator
sys.path.append(str(Path(__file__).parent.parent))

from io_simulator import VirtualFile


def merge_join(sim, vf1, key1_index, vf2, key2_index, join_type='inner', default_val=1):
    """
    Performs a Merge Join on two pre-sorted VirtualFiles.

    This function implements a simplified external memory merge join algorithm.
    It reads both files sequentially, matching records with equal keys.

    Assumptions & Limitations:
    1. Pre-sorted Inputs: Both input files (vf1 and vf2) must be pre-sorted by
       their respective join keys before calling this function.
    2. Unique Join Keys: This implementation assumes unique keys in both files
       (or pairwise 1-to-1 mapping), advancing both pointers on a match. It does
       not handle general many-to-many matches which would require backtracking
       or buffering matching groups in memory.
    3. Memory Model: It utilizes VirtualFile reads and writes, which are handled
       block-by-block under the hood by the IOSimulator's LRU cache.

    Args:
        sim (IOSimulator): The simulator handling I/O operations and caching.
        vf1 (VirtualFile): The first (left) pre-sorted input file.
        key1_index (int): The record field index of the join key in vf1.
        vf2 (VirtualFile): The second (right) pre-sorted input file.
        key2_index (int): The record field index of the join key in vf2.
        join_type (str): Either 'inner' or 'left_outer'.
        default_val (any): Value used to pad right-side fields in a left_outer join
                           when no matching record is found in vf2.

    Returns:
        VirtualFile: A new virtual file containing the joined records.
    """
    # -------------------------------------------------------------------------
    # Step 1: Calculate output file dimensions
    # -------------------------------------------------------------------------
    # Determine the maximum possible number of records in the join output.
    # For a Left Outer join, every record from vf1 is preserved, so maximum size is vf1.size.
    # For an Inner join (under 1-to-1 assumption), we can have at most min(vf1.size, vf2.size) records.
    if join_type == 'left_outer':
        max_size = vf1.size
    else:
        max_size = min(vf1.size, vf2.size)

    # Output record size is the sum of sizes of both records minus 1,
    # because we discard the duplicate join key from the second file.
    out_record_size = vf1.record_size + vf2.record_size - 1

    # Temporarily allocate a virtual file of maximum possible size.
    # Since virtual files have fixed pre-allocated sizes in the simulator,
    # we will shrink this to the exact size once the join is complete.
    temp_out_vf = VirtualFile(sim, max_size, out_record_size)

    # Initialize pointers for scanning files, and a counter for output records.
    ptr1 = 0
    ptr2 = 0
    out_count = 0

    # -------------------------------------------------------------------------
    # Step 2: Main Merge-Join Loop
    # -------------------------------------------------------------------------
    # We iterate through the left file (vf1). For each record, we try to find
    # a matching key in the right file (vf2).
    while ptr1 < vf1.size:
        rec1 = vf1.read_record(ptr1)
        
        # Case A: The right file (vf2) has been fully consumed
        if ptr2 >= vf2.size:
            # For Left Outer join, we must still output the left record, padded with default values.
            if join_type == 'left_outer':
                joined = rec1 + [default_val] * (vf2.record_size - 1)
                temp_out_vf.write_record(out_count, joined)
                out_count += 1
            ptr1 += 1
            continue

        rec2 = vf2.read_record(ptr2)
        k1 = rec1[key1_index]
        k2 = rec2[key2_index]

        # Case B: Key in vf1 is smaller than key in vf2
        if k1 < k2:
            # Since vf2 is sorted, all remaining keys in vf2 will be >= k2 > k1.
            # Thus, rec1 has no matching record in vf2.
            if join_type == 'left_outer':
                # Pad with default values for left outer join
                joined = rec1 + [default_val] * (vf2.record_size - 1)
                temp_out_vf.write_record(out_count, joined)
                out_count += 1
            # Advance the left pointer to check the next record in vf1
            ptr1 += 1
            
        # Case C: Key in vf1 is larger than key in vf2
        elif k1 > k2:
            # Since vf1 is sorted, all remaining keys in vf1 will be >= k1 > k2.
            # Thus, rec2 has no matching record in vf1 (we've already passed where it could be).
            # Advance the right pointer to look at the next record in vf2.
            ptr2 += 1
            
        # Case D: Match found (k1 == k2)
        else:
            # Extract all fields of rec2 except the join key itself
            extra = [val for idx, val in enumerate(rec2) if idx != key2_index]
            # Concatenate the entire left record with the non-key fields of the right record
            joined = rec1 + extra
            temp_out_vf.write_record(out_count, joined)
            out_count += 1
            
            # Under the 1-to-1 assumption, advance both pointers
            ptr1 += 1
            ptr2 += 1

    # -------------------------------------------------------------------------
    # Step 3: Shrink output file to exact size
    # -------------------------------------------------------------------------
    # Since the final output size (out_count) may be smaller than the pre-allocated
    # max_size, we create a new file of the exact size and copy all joined records into it.
    out_vf = VirtualFile(sim, out_count, out_record_size)
    for i in range(out_count):
        rec = temp_out_vf.read_record(i)
        out_vf.write_record(i, rec)

    # Clean up the temporary file resources
    temp_out_vf.close()
    return out_vf
