import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from io_simulator import VirtualFile
from external_memory_primitives.external_sort import external_sort
from external_memory_primitives.merge_join import merge_join

def static_offline_rsq(sim, array_vf: VirtualFile, queries_vf: VirtualFile, M: int) -> VirtualFile:
    """
    Static Offline Range Sum Queries using External Merge Sort and Merge Join.
    
    Theoretical Context:
    - Answers K queries of the form (query_id, i, j) offline.
    - Does not build a large persistent index on disk.
    - Algorithm workflow:
      1. Compute prefix sums S[0..N-1] in a single scan of the input file.
      2. Sort queries by j, and merge-join with S to obtain S[j] for each query.
      3. Sort intermediate query records by i-1, and left-outer join with S
         (defaulting to 0 when i-1 = -1) to obtain S[i-1].
      4. For each query, calculate the range sum as S[j] - S[i-1].
      5. Sort the results back by query_id to match the input order.
    
    Complexity: O(sort(N + K)) I/O operations.
    """
    N = array_vf.size
    K = queries_vf.size
    
    # -------------------------------------------------------------------------
    # Step 1: Scan input array and build prefix sums file on disk
    # -------------------------------------------------------------------------
    prefix_sums_vf = VirtualFile(sim, N, 2)  # Records: [index, prefix_sum]
    current_sum = 0
    for idx in range(N):
        val = array_vf.read_record(idx)[0]
        current_sum += val
        prefix_sums_vf.write_record(idx, [idx, current_sum])
        
    # -------------------------------------------------------------------------
    # Step 2: Sort queries by j (index 2) and join with prefix_sums_vf
    # -------------------------------------------------------------------------
    # Sort queries by j (query record: [query_id, i, j])
    sorted_queries_by_j = external_sort(sim, queries_vf, key_index=2, M=M)
    
    # Join queries and prefix_sums on query.j == prefix_sums.index
    # Output record: [query_id, i, j, S_j]
    join_j_vf = merge_join(sim, sorted_queries_by_j, 2, prefix_sums_vf, 0, join_type='inner')
    sorted_queries_by_j.close()
    
    # -------------------------------------------------------------------------
    # Step 3: Transform intermediate file to join on i-1
    # -------------------------------------------------------------------------
    # Map [query_id, i, j, S_j] -> [query_id, i-1, S_j]
    transformed_vf = VirtualFile(sim, K, 3)
    for k in range(K):
        rec = join_j_vf.read_record(k)
        query_id = rec[0]
        i = rec[1]
        s_j = rec[3]
        transformed_vf.write_record(k, [query_id, i - 1, s_j])
    join_j_vf.close()
    
    # Sort transformed file by i-1 (index 1)
    sorted_transformed = external_sort(sim, transformed_vf, key_index=1, M=M)
    transformed_vf.close()
    
    # Left-outer join on query.i-1 == prefix_sums.index
    # If query.i-1 == -1, the join will default the prefix_sums value to 0.
    # Output record: [query_id, i-1, S_j, S_i_minus_1]
    join_i_vf = merge_join(sim, sorted_transformed, 1, prefix_sums_vf, 0, join_type='left_outer', default_val=0)
    sorted_transformed.close()
    prefix_sums_vf.close()
    
    # -------------------------------------------------------------------------
    # Step 4: Compute the final range sums S[j] - S[i-1]
    # -------------------------------------------------------------------------
    # Intermediate output: [query_id, sum]
    results_vf = VirtualFile(sim, K, 2)
    for k in range(K):
        rec = join_i_vf.read_record(k)
        query_id = rec[0]
        s_j = rec[2]
        s_i_minus_1 = rec[3]
        results_vf.write_record(k, [query_id, s_j - s_i_minus_1])
    join_i_vf.close()
    
    # -------------------------------------------------------------------------
    # Step 5: Sort final results by query_id to restore original order
    # -------------------------------------------------------------------------
    sorted_results = external_sort(sim, results_vf, key_index=0, M=M)
    results_vf.close()
    
    return sorted_results
