import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from io_simulator import VirtualFile
from external_memory_primitives.external_sort import external_sort

# Operation types in the input file
OP_REL_UPDATE = 0
OP_QUERY = 1
OP_ABS_UPDATE = 2

# Event types for the segment tree processing
EVENT_UPDATE = 0
EVENT_QUERY = 1

# Large constant for compound sorting keys (secondary sorting)
KEY_MULTIPLIER = 1_000_000_000_000

def _decompose_update(idx: int, val: int, time_id: int, N: int, events_list: list):
    """
    Decomposes a single index update into O(log N) node-level update events
    from the root to the leaf of the segment tree.
    """
    u = 1
    L = 0
    R = N - 1
    
    while L <= R:
        # Generate update event for the current node u
        compound_key = u * KEY_MULTIPLIER + time_id
        events_list.append([u, time_id, EVENT_UPDATE, val, 0, compound_key])
        
        if L == R:
            break
            
        M = (L + R) // 2
        if idx <= M:
            u = 2 * u
            R = M
        else:
            u = 2 * u + 1
            L = M + 1

def _decompose_query(u: int, L: int, R: int, q_i: int, q_j: int, time_id: int, query_id: int, events_list: list):
    """
    Recursively decomposes a query range [q_i, q_j] into canonical covered segment tree nodes.
    """
    if q_i <= L and R <= q_j:
        compound_key = u * KEY_MULTIPLIER + time_id
        events_list.append([u, time_id, EVENT_QUERY, 0, query_id, compound_key])
        return
        
    M = (L + R) // 2
    if q_i <= M:
        _decompose_query(2 * u, L, M, q_i, q_j, time_id, query_id, events_list)
    if q_j > M:
        _decompose_query(2 * u + 1, M + 1, R, q_i, q_j, time_id, query_id, events_list)

def dynamic_offline_rsq(sim, N: int, ops_vf: VirtualFile, M: int) -> VirtualFile:
    """
    Dynamic Offline Range Sum Queries with relative and absolute updates.
    
    Args:
        sim (IOSimulator): Simulator handling caching and I/O.
        N (int): Size of the array (0-indexed, 0 to N-1).
        ops_vf (VirtualFile): Chronological operations file. Records: [time_id, op_type, arg1, arg2]
          * OP_REL_UPDATE: [time_id, 0, index, add_val]
          * OP_QUERY: [time_id, 1, i, j]
          * OP_ABS_UPDATE: [time_id, 2, index, new_val]
        M (int): Maximum records allowed in memory for sorting phases.
        
    Returns:
        VirtualFile: Sorted final query results. Records: [query_id, range_sum]
    """
    K = ops_vf.size
    
    # -------------------------------------------------------------------------
    # Phase 1: Convert absolute updates to relative updates if any exist
    # -------------------------------------------------------------------------
    # Step 1.1: Extract updates to a temporary file
    updates_count = 0
    for k in range(K):
        op = ops_vf.read_record(k)
        if op[1] in (OP_REL_UPDATE, OP_ABS_UPDATE):
            updates_count += 1
            
    updates_vf = VirtualFile(sim, updates_count, 5)  # [time_id, op_type, index, val, compound_sort_key]
    u_idx = 0
    for k in range(K):
        op = ops_vf.read_record(k)
        if op[1] in (OP_REL_UPDATE, OP_ABS_UPDATE):
            time_id, op_type, idx, val = op[0], op[1], op[2], op[3]
            compound_key = idx * KEY_MULTIPLIER + time_id
            updates_vf.write_record(u_idx, [time_id, op_type, idx, val, compound_key])
            u_idx += 1
            
    # Sort updates by (index, time_id)
    sorted_updates = external_sort(sim, updates_vf, key_index=4, M=M)
    updates_vf.close()
    
    # Step 1.2: Scan and convert absolute updates to relative updates
    converted_updates_vf = VirtualFile(sim, updates_count, 4)  # [time_id, OP_REL_UPDATE, index, rel_val]
    prev_idx = -1
    curr_abs_val = 0
    for u in range(updates_count):
        rec = sorted_updates.read_record(u)
        time_id, op_type, idx, val = rec[0], rec[1], rec[2], rec[3]
        if idx != prev_idx:
            prev_idx = idx
            curr_abs_val = 0
            
        if op_type == OP_REL_UPDATE:
            rel_val = val
            curr_abs_val += val
        else:
            rel_val = val - curr_abs_val
            curr_abs_val = val
            
        converted_updates_vf.write_record(u, [time_id, OP_REL_UPDATE, idx, rel_val])
    sorted_updates.close()
    
    # Step 1.3: Merge queries and converted relative updates back into chronological order
    queries_count = K - updates_count
    queries_vf = VirtualFile(sim, queries_count, 4)
    q_idx = 0
    for k in range(K):
        op = ops_vf.read_record(k)
        if op[1] == OP_QUERY:
            queries_vf.write_record(q_idx, op)
            q_idx += 1
            
    # Chronological merge-sort of queries and relative updates (since both are sorted by time_id)
    ops_converted_vf = VirtualFile(sim, K, 4)  # [time_id, op_type, arg1, arg2]
    ptr_u = 0
    ptr_q = 0
    for k in range(K):
        rec_u = converted_updates_vf.read_record(ptr_u) if ptr_u < updates_count else None
        rec_q = queries_vf.read_record(ptr_q) if ptr_q < queries_count else None
        
        if rec_u is not None and (rec_q is None or rec_u[0] < rec_q[0]):
            ops_converted_vf.write_record(k, rec_u)
            ptr_u += 1
        else:
            ops_converted_vf.write_record(k, rec_q)
            ptr_q += 1
            
    converted_updates_vf.close()
    queries_vf.close()
    
    # -------------------------------------------------------------------------
    # Phase 2: Generate elementary node events for virtual segment tree
    # -------------------------------------------------------------------------
    elem_events = []
    query_id_map = {}  # Map time_id of queries to sequential query ids
    next_query_id = 0
    
    for k in range(K):
        op = ops_converted_vf.read_record(k)
        time_id, op_type, arg1, arg2 = op[0], op[1], op[2], op[3]
        if op_type == OP_REL_UPDATE:
            _decompose_update(arg1, arg2, time_id, N, elem_events)
        elif op_type == OP_QUERY:
            query_id_map[time_id] = next_query_id
            _decompose_query(1, 0, N - 1, arg1, arg2, time_id, next_query_id, elem_events)
            next_query_id += 1
            
    ops_converted_vf.close()
    
    # Write elementary events to file
    total_events = len(elem_events)
    events_vf = VirtualFile(sim, total_events, 6)  # [node_id, time_id, event_type, val, query_id, compound_key]
    for i, event in enumerate(elem_events):
        events_vf.write_record(i, event)
        
    # Sort events by (node_id, time_id) using compound_key at index 5
    sorted_events = external_sort(sim, events_vf, key_index=5, M=M)
    events_vf.close()
    
    # -------------------------------------------------------------------------
    # Phase 3: Scan sorted events and compute subquery answers
    # -------------------------------------------------------------------------
    # Count how many elementary query events we have
    subqueries_count = 0
    for idx in range(total_events):
        if sorted_events.read_record(idx)[2] == EVENT_QUERY:
            subqueries_count += 1
            
    subquery_results = VirtualFile(sim, subqueries_count, 2)  # [query_id, subquery_sum]
    
    curr_node = -1
    curr_sum = 0
    sub_idx = 0
    
    for idx in range(total_events):
        rec = sorted_events.read_record(idx)
        node_id, time_id, event_type, val, query_id = rec[0], rec[1], rec[2], rec[3], rec[4]
        
        if node_id != curr_node:
            curr_node = node_id
            curr_sum = 0
            
        if event_type == EVENT_UPDATE:
            curr_sum += val
        elif event_type == EVENT_QUERY:
            subquery_results.write_record(sub_idx, [query_id, curr_sum])
            sub_idx += 1
            
    sorted_events.close()
    
    # -------------------------------------------------------------------------
    # Phase 4: Aggregate subquery results by query_id
    # -------------------------------------------------------------------------
    # Sort subqueries by query_id (index 0)
    sorted_subqueries = external_sort(sim, subquery_results, key_index=0, M=M)
    subquery_results.close()
    
    # Sum subquery values to get total query sums
    final_results = VirtualFile(sim, queries_count, 2)  # [query_id, total_sum]
    
    if queries_count > 0:
        prev_q_id = -1
        q_sum = 0
        final_idx = 0
        
        for idx in range(subqueries_count):
            rec = sorted_subqueries.read_record(idx)
            q_id, val = rec[0], rec[1]
            
            if q_id != prev_q_id:
                if prev_q_id != -1:
                    final_results.write_record(final_idx, [prev_q_id, q_sum])
                    final_idx += 1
                prev_q_id = q_id
                q_sum = val
            else:
                q_sum += val
                
        # Write last query
        if prev_q_id != -1:
            final_results.write_record(final_idx, [prev_q_id, q_sum])
            
    sorted_subqueries.close()
    return final_results
