import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from io_simulator import VirtualFile
from external_memory_primitives import external_sort, merge_join

def find_head_external(sim, dm, vf_in, M):
    """
    Finds the head of the linked list in the simulator.

    Args:
        sim (IOSimulator): The simulator handling I/O operations and caching.
        dm (VirtualDisk): The virtual disk used for temp file allocations.
        vf_in (VirtualFile): The input virtual file containing the list nodes.
        M (int): Memory size limit in records.

    Returns:
        int: The ID of the head node of the linked list.
    """
    # To find the head node (the node with no incoming pointer), we perform a left outer join
    # of all nodes against all next pointers.
    # A merge join requires both input streams to be sorted by their respective join keys.
    # Therefore, we sort the file twice:
    # 1. By node ID (index 0) to align the node records.
    # 2. By next ID (index 1) to align the pointer destinations.
    vf_sort_node = external_sort(sim, vf_in, key_index=0, M=M)
    vf_sort_next = external_sort(sim, vf_in, key_index=1, M=M)
    
    # Perform a left outer join of all nodes on node_id (key1_index=0) against next pointers (key2_index=1).
    # Since it is a left outer join, any node that has no incoming pointer (i.e. is not pointed to by any node)
    # will not match any key in vf_sort_next, and will be padded with the default_val (-999).
    vf_join = merge_join(
        sim, 
        vf1=vf_sort_node, 
        key1_index=0, 
        vf2=vf_sort_next, 
        key2_index=1, 
        join_type='left_outer', 
        default_val=-999
    )
    
    # Identify the head of the list:
    # In a valid linked list, the head node is the only node that has no incoming pointers
    # (i.e. no other node's next_id points to it). Consequently, it is the only record
    # in the left outer join result that failed to find a match on the right side,
    # leaving its last field padded with the default value of -999.
    # We scan the joined file sequentially to find this padded record and extract its node ID.
    head_id = None
    for i in range(vf_join.size):
        rec = vf_join.read_record(i)
        if rec and rec[-1] == -999:
            head_id = rec[0]
            break
            
    vf_sort_node.close()
    vf_sort_next.close()
    vf_join.close()
    return head_id

def solve_base_case(sim, dm, vf_in, head_id):
    """
    Solves the list ranking problem in memory when list size <= M.
    """
    records = []
    for i in range(vf_in.size):
        records.append(vf_in.read_record(i))
        
    if not records:
        return VirtualFile(sim, 0, 2)
        
    adj = {}
    for node_id, next_id, weight in records:
        adj[node_id] = (next_id, weight)
        
    head = head_id
    if head not in adj and adj:
        head = list(adj.keys())[0]
        
    ranks = {}
    curr = head
    curr_rank = 0
    while curr != -1 and curr in adj:
        ranks[curr] = curr_rank
        next_id, weight = adj[curr]
        curr_rank += weight
        curr = next_id
        
    vf_out = VirtualFile(sim, vf_in.size, 2)
    for i, node_id in enumerate(adj.keys()):
        r = ranks.get(node_id, 0)
        vf_out.write_record(i, [node_id, r])
        
    return vf_out

def resolve_surviving_links(sim, dm, vf_join):
    vf_out = VirtualFile(sim, vf_join.size, 3)
    for i in range(vf_join.size):
        rec = vf_join.read_record(i)
        node_id, next_id, weight, match_next_id, match_weight = rec
        if match_next_id == -1:
            vf_out.write_record(i, [node_id, next_id, weight])
        else:
            vf_out.write_record(i, [node_id, match_next_id, weight + match_weight])
    return vf_out

def extract_parent_link(sim, dm, vf_join):
    vf_out = VirtualFile(sim, vf_join.size, 3)
    for i in range(vf_join.size):
        rec = vf_join.read_record(i)
        node_id, next_id, weight, match_next_id, match_weight = rec
        vf_out.write_record(i, [node_id, next_id, weight])
    return vf_out

def calculate_restored_ranks(sim, dm, vf_join):
    vf_out = VirtualFile(sim, vf_join.size, 2)
    for i in range(vf_join.size):
        rec = vf_join.read_record(i)
        parent_id, child_id, parent_weight, parent_rank = rec
        vf_out.write_record(i, [child_id, parent_rank + parent_weight])
    return vf_out

def list_ranking_rec(sim, dm, vf_in, M, head_id, depth=0):
    """
    Recursively ranks the linked list using the randomized independent set compression algorithm.
    
    This function implements the optimal external memory list ranking algorithm:
    1. Selects an independent set of nodes to remove (expected size: N/4).
    2. Compresses the list by updating the successor pointers of surviving nodes to skip
       the removed nodes, accumulating the edge weights.
    3. Recursively ranks the compressed list (of size ~3/4 N).
    4. Restores the ranks of the removed nodes using the ranks of their predecessors.
    
    All steps are done using sequential operations (sorting and joining) to match the
    Aggarwal-Vitter external memory complexity bounds, completely avoiding random I/O.
    """
    n = vf_in.size
    
    # -------------------------------------------------------------------------
    # Base Case: The active list size fits in internal memory (N <= M)
    # -------------------------------------------------------------------------
    if n <= M:
        # If it fits in memory, solve it using a standard in-memory pointer-chasing pass
        # which is fast and takes only O(N/B) I/Os.
        return solve_base_case(sim, dm, vf_in, head_id)
        
    while True:
        # ---------------------------------------------------------------------
        # Phase 1: Independent Set Selection
        # ---------------------------------------------------------------------
        # We assign a random bit (0 or 1) to each node.
        # A node is selected for removal if:
        #   - Its bit is 1.
        #   - Its successor's bit is 0.
        # This guarantees that:
        #   1. No two adjacent nodes are removed (independent set property).
        #   2. The head of the list is never removed (since its bit is hardcoded to 0).
        #   3. On average, 1/4 of all nodes are removed.
        
        # Step 1.1: Generate a file of random bits for each node.
        vf_rnd = VirtualFile(sim, n, 2)
        for i in range(n):
            rec = vf_in.read_record(i)
            node_id = rec[0]
            if node_id == head_id:
                bit = 0  # Guarantee the head node is never removed
            else:
                bit = random.choice([0, 1])
            vf_rnd.write_record(i, [node_id, bit])
            
        # Step 1.2: Sort both files by node_id to align nodes with their assigned bits.
        vf_L_sort_node = external_sort(sim, vf_in, key_index=0, M=M)
        vf_Rnd_sort_node = external_sort(sim, vf_rnd, key_index=0, M=M)
        
        # Step 1.3: Inner join to attach my_bit to each node: [node_id, next_id, weight, my_bit]
        vf_L_with_my_bit = merge_join(sim, vf_L_sort_node, 0, vf_Rnd_sort_node, 0, 'inner')
        
        # Step 1.4: Sort the list by next_id to prepare for joining with the successor's bit.
        vf_L_sort_next = external_sort(sim, vf_L_with_my_bit, key_index=1, M=M)
        
        # Step 1.5: Left outer join on next_id = node_id to get the successor's bit (next_bit).
        # If a node points to -1 (end of list), it won't find a match, and gets default next_bit = 1.
        vf_L_flags = merge_join(sim, vf_L_sort_next, 1, vf_Rnd_sort_node, 0, 'left_outer', default_val=1)
        
        # Step 1.6: Split nodes into 'removed' and 'surviving' sets based on the condition.
        temp_rem = VirtualFile(sim, n, 3)
        temp_surv = VirtualFile(sim, n, 3)
        num_rem = 0
        num_surv = 0
        
        for i in range(n):
            rec = vf_L_flags.read_record(i)
            node_id, next_id, weight, my_bit, next_bit = rec
            # Independent set condition: my_bit == 1 and next_bit == 0
            if my_bit == 1 and next_bit == 0:
                temp_rem.write_record(num_rem, [node_id, next_id, weight])
                num_rem += 1
            else:
                temp_surv.write_record(num_surv, [node_id, next_id, weight])
                num_surv += 1
                
        # Clean up temporary resources used during the independent set detection.
        vf_rnd.close()
        vf_L_sort_node.close()
        vf_Rnd_sort_node.close()
        vf_L_with_my_bit.close()
        vf_L_sort_next.close()
        vf_L_flags.close()
        
        # Ensure we successfully selected a non-empty independent set of nodes to remove.
        # Otherwise, retry with a new set of random bits.
        if num_rem > 0:
            # Copy elements to exact-sized files to free up disk space in the simulator.
            vf_removed = VirtualFile(sim, num_rem, 3)
            for i in range(num_rem):
                vf_removed.write_record(i, temp_rem.read_record(i))
                
            vf_surviving = VirtualFile(sim, num_surv, 3)
            for i in range(num_surv):
                vf_surviving.write_record(i, temp_surv.read_record(i))
                
            temp_rem.close()
            temp_surv.close()
            break
            
        temp_rem.close()
        temp_surv.close()
        
    # -------------------------------------------------------------------------
    # Phase 2: Update Surviving Links (List Compression)
    # -------------------------------------------------------------------------
    # We now update the next pointers of surviving nodes that point to removed nodes.
    # If surviving node A points to removed node B, and B points to C, we update
    # A's pointer to point to C directly, and update the edge weight to weight(A->B) + weight(B->C).
    #
    # We do this using a left outer join of surviving nodes (sorted by next_id)
    # against removed nodes (sorted by node_id).
    vf_surviving_sort_next = external_sort(sim, vf_surviving, key_index=1, M=M)
    vf_removed_sort_node = external_sort(sim, vf_removed, key_index=0, M=M)
    
    # Left outer join: match surviving successor (key1=1) against removed node (key2=0).
    vf_surv_rem_join = merge_join(sim, vf_surviving_sort_next, 1, vf_removed_sort_node, 0, 'left_outer', default_val=-1)
    
    # Update pointers and weights for the matches
    vf_L_prime = resolve_surviving_links(sim, dm, vf_surv_rem_join)
    
    vf_surviving_sort_next.close()
    vf_surv_rem_join.close()
    
    # -------------------------------------------------------------------------
    # Phase 3: Recursive Call
    # -------------------------------------------------------------------------
    # Solve the list ranking problem recursively on the compressed list vf_L_prime.
    # Since we removed an expected N/4 nodes, the input size is reduced by a constant fraction.
    # This guarantees the depth of recursion is O(log N) with high probability,
    # and the total recursive cost is dominated by the top-level call: O(Sort(N)).
    vf_ranks_prime = list_ranking_rec(sim, dm, vf_L_prime, M, head_id, depth + 1)
    vf_L_prime.close()
    
    # -------------------------------------------------------------------------
    # Phase 4: Restore Ranks (Reranking)
    # -------------------------------------------------------------------------
    # We must calculate the ranks of the removed nodes.
    # A removed node (child) is always pointed to by a surviving node (parent).
    # Its rank is simply: rank(child) = rank(parent) + weight(parent -> child).
    #
    # Step 4.1: Find parent-child links. We sort surviving nodes by next_id,
    # and inner join them with removed nodes sorted by node_id.
    vf_surviving_sort_next_re = external_sort(sim, vf_surviving, key_index=1, M=M)
    vf_parent_link_join = merge_join(sim, vf_surviving_sort_next_re, 1, vf_removed_sort_node, 0, 'inner')
    
    # Extract the links: [parent_id, child_id, parent_weight]
    vf_parent_link = extract_parent_link(sim, dm, vf_parent_link_join)
    
    vf_surviving_sort_next_re.close()
    vf_removed_sort_node.close()
    vf_parent_link_join.close()
    
    # Step 4.2: To fetch the parent's rank, we sort the parent links by parent_id,
    # and sort the recursively computed ranks file by node_id (parent_id).
    vf_parent_link_sort_parent = external_sort(sim, vf_parent_link, key_index=0, M=M)
    vf_parent_link.close()
    
    vf_ranks_prime_sort = external_sort(sim, vf_ranks_prime, key_index=0, M=M)
    
    # Step 4.3: Inner join parent links with parent ranks:
    # [parent_id, child_id, parent_weight] JOIN [parent_id, parent_rank]
    vf_restored_ranks_join = merge_join(sim, vf_parent_link_sort_parent, 0, vf_ranks_prime_sort, 0, 'inner')
    
    # Calculate child ranks: rank(child) = parent_rank + parent_weight
    vf_restored_ranks = calculate_restored_ranks(sim, dm, vf_restored_ranks_join)
    
    vf_parent_link_sort_parent.close()
    vf_ranks_prime_sort.close()
    vf_restored_ranks_join.close()
    
    # -------------------------------------------------------------------------
    # Phase 5: Merge ranks of surviving and removed nodes
    # -------------------------------------------------------------------------
    # Concatenate the computed ranks of surviving nodes (vf_ranks_prime)
    # and the restored ranks of removed nodes (vf_restored_ranks) into the output file.
    vf_out = VirtualFile(sim, vf_ranks_prime.size + vf_restored_ranks.size, 2)
    for i in range(vf_ranks_prime.size):
        vf_out.write_record(i, vf_ranks_prime.read_record(i))
    for i in range(vf_restored_ranks.size):
        vf_out.write_record(vf_ranks_prime.size + i, vf_restored_ranks.read_record(i))
        
    # Close files and clean up resources
    vf_removed.close()
    vf_surviving.close()
    vf_ranks_prime.close()
    vf_restored_ranks.close()
    
    return vf_out

def list_ranking(sim, dm, vf_in, M):
    """
    Main entry point for External Memory List Ranking in the simulator.
    """
    # Note: find_head_external only finds the single ID of the head node.
    # It does NOT traverse the linked list (traversing sequentially would cost O(N) I/Os due to pointer chasing).
    # Instead, it finds the start point in O(Sort(N)) I/Os, after which list_ranking_rec is used
    # to recursively rank the list in O(Sort(N)) total I/Os.
    with sim.measure_io() as measurement:
        head_id = find_head_external(sim, dm, vf_in, M)
    print(f"I/Os for find_head_external: {measurement.io_count}")
    if head_id is None:
        raise ValueError("Could not find list head!")
        
    vf_temp_ranks = list_ranking_rec(sim, dm, vf_in, M, head_id, 0)
    vf_out = external_sort(sim, vf_temp_ranks, key_index=0, M=M)
    vf_temp_ranks.close()
    return vf_out
