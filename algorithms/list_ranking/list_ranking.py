import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from io_simulator import VirtualFile
if __package__ is None or __package__ == "":
    # Direct execution: e.g. python3 main.py (from within directory)
    from io_utils import external_sort, merge_join
else:
    # Module execution: e.g. python3 -m algorithms.list_ranking.main (from root)
    from .io_utils import external_sort, merge_join

def find_head_external(sim, dm, vf_in, M):
    """
    Finds the head of the linked list in the simulator.
    """
    vf_sort_node = external_sort(sim, dm, vf_in, 0, M)
    vf_sort_next = external_sort(sim, dm, vf_in, 1, M)
    
    vf_join = merge_join(sim, dm, vf_sort_node, 0, vf_sort_next, 1, 'left_outer', default_val=-999)
    
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
    n = vf_in.size
    if n <= M:
        return solve_base_case(sim, dm, vf_in, head_id)
        
    while True:
        # 1. Assign random bits
        vf_rnd = VirtualFile(sim, n, 2)
        for i in range(n):
            rec = vf_in.read_record(i)
            node_id = rec[0]
            if node_id == head_id:
                bit = 0
            else:
                bit = random.choice([0, 1])
            vf_rnd.write_record(i, [node_id, bit])
            
        vf_L_sort_node = external_sort(sim, dm, vf_in, 0, M)
        vf_Rnd_sort_node = external_sort(sim, dm, vf_rnd, 0, M)
        
        vf_L_with_my_bit = merge_join(sim, dm, vf_L_sort_node, 0, vf_Rnd_sort_node, 0, 'inner')
        vf_L_sort_next = external_sort(sim, dm, vf_L_with_my_bit, 1, M)
        
        vf_L_flags = merge_join(sim, dm, vf_L_sort_next, 1, vf_Rnd_sort_node, 0, 'left_outer', default_val=1)
        
        temp_rem = VirtualFile(sim, n, 3)
        temp_surv = VirtualFile(sim, n, 3)
        num_rem = 0
        num_surv = 0
        
        for i in range(n):
            rec = vf_L_flags.read_record(i)
            node_id, next_id, weight, my_bit, next_bit = rec
            if my_bit == 1 and next_bit == 0:
                temp_rem.write_record(num_rem, [node_id, next_id, weight])
                num_rem += 1
            else:
                temp_surv.write_record(num_surv, [node_id, next_id, weight])
                num_surv += 1
                
        # Close temp flags files
        vf_rnd.close()
        vf_L_sort_node.close()
        vf_Rnd_sort_node.close()
        vf_L_with_my_bit.close()
        vf_L_sort_next.close()
        vf_L_flags.close()
        
        if num_rem > 0:
            # Allocate exact sized files
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
        
    # 2. Update Surviving links (compression)
    vf_surviving_sort_next = external_sort(sim, dm, vf_surviving, 1, M)
    vf_removed_sort_node = external_sort(sim, dm, vf_removed, 0, M)
    
    vf_surv_rem_join = merge_join(sim, dm, vf_surviving_sort_next, 1, vf_removed_sort_node, 0, 'left_outer', default_val=-1)
    vf_L_prime = resolve_surviving_links(sim, dm, vf_surv_rem_join)
    
    vf_surviving_sort_next.close()
    vf_surv_rem_join.close()
    
    # 3. Recursive Call
    vf_ranks_prime = list_ranking_rec(sim, dm, vf_L_prime, M, head_id, depth + 1)
    vf_L_prime.close()
    
    # 4. Restore Ranks
    vf_surviving_sort_next_re = external_sort(sim, dm, vf_surviving, 1, M)
    vf_parent_link_join = merge_join(sim, dm, vf_surviving_sort_next_re, 1, vf_removed_sort_node, 0, 'inner')
    vf_parent_link = extract_parent_link(sim, dm, vf_parent_link_join)
    
    vf_surviving_sort_next_re.close()
    vf_removed_sort_node.close()
    vf_parent_link_join.close()
    
    vf_parent_link_sort_parent = external_sort(sim, dm, vf_parent_link, 0, M)
    vf_parent_link.close()
    
    vf_ranks_prime_sort = external_sort(sim, dm, vf_ranks_prime, 0, M)
    
    vf_restored_ranks_join = merge_join(sim, dm, vf_parent_link_sort_parent, 0, vf_ranks_prime_sort, 0, 'inner')
    vf_restored_ranks = calculate_restored_ranks(sim, dm, vf_restored_ranks_join)
    
    vf_parent_link_sort_parent.close()
    vf_ranks_prime_sort.close()
    vf_restored_ranks_join.close()
    
    # 5. Merge ranks: vf_ranks_prime + vf_restored_ranks -> vf_out
    vf_out = VirtualFile(sim, vf_ranks_prime.size + vf_restored_ranks.size, 2)
    for i in range(vf_ranks_prime.size):
        vf_out.write_record(i, vf_ranks_prime.read_record(i))
    for i in range(vf_restored_ranks.size):
        vf_out.write_record(vf_ranks_prime.size + i, vf_restored_ranks.read_record(i))
        
    vf_removed.close()
    vf_surviving.close()
    vf_ranks_prime.close()
    vf_restored_ranks.close()
    
    return vf_out

def list_ranking(sim, dm, vf_in, M):
    """
    Main entry point for External Memory List Ranking in the simulator.
    """
    head_id = find_head_external(sim, dm, vf_in, M)
    if head_id is None:
        raise ValueError("Could not find list head!")
        
    vf_temp_ranks = list_ranking_rec(sim, dm, vf_in, M, head_id, 0)
    vf_out = external_sort(sim, dm, vf_temp_ranks, 0, M)
    vf_temp_ranks.close()
    return vf_out
