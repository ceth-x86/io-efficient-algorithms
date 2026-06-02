import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from io_simulator import VirtualDisk, IOSimulator, VirtualFile
from algorithms.range_sum_queries import (
    StaticOnlineRSQNaive,
    StaticOnlineRSQBlock,
    static_offline_rsq,
    dynamic_offline_rsq
)

def run_static_online_demo():
    print("\n--- 1. Static Online RSQ Demonstration ---")
    vd = VirtualDisk(size=1000)
    sim = IOSimulator(vd, block_size=4, cache_memory_size=40)
    
    # Array: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    array_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    array_vf = VirtualFile(sim, len(array_data), 1)
    for idx, val in enumerate(array_data):
        array_vf.write_record(idx, [val])
        
    print(f"Input Array: {array_data}")
    
    # Naive Version
    print("\nInitializing StaticOnlineRSQNaive (O(N) disk storage)...")
    sim.io_count = 0
    rsq_naive = StaticOnlineRSQNaive(sim, array_vf)
    print(f"Setup finished. I/O operations for precomputing prefix sums on disk: {sim.io_count}")
    
    sim.io_count = 0
    ans1 = rsq_naive.query(2, 7)  # 3 + 4 + 5 + 6 + 7 + 8 = 33
    print(f"Query(2, 7) Result: {ans1}")
    print(f"I/O operations for naive query: {sim.io_count} (Reads S[7] and S[1])")
    
    # Block-Grouped Version
    print("\nInitializing StaticOnlineRSQBlock (O(N/B) disk storage)...")
    sim.io_count = 0
    rsq_block = StaticOnlineRSQBlock(sim, array_vf)
    print(f"Setup finished. I/O operations for block sums: {sim.io_count}")
    
    sim.flush_memory()
    sim.io_count = 0
    ans2 = rsq_block.query(2, 7)
    print(f"Query(2, 7) Result: {ans2}")
    print(f"I/O operations for block-grouped query: {sim.io_count} (Reads only partial boundary blocks)")
    
    rsq_naive.close()
    array_vf.close()

def run_static_offline_demo():
    print("\n--- 2. Static Offline RSQ Demonstration ---")
    vd = VirtualDisk(size=1000)
    sim = IOSimulator(vd, block_size=4, cache_memory_size=40)
    
    # Array: [10, 20, 30, 40, 50, 60, 70, 80]
    array_data = [10, 20, 30, 40, 50, 60, 70, 80]
    array_vf = VirtualFile(sim, len(array_data), 1)
    for idx, val in enumerate(array_data):
        array_vf.write_record(idx, [val])
        
    print(f"Input Array: {array_data}")
    
    # Queries: (query_id, i, j)
    # q0: (0, 7) -> 360
    # q1: (2, 5) -> 180
    # q2: (4, 4) -> 50
    queries_data = [
        [0, 0, 7],
        [1, 2, 5],
        [2, 4, 4]
    ]
    queries_vf = VirtualFile(sim, len(queries_data), 3)
    for idx, q in enumerate(queries_data):
        queries_vf.write_record(idx, q)
        
    print("Queries:")
    for q in queries_data:
        print(f"  Query {q[0]}: range [{q[1]}, {q[2]}]")
        
    sim.io_count = 0
    results_vf = static_offline_rsq(sim, array_vf, queries_vf, M=4)
    print(f"\nProcessing finished. Total I/O operations for offline join: {sim.io_count}")
    
    print("Results:")
    for idx in range(results_vf.size):
        rec = results_vf.read_record(idx)
        print(f"  Query {rec[0]}: sum = {rec[1]}")
        
    array_vf.close()
    queries_vf.close()
    results_vf.close()

def run_dynamic_offline_demo():
    print("\n--- 3. Dynamic Offline RSQ Demonstration ---")
    vd = VirtualDisk(size=1000)
    sim = IOSimulator(vd, block_size=4, cache_memory_size=40)
    
    # Operations: [time_id, op_type, arg1, arg2]
    # OP_REL_UPDATE = 0, OP_QUERY = 1, OP_ABS_UPDATE = 2
    ops_data = [
        [0, 0, 2, 5],    # t0: a[2] += 5
        [1, 1, 0, 4],    # t1: query(0, 4) -> expected: 5
        [2, 2, 2, 3],    # t2: a[2] = 3
        [3, 1, 0, 4],    # t3: query(0, 4) -> expected: 3
        [4, 2, 4, 10],   # t4: a[4] = 10
        [5, 0, 4, -2],   # t5: a[4] += -2
        [6, 1, 0, 4]     # t6: query(0, 4) -> expected: 3 + 8 = 11
    ]
    
    ops_vf = VirtualFile(sim, len(ops_data), 4)
    for idx, op in enumerate(ops_data):
        ops_vf.write_record(idx, op)
        
    print("Chronological Operations (Updates and Queries mixed):")
    for op in ops_data:
        if op[1] == 0:
            print(f"  t{op[0]}: Update a[{op[2]}] += {op[3]}")
        elif op[1] == 1:
            print(f"  t{op[0]}: Query sum({op[2]}, {op[3]})")
        elif op[1] == 2:
            print(f"  t{op[0]}: Update a[{op[2]}] = {op[3]}")
            
    sim.io_count = 0
    results_vf = dynamic_offline_rsq(sim, N=8, ops_vf=ops_vf, M=4)
    print(f"\nProcessing finished. Total I/O operations for dynamic segment tree events: {sim.io_count}")
    
    print("Results:")
    for idx in range(results_vf.size):
        rec = results_vf.read_record(idx)
        print(f"  Query {rec[0]}: sum = {rec[1]}")
        
    ops_vf.close()
    results_vf.close()

def main():
    run_static_online_demo()
    run_static_offline_demo()
    run_dynamic_offline_demo()

if __name__ == "__main__":
    main()
