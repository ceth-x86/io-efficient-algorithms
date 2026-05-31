import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from external_memory_primitives import merge_join
from io_simulator import VirtualDisk, IOSimulator, VirtualFile


def main():
    print("Initializing virtual disk and simulator...")
    vd = VirtualDisk(size=1000)
    sim = IOSimulator(vd, block_size=3, cache_memory_size=30)

    # Pre-sorted files on join key (index 0)
    # File 1: [node_id, label]
    data1 = [[1, 100], [2, 200], [4, 400]]
    # File 2: [node_id, next_id]
    data2 = [[2, 3], [3, 4], [4, -1]]

    vf1 = VirtualFile(sim, len(data1), record_size=2)
    for i, rec in enumerate(data1):
        vf1.write_record(i, rec)

    vf2 = VirtualFile(sim, len(data2), record_size=2)
    for i, rec in enumerate(data2):
        vf2.write_record(i, rec)
    sim.flush_memory()

    print("Data 1 (node_id -> label):", data1)
    print("Data 2 (node_id -> next_id):", data2)

    # 1. Inner Join on key index 0
    print("\nPerforming Inner Join on node_id...")
    vf_inner = merge_join(sim, vf1, key1_index=0, vf2=vf2, key2_index=0, join_type='inner')
    sim.flush_memory()
    
    print("Inner Join Results:")
    for i in range(vf_inner.size):
        print(vf_inner.read_record(i))
    vf_inner.close()

    # 2. Left Outer Join on key index 0
    print("\nPerforming Left Outer Join on node_id (padding missing matches with -999)...")
    vf_outer = merge_join(
        sim, vf1, key1_index=0, vf2=vf2, key2_index=0, 
        join_type='left_outer', default_val=-999
    )
    sim.flush_memory()
    
    print("Left Outer Join Results:")
    for i in range(vf_outer.size):
        print(vf_outer.read_record(i))
    vf_outer.close()

    vf1.close()
    vf2.close()


if __name__ == "__main__":
    main()
