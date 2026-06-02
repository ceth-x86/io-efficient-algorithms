import sys
from pathlib import Path
import random

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from io_simulator import VirtualDisk, IOSimulator
from algorithms.sorting.partition_sort.in_place_partition_sort import in_place_sort

def main():
    print("Initializing virtual disk and simulator...")
    vd = VirtualDisk(size=1000)
    # Using block size B = 2 and memory size M = 4
    # This gives us d = M/B = 2 buffers in memory.
    sim = IOSimulator(vd, block_size=2, cache_memory_size=10)
    
    # Generate 12 random elements in range [1, 5]
    random.seed(123)
    N = 12
    K = 5
    data = [random.randint(1, K) for _ in range(N)]
    
    start_idx = 100
    print(f"\nWriting {N} elements to disk starting at index {start_idx}:")
    print(f"Original elements: {data}")
    
    for i, val in enumerate(data):
        sim.write_element(start_idx + i, val)
        
    sim.flush_memory()
    sim.io_count = 0
    
    print("\nSorting in-place on disk...")
    in_place_sort(sim, start_idx, N, K=K, M=4)
    sim.flush_memory()
    
    # Read the sorted array directly from the virtual disk
    sorted_data = [vd.disk[start_idx + i] for i in range(N)]
    print(f"Sorted elements on disk: {sorted_data}")
    print(f"Total I/O operations for in-place sort: {sim.io_count}")
    
    # Double check correctness
    assert sorted_data == sorted(data)
    print("\nVerification successful! The array on disk is sorted correctly.")

if __name__ == "__main__":
    main()
