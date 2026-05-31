import sys
from pathlib import Path
import random

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from external_memory_primitives import external_sort
from io_simulator import VirtualDisk, IOSimulator, VirtualFile


def main():
    print("Initializing virtual disk and simulator...")
    vd = VirtualDisk(size=1000)
    # Memory size is 15 elements (5 records of size 3 each)
    sim = IOSimulator(vd, block_size=3, cache_memory_size=15)

    print("Generating unsorted records...")
    # Records: [key, value1, value2]
    data = [[random.randint(1, 100), i, i * 10] for i in range(20)]
    
    print("Original data:")
    for record in data:
        print(record)

    # Write data to virtual file
    vf_in = VirtualFile(sim, len(data), record_size=3)
    for i, record in enumerate(data):
        vf_in.write_record(i, record)
    sim.flush_memory()

    print("\nSorting records using external_sort (M=5 records limit)...")
    # Sort by key (index 0) using M = 5 records limit (fits 5 records in memory at once)
    vf_out = external_sort(sim, vf_in, key_index=0, M=5)
    sim.flush_memory()

    print("\nSorted data on virtual disk:")
    for i in range(vf_out.size):
        print(vf_out.read_record(i))

    print(f"\nTotal simulated I/O operations: {sim.io_count}")

    vf_in.close()
    vf_out.close()


if __name__ == "__main__":
    main()
