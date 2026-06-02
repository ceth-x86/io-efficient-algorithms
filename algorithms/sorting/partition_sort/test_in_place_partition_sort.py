import sys
from pathlib import Path
import random

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from io_simulator import VirtualDisk, IOSimulator
from algorithms.sorting.partition_sort.in_place_partition_sort import in_place_sort

class TestInPlacePartitionSort:
    def test_basic_in_place_sort(self):
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=2, cache_memory_size=10)
        
        # Write array data at start index 100
        start_idx = 100
        N = 8
        array_data = [2, 1, 3, 2, 1, 3, 2, 1]
        
        for i, val in enumerate(array_data):
            sim.write_element(start_idx + i, val)
            
        # Run sorting. Elements are in range [1, 3]. M = 4.
        in_place_sort(sim, start_idx, N, K=3, M=4)
        sim.flush_memory()
        
        # Verify correctness
        sorted_data = [vd.disk[start_idx + i] for i in range(N)]
        assert sorted_data == sorted(array_data)

    def test_larger_in_place_sort(self):
        vd = VirtualDisk(size=2000)
        sim = IOSimulator(vd, block_size=5, cache_memory_size=50)
        
        start_idx = 200
        N = 100
        K = 15
        
        # Random data in range [1, 15]
        random.seed(42)
        array_data = [random.randint(1, K) for _ in range(N)]
        
        for i, val in enumerate(array_data):
            sim.write_element(start_idx + i, val)
            
        # Sort in-place with M = 20 (meaning 4 buffers of size 5)
        in_place_sort(sim, start_idx, N, K=K, M=20)
        sim.flush_memory()
        
        sorted_data = [vd.disk[start_idx + i] for i in range(N)]
        assert sorted_data == sorted(array_data)
        
    def test_already_sorted(self):
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=3, cache_memory_size=30)
        
        start_idx = 0
        N = 12
        array_data = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        
        for i, val in enumerate(array_data):
            sim.write_element(start_idx + i, val)
            
        in_place_sort(sim, start_idx, N, K=6, M=6)
        sim.flush_memory()
        
        sorted_data = [vd.disk[start_idx + i] for i in range(N)]
        assert sorted_data == array_data
