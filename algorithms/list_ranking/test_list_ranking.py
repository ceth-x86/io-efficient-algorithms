import sys
from pathlib import Path
import random
import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from algorithms.list_ranking.list_ranking import list_ranking, find_head_external
from algorithms.list_ranking.main import generate_random_linked_list, verify_results
from io_simulator import VirtualDisk, IOSimulator, VirtualFile


class TestListRanking:
    """Test cases for External Memory List Ranking."""

    @pytest.mark.parametrize(
        ("list_size", "memory_records"),
        [
            (10, 50),   # Fits in memory (base case)
            (100, 20),  # Multi-level recursion
            (300, 15),  # Deep recursion
        ]
    )
    def test_list_ranking_correctness(self, list_size: int, memory_records: int):
        # Set random seed for reproducibility
        random.seed(42)
        
        records, expected_ranks = generate_random_linked_list(list_size)
        
        record_size = 3
        block_size_elements = 9  # B = 3 records per block
        memory_size_elements = memory_records * record_size
        
        vd = VirtualDisk(size=100000)
        sim = IOSimulator(vd, block_size=block_size_elements, cache_memory_size=memory_size_elements)
        
        vf_in = VirtualFile(sim, list_size, record_size)
        for i, r in enumerate(records):
            vf_in.write_record(i, r)
            
        sim.flush_memory()
        
        # Run algorithm
        vf_out = list_ranking(sim, vd, vf_in, M=memory_records)
        
        sim.flush_memory()
        
        # Verify
        assert verify_results(vf_out, expected_ranks), "Computed ranks do not match ground truth!"
        
        vf_in.close()
        vf_out.close()

    @pytest.mark.parametrize(
        ("list_size", "memory_records", "block_records", "expected_io", "epsilon"),
        [
            (100, 50, 3, 7386, 0.1),
            (100, 20, 3, 10935, 0.1),
            (300, 30, 3, 33209, 0.1),
            (500, 50, 5, 34054, 0.1),
        ]
    )
    def test_list_ranking_io_complexity(self, list_size: int, memory_records: int, block_records: int, expected_io: int, epsilon: float):
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        records, expected_ranks = generate_random_linked_list(list_size)
        
        record_size = 3
        block_size_elements = block_records * record_size
        memory_size_elements = memory_records * record_size
        
        vd = VirtualDisk(size=10**6)
        sim = IOSimulator(vd, block_size=block_size_elements, cache_memory_size=memory_size_elements)
        
        vf_in = VirtualFile(sim, list_size, record_size)
        for i, r in enumerate(records):
            vf_in.write_record(i, r)
            
        sim.flush_memory()
        sim.io_count = 0
        
        # Run algorithm
        vf_out = list_ranking(sim, vd, vf_in, M=memory_records)
        sim.flush_memory()
        

        # Verify I/O complexity stays within the specified tolerance (epsilon)
        assert abs(sim.io_count - expected_io) <= expected_io * epsilon, f"Expected around {expected_io} I/Os, got {sim.io_count}"
        
        vf_in.close()
        vf_out.close()


class TestFindHeadExternal:
    """Test cases for the find_head_external function."""

    def test_find_head_simple(self):
        # A simple list: 2 -> 3 -> -1, 1 -> 2
        # Records: [[2, 3, 10], [3, -1, 10], [1, 2, 10]]
        # Head is 1.
        records = [[2, 3, 10], [3, -1, 10], [1, 2, 10]]
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=3, cache_memory_size=15)
        
        vf_in = VirtualFile(sim, len(records), 3)
        for i, rec in enumerate(records):
            vf_in.write_record(i, rec)
        sim.flush_memory()
        
        head = find_head_external(sim, vd, vf_in, M=10)
        assert head == 1
        
        vf_in.close()

    def test_find_head_single_element(self):
        # List: 42 -> -1
        records = [[42, -1, 5]]
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=3, cache_memory_size=15)
        
        vf_in = VirtualFile(sim, len(records), 3)
        for i, rec in enumerate(records):
            vf_in.write_record(i, rec)
        sim.flush_memory()
        
        head = find_head_external(sim, vd, vf_in, M=5)
        assert head == 42
        
        vf_in.close()

    def test_find_head_large_random(self):
        random.seed(42)
        records, expected_ranks = generate_random_linked_list(150)
        
        # Ground truth head is the one with rank 0
        expected_head = [node_id for node_id, rank in expected_ranks.items() if rank == 0][0]
        
        vd = VirtualDisk(size=10000)
        sim = IOSimulator(vd, block_size=9, cache_memory_size=90)
        
        vf_in = VirtualFile(sim, len(records), 3)
        for i, rec in enumerate(records):
            vf_in.write_record(i, rec)
        sim.flush_memory()
        
        head = find_head_external(sim, vd, vf_in, M=20)
        assert head == expected_head
        
        vf_in.close()

