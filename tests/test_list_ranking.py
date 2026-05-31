import sys
from pathlib import Path
import random
import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.list_ranking.list_ranking import list_ranking
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
        sim = IOSimulator(vd, block_size=block_size_elements, memory_size=memory_size_elements)
        
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
