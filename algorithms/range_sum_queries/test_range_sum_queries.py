import sys
from pathlib import Path
import pytest

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from io_simulator import VirtualDisk, IOSimulator, VirtualFile
from algorithms.range_sum_queries import (
    StaticOnlineRSQNaive,
    StaticOnlineRSQBlock,
    static_offline_rsq,
    dynamic_offline_rsq
)

class TestRangeSumQueries:
    def test_static_online_naive(self):
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=4, cache_memory_size=40)
        
        # Array of size 8: [10, 20, 30, 40, 50, 60, 70, 80]
        array_data = [10, 20, 30, 40, 50, 60, 70, 80]
        array_vf = VirtualFile(sim, 8, 1)
        for i, val in enumerate(array_data):
            array_vf.write_record(i, [val])
            
        rsq = StaticOnlineRSQNaive(sim, array_vf)
        
        # Test queries
        assert rsq.query(0, 7) == 360
        assert rsq.query(2, 5) == 180
        assert rsq.query(4, 4) == 50
        assert rsq.query(0, 0) == 10
        
        # Test invalid index
        with pytest.raises(ValueError):
            rsq.query(-1, 5)
        with pytest.raises(ValueError):
            rsq.query(2, 8)
        with pytest.raises(ValueError):
            rsq.query(4, 3)
            
        rsq.close()
        array_vf.close()

    def test_static_online_block(self):
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=4, cache_memory_size=40)
        
        array_data = [10, 20, 30, 40, 50, 60, 70, 80]
        array_vf = VirtualFile(sim, 8, 1)
        for i, val in enumerate(array_data):
            array_vf.write_record(i, [val])
            
        rsq = StaticOnlineRSQBlock(sim, array_vf)
        
        # Test queries
        assert rsq.query(0, 7) == 360
        assert rsq.query(2, 5) == 180
        assert rsq.query(4, 7) == 260
        assert rsq.query(4, 4) == 50
        
        # Check I/O operations for query (2, 5)
        # Block size B = 4
        # index 2 is in block 0
        # index 5 is in block 1
        # Reading boundary elements should cost 2 Read I/Os.
        sim.flush_memory()
        sim.io_count = 0
        ans = rsq.query(2, 5)
        assert ans == 180
        assert sim.io_count <= 2
        
        array_vf.close()

    def test_static_offline(self):
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=4, cache_memory_size=40)
        
        # Array: [10, 20, 30, 40, 50, 60, 70, 80]
        array_data = [10, 20, 30, 40, 50, 60, 70, 80]
        array_vf = VirtualFile(sim, 8, 1)
        for i, val in enumerate(array_data):
            array_vf.write_record(i, [val])
            
        # Queries file: [query_id, i, j]
        # q0: (0, 7) -> 360
        # q1: (2, 5) -> 180
        # q2: (4, 4) -> 50
        # q3: (0, 0) -> 10
        queries_data = [
            [0, 0, 7],
            [1, 2, 5],
            [2, 4, 4],
            [3, 0, 0]
        ]
        queries_vf = VirtualFile(sim, len(queries_data), 3)
        for idx, q in enumerate(queries_data):
            queries_vf.write_record(idx, q)
            
        # Call static offline query solver
        results_vf = static_offline_rsq(sim, array_vf, queries_vf, M=4)
        
        # Verify results: should be sorted by query_id
        assert results_vf.size == 4
        assert results_vf.read_record(0) == [0, 360]
        assert results_vf.read_record(1) == [1, 180]
        assert results_vf.read_record(2) == [2, 50]
        assert results_vf.read_record(3) == [3, 10]
        
        array_vf.close()
        queries_vf.close()
        results_vf.close()

    def test_dynamic_offline(self):
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
            
        # N = 8
        results_vf = dynamic_offline_rsq(sim, N=8, ops_vf=ops_vf, M=4)
        
        # Verify results (3 queries total)
        assert results_vf.size == 3
        assert results_vf.read_record(0) == [0, 5]
        assert results_vf.read_record(1) == [1, 3]
        assert results_vf.read_record(2) == [2, 11]
        
        ops_vf.close()
        results_vf.close()
