import sys
from pathlib import Path
import random
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from algorithms.list_ranking.io_utils import external_sort, merge_join
from io_simulator import VirtualDisk, IOSimulator, VirtualFile


class TestExternalSort:
    """Test cases for the external_sort utility."""

    def test_sort_empty_file(self):
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=3, cache_memory_size=9)
        vf_in = VirtualFile(sim, 0, 3)
        vf_out = external_sort(sim, vd, vf_in, key_index=0, M=3)
        
        assert vf_out.size == 0
        vf_in.close()
        vf_out.close()

    @pytest.mark.parametrize("key_index", [0, 1, 2])
    def test_sort_in_memory(self, key_index: int):
        """
        Tests sorting when the entire dataset fits within memory (size <= M).
        This executes the single sorted run (chunk) pathway without multi-way merging.
        """
        # Set random seed
        random.seed(42)
        
        data = [[random.randint(1, 100) for _ in range(3)] for _ in range(10)]
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=3, cache_memory_size=60) # M=20 records capacity
        
        vf_in = VirtualFile(sim, len(data), 3)
        for i, rec in enumerate(data):
            vf_in.write_record(i, rec)
        sim.flush_memory()
        
        vf_out = external_sort(sim, vd, vf_in, key_index=key_index, M=20)
        sim.flush_memory()
        
        expected = sorted(data, key=lambda x: x[key_index])
        actual = [vf_out.read_record(i) for i in range(len(data))]
        
        assert actual == expected
        vf_in.close()
        vf_out.close()

    @pytest.mark.parametrize(
        ("file_size", "memory_records", "block_size"),
        [
            (100, 15, 3),
            (250, 20, 5),
        ]
    )
    def test_sort_external(self, file_size: int, memory_records: int, block_size: int):
        """
        Tests sorting when the dataset is larger than memory (size > M).
        This forces creation of multiple sorted runs on disk and triggers the
        full multi-way external merge using a priority queue.
        """
        # Set random seed
        random.seed(123)
        
        record_size = 3
        data = [[random.randint(1, 1000) for _ in range(record_size)] for _ in range(file_size)]
        vd = VirtualDisk(size=10000)
        
        sim = IOSimulator(
            vd, 
            block_size=block_size * record_size, 
            cache_memory_size=memory_records * record_size
        )
        
        vf_in = VirtualFile(sim, len(data), record_size)
        for i, rec in enumerate(data):
            vf_in.write_record(i, rec)
        sim.flush_memory()
        
        # Sort by key_index = 0
        vf_out = external_sort(sim, vd, vf_in, key_index=0, M=memory_records)
        sim.flush_memory()
        
        expected = sorted(data, key=lambda x: x[0])
        actual = [vf_out.read_record(i) for i in range(file_size)]
        
        assert actual == expected
        vf_in.close()
        vf_out.close()


class TestMergeJoin:
    """Test cases for the merge_join utility."""

    def test_inner_join_basic(self):
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=3, cache_memory_size=30)
        
        # vf1 sorted by key index 0
        data1 = [[1, 10], [2, 20], [4, 40]]
        # vf2 sorted by key index 0
        data2 = [[2, 100], [3, 200], [4, 400]]
        
        vf1 = VirtualFile(sim, len(data1), 2)
        for i, rec in enumerate(data1):
            vf1.write_record(i, rec)
            
        vf2 = VirtualFile(sim, len(data2), 2)
        for i, rec in enumerate(data2):
            vf2.write_record(i, rec)
            
        sim.flush_memory()
        
        vf_out = merge_join(sim, vd, vf1, key1_index=0, vf2=vf2, key2_index=0, join_type='inner')
        sim.flush_memory()
        
        # Expected: [[2, 20, 100], [4, 40, 400]]
        expected = [[2, 20, 100], [4, 40, 400]]
        actual = [vf_out.read_record(i) for i in range(vf_out.size)]
        
        assert actual == expected
        
        vf1.close()
        vf2.close()
        vf_out.close()

    def test_left_outer_join_basic(self):
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=3, cache_memory_size=30)
        
        data1 = [[1, 10], [2, 20], [4, 40]]
        data2 = [[2, 100], [3, 200], [4, 400]]
        
        vf1 = VirtualFile(sim, len(data1), 2)
        for i, rec in enumerate(data1):
            vf1.write_record(i, rec)
            
        vf2 = VirtualFile(sim, len(data2), 2)
        for i, rec in enumerate(data2):
            vf2.write_record(i, rec)
            
        sim.flush_memory()
        
        vf_out = merge_join(
            sim, vd, vf1, key1_index=0, vf2=vf2, key2_index=0, 
            join_type='left_outer', default_val=-999
        )
        sim.flush_memory()
        
        # Expected: [[1, 10, -999], [2, 20, 100], [4, 40, 400]]
        expected = [[1, 10, -999], [2, 20, 100], [4, 40, 400]]
        actual = [vf_out.read_record(i) for i in range(vf_out.size)]
        
        assert actual == expected
        
        vf1.close()
        vf2.close()
        vf_out.close()

    def test_join_empty_files(self):
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=3, cache_memory_size=30)
        
        vf_empty = VirtualFile(sim, 0, 2)
        vf_nonempty = VirtualFile(sim, 2, 2)
        vf_nonempty.write_record(0, [1, 10])
        vf_nonempty.write_record(1, [2, 20])
        sim.flush_memory()
        
        # 1. Empty left, nonempty right (inner) -> empty
        vf_out1 = merge_join(sim, vd, vf_empty, 0, vf_nonempty, 0, 'inner')
        assert vf_out1.size == 0
        vf_out1.close()
        
        # 2. Empty left, nonempty right (left_outer) -> empty (since left is size 0)
        vf_out2 = merge_join(sim, vd, vf_empty, 0, vf_nonempty, 0, 'left_outer')
        assert vf_out2.size == 0
        vf_out2.close()
        
        # 3. Nonempty left, empty right (inner) -> empty
        vf_out3 = merge_join(sim, vd, vf_nonempty, 0, vf_empty, 0, 'inner')
        assert vf_out3.size == 0
        vf_out3.close()
        
        # 4. Nonempty left, empty right (left_outer) -> nonempty padded
        vf_out4 = merge_join(sim, vd, vf_nonempty, 0, vf_empty, 0, 'left_outer', default_val=-5)
        assert vf_out4.size == 2
        assert vf_out4.read_record(0) == [1, 10, -5]
        assert vf_out4.read_record(1) == [2, 20, -5]
        vf_out4.close()
        
        vf_empty.close()
        vf_nonempty.close()

    def test_join_custom_keys(self):
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=3, cache_memory_size=30)
        
        # vf1 record: [id, parent_id, weight] -> key at index 1 (parent_id)
        # vf2 record: [node_id, rank] -> key at index 0 (node_id)
        data1 = [[10, 2, 100], [11, 4, 200]] # sorted by parent_id: 2, then 4
        data2 = [[2, 50], [3, 60], [4, 70]]   # sorted by node_id: 2, 3, 4
        
        vf1 = VirtualFile(sim, len(data1), 3)
        for i, rec in enumerate(data1):
            vf1.write_record(i, rec)
            
        vf2 = VirtualFile(sim, len(data2), 2)
        for i, rec in enumerate(data2):
            vf2.write_record(i, rec)
            
        sim.flush_memory()
        
        vf_out = merge_join(sim, vd, vf1, key1_index=1, vf2=vf2, key2_index=0, join_type='inner')
        sim.flush_memory()
        
        # Output record size = 3 + 2 - 1 = 4
        # Elements: vf1 record + extra fields of vf2 (excluding its join key)
        # Match 1: vf1's [10, 2, 100] joins with vf2's [2, 50]. Extra in vf2 is [50]. Joined = [10, 2, 100, 50]
        # Match 2: vf1's [11, 4, 200] joins with vf2's [4, 70]. Extra in vf2 is [70]. Joined = [11, 4, 200, 70]
        
        expected = [[10, 2, 100, 50], [11, 4, 200, 70]]
        actual = [vf_out.read_record(i) for i in range(vf_out.size)]
        
        assert actual == expected
        
        vf1.close()
        vf2.close()
        vf_out.close()

