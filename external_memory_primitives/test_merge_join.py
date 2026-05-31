import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from external_memory_primitives.merge_join import merge_join
from io_simulator import VirtualDisk, IOSimulator, VirtualFile


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
        
        vf_out = merge_join(sim, vf1, key1_index=0, vf2=vf2, key2_index=0, join_type='inner')
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
            sim, vf1, key1_index=0, vf2=vf2, key2_index=0, 
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
        vf_out1 = merge_join(sim, vf_empty, 0, vf_nonempty, 0, 'inner')
        assert vf_out1.size == 0
        vf_out1.close()
        
        # 2. Empty left, nonempty right (left_outer) -> empty (since left is size 0)
        vf_out2 = merge_join(sim, vf_empty, 0, vf_nonempty, 0, 'left_outer')
        assert vf_out2.size == 0
        vf_out2.close()
        
        # 3. Nonempty left, empty right (inner) -> empty
        vf_out3 = merge_join(sim, vf_nonempty, 0, vf_empty, 0, 'inner')
        assert vf_out3.size == 0
        vf_out3.close()
        
        # 4. Nonempty left, empty right (left_outer) -> nonempty padded
        vf_out4 = merge_join(sim, vf_nonempty, 0, vf_empty, 0, 'left_outer', default_val=-5)
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
        
        vf_out = merge_join(sim, vf1, key1_index=1, vf2=vf2, key2_index=0, join_type='inner')
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

    @pytest.mark.parametrize(
        ("file_size", "expected_io"),
        [
            (100, 181),
            (300, 530),
            (500, 881),
            (1000, 1756),
        ]
    )
    def test_merge_join_io_complexity(self, file_size: int, expected_io: int):
        """
        Validates the block I/O complexity of merge_join for different file sizes,
        with constant memory size (cache limit) and block size.

        Theoretical Context:
        In the External Memory model, the theoretical I/O complexity of a Merge Join
        on two pre-sorted files of size N_1 and N_2 is:
            O( (N_1 / B) + (N_2 / B) + (N_out / B) ) block I/Os.
        
        Where:
            N_1, N_2 = number of records in the input files
            N_out = number of records in the output file
            B = block size in records

        In our implementation:
        1. We scan vf1 sequentially (taking N_1 / B read block I/Os).
        2. We scan vf2 sequentially (taking N_2 / B read block I/Os).
        3. We write the joined result to a temporary file of size N_out (taking N_out / B write block I/Os).
        4. We copy the temporary file to the final output file of exact size (taking N_out / B read block I/Os and N_out / B write block I/Os).

        Therefore, the total I/O complexity is:
            Theta( (N_1 + N_2 + 3 * N_out) / B ) block I/Os.

        Under a constant block size B and cache size, if the input sizes scale with N
        and the output size scales linearly with N (e.g., N_1 = N, N_2 = N/2, N_out = N/2),
        the total I/O cost simplifies to:
            O(N) - i.e., strictly linear growth with respect to the input size N.

        This is verified by the empirical test data:
            - N = 100  => 181 I/Os
            - N = 300  => 530 I/Os  (~ 3x increase)
            - N = 500  => 881 I/Os  (~ 5x increase)
            - N = 1000 => 1756 I/Os (~ 10x increase)
        """
        record_size = 2
        block_records = 3
        memory_records = 30
        epsilon = 0.05
        
        vd = VirtualDisk(size=10000)
        sim = IOSimulator(
            vd,
            block_size=block_records * record_size,
            cache_memory_size=memory_records * record_size
        )
        
        # Prepare two sorted files
        # Left: [0, 0], [1, 10], ..., [N-1, (N-1)*10]
        # Right: [0, 0], [2, 200], ..., [2*i, 2*i*100] (half matches)
        data1 = [[i, i * 10] for i in range(file_size)]
        data2 = [[i, i * 100] for i in range(0, file_size, 2)]
        
        vf1 = VirtualFile(sim, len(data1), record_size)
        for i, rec in enumerate(data1):
            vf1.write_record(i, rec)
            
        vf2 = VirtualFile(sim, len(data2), record_size)
        for i, rec in enumerate(data2):
            vf2.write_record(i, rec)
            
        sim.flush_memory()
        
        # Reset IO count before executing the join
        sim.io_count = 0
        
        vf_out = merge_join(sim, vf1, key1_index=0, vf2=vf2, key2_index=0, join_type='inner')
        sim.flush_memory()
        
        # Verify the number of I/O operations is within 5% tolerance of expected
        assert abs(sim.io_count - expected_io) <= expected_io * epsilon, (
            f"Expected around {expected_io} I/Os, got {sim.io_count}"
        )
        
        vf1.close()
        vf2.close()
        vf_out.close()

