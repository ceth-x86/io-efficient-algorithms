import sys
from pathlib import Path
import random
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from external_memory_primitives.external_sort import external_sort
from io_simulator import VirtualDisk, IOSimulator, VirtualFile


class TestExternalSort:
    """Test cases for the external_sort utility."""

    def test_sort_empty_file(self):
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=3, cache_memory_size=9)
        vf_in = VirtualFile(sim, 0, 3)
        vf_out = external_sort(sim, vf_in, key_index=0, M=3)
        
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
        
        vf_out = external_sort(sim, vf_in, key_index=key_index, M=20)
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
        vf_out = external_sort(sim, vf_in, key_index=0, M=memory_records)
        sim.flush_memory()
        
        expected = sorted(data, key=lambda x: x[0])
        actual = [vf_out.read_record(i) for i in range(file_size)]
        
        assert actual == expected
        vf_in.close()
        vf_out.close()

    @pytest.mark.parametrize(
        ("file_size", "expected_io"),
        [
            (100, 211),
            (300, 632),
            (500, 1111),
            (1000, 2380),
        ]
    )
    def test_external_sort_io_complexity(self, file_size: int, expected_io: int):
        """
        Validates the block I/O complexity of external_sort for different file sizes,
        with constant memory size (M=50 records) and block size (B=3 records).

        Theoretical Context:
        In the External Memory model (Aggarwal-Vitter I/O model), the theoretical I/O
        complexity of External Merge Sort is:
            Theta( (N/B) * log_{M/B} (N/B) ) block I/Os.
        
        Where:
            N = total number of records (file_size)
            M = memory capacity in records (memory_records)
            B = block size in records (block_records)
            N/B = number of blocks in the file
            M/B = merge fan-in factor (capacity of memory in terms of blocks)

        How the formula arises:
        1. Run Generation Phase:
           The input file is divided into runs of size M. Each run is read into memory,
           sorted, and written back to disk.
           - Number of runs: N/M
           - I/O cost: 1 read pass and 1 write pass over the entire data = 2 * (N/B) I/Os.
        2. Merging Phase:
           We merge the runs. Since we keep one block (or at least one record) from each
           run in memory, the merge fan-out is limited by the memory capacity.
           Specifically, we can merge at most d = M/B - 1 runs at once in a single pass.
           - Height of the merge tree (number of merge passes):
             log_{M/B} (N/M) = log_{M/B} (N/B) - 1.
           - Each pass reads and writes the entire dataset = 2 * (N/B) I/Os.
           - Merging I/O cost: 2 * (N/B) * log_{M/B} (N/M) I/Os.
        Total general complexity: 2 * (N/B) * (1 + log_{M/B} (N/M)) = Theta( (N/B) * log_{M/B} (N/B) ).

        Single-Pass vs Multi-Pass Merge Limits:
        - Single-Pass Merge (N <= M^2):
          If the number of runs N/M does not exceed the merge capacity (i.e. N/M <= M),
          we can merge all runs in a single pass. This corresponds to the condition:
              N/M <= M  =>  N <= M^2.
          Under this condition, the number of merge passes is exactly 1.
          Therefore, the total I/O cost is:
              2 * (N/B) [run generation] + 2 * (N/B) [single merge pass] = 4 * (N/B) block I/Os.
          Since M and B are held constant, the I/O complexity scales strictly linearly with N:
              O(N) block I/Os.
        - Multi-Pass Merge (N > M^2):
          If the number of runs N/M exceeds memory capacity M, we cannot merge them all
          at once because we cannot hold one record from each run in the min-heap.
          Instead, we must merge them in groups of d = M/B - 1 recursively, producing
          intermediate sorted runs.
          This requires multiple merge passes (height of the tree > 1).
          In this general case, for constant M and B, the logarithmic term is not constant,
          and the complexity scales as:
              Theta(N log N) block I/Os as N -> infinity.

        Test Parameters Analysis:
        - M = 50 records, B = 3 records.
        - The boundary for single-pass merge: N <= M^2 = 2500 records.
        - Since our test cases use file sizes N <= 1000 (which is <= 2500), they fall
          entirely within the Single-Pass Merge regime.
        - Thus, we expect the I/O complexity to scale linearly:
          - N = 100  => 211 I/Os
          - N = 300  => 632 I/Os  (~ 3x increase)
          - N = 500  => 1111 I/Os (~ 5x increase)
          - N = 1000 => 2380 I/Os (~ 10x increase)
        """
        random.seed(42)
        record_size = 3
        memory_records = 50
        block_records = 3
        epsilon = 0.1
        
        data = [[random.randint(1, 1000) for _ in range(record_size)] for _ in range(file_size)]
        vd = VirtualDisk(size=100000)
        
        sim = IOSimulator(
            vd, 
            block_size=block_records * record_size, 
            cache_memory_size=memory_records * record_size
        )
        
        vf_in = VirtualFile(sim, len(data), record_size)
        for i, rec in enumerate(data):
            vf_in.write_record(i, rec)
        sim.flush_memory()
        
        # Reset IO count before executing the sort
        sim.io_count = 0
        
        vf_out = external_sort(sim, vf_in, key_index=0, M=memory_records)
        sim.flush_memory()
        
        # Check actual block I/O operations against expected count within epsilon tolerance
        assert abs(sim.io_count - expected_io) <= expected_io * epsilon, (
            f"Expected around {expected_io} I/Os, got {sim.io_count}"
        )
        
        vf_in.close()
        vf_out.close()

