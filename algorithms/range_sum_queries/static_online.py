import sys
from pathlib import Path
import math

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from io_simulator import VirtualFile

class StaticOnlineRSQNaive:
    """
    Static Online Range Sum Queries with O(N) disk storage.
    
    Theoretical Context:
    - Precomputes and stores prefix sums on the virtual disk.
    - Uses O(N) disk space.
    - Resolves each query in 2 I/O operations (reading S[j] and S[i-1]).
    """
    def __init__(self, sim, array_vf: VirtualFile):
        self.sim = sim
        self.size = array_vf.size
        self.prefix_sums_vf = VirtualFile(sim, self.size, 1)
        
        # Build prefix sums on disk
        current_sum = 0
        for i in range(self.size):
            val = array_vf.read_record(i)[0]
            current_sum += val
            self.prefix_sums_vf.write_record(i, [current_sum])
            
    def query(self, i: int, j: int) -> int:
        if i < 0 or j >= self.size or i > j:
            raise ValueError("Invalid range indices")
            
        # Read S[j]
        s_j = self.prefix_sums_vf.read_record(j)[0]
        
        # Read S[i-1] (if i > 0)
        s_i_minus_1 = 0
        if i > 0:
            s_i_minus_1 = self.prefix_sums_vf.read_record(i - 1)[0]
            
        return s_j - s_i_minus_1

    def close(self):
        self.prefix_sums_vf.close()


class StaticOnlineRSQBlock:
    """
    Static Online Range Sum Queries with O(N/B) disk storage.
    
    Theoretical Context:
    - Computes block-level prefix sums and stores them in RAM (O(N/B) elements).
    - Stores only the original array on the virtual disk.
    - To answer query (i, j):
      * Fetches full block sums from RAM in O(1) time (0 I/O).
      * Reads boundary elements inside the partial start and end blocks from disk.
      * Costs at most 2 Read I/Os (since the elements are cached within their respective blocks).
    """
    def __init__(self, sim, array_vf: VirtualFile):
        self.sim = sim
        self.B = sim.block_size
        self.size = array_vf.size
        self.array_vf = array_vf
        
        # Compute block prefix sums in RAM
        num_blocks = math.ceil(self.size / self.B)
        self.block_prefix_sums = [0] * num_blocks
        
        running_block_sum = 0
        for b in range(num_blocks):
            block_sum = 0
            start_idx = b * self.B
            end_idx = min(self.size, (b + 1) * self.B)
            for idx in range(start_idx, end_idx):
                block_sum += array_vf.read_record(idx)[0]
            running_block_sum += block_sum
            self.block_prefix_sums[b] = running_block_sum

    def query(self, i: int, j: int) -> int:
        if i < 0 or j >= self.size or i > j:
            raise ValueError("Invalid range indices")
            
        block_i = i // self.B
        block_j = j // self.B
        
        if block_i == block_j:
            # Entire query lies inside a single block.
            # Reading elements will only trigger 1 I/O due to caching.
            block_sum = 0
            for idx in range(i, j + 1):
                block_sum += self.array_vf.read_record(idx)[0]
            return block_sum
            
        # 1. Sum partial start block from i to the end of block_i
        start_block_sum = 0
        start_block_end = (block_i + 1) * self.B
        for idx in range(i, start_block_end):
            start_block_sum += self.array_vf.read_record(idx)[0]
            
        # 2. Sum full intermediate blocks using the RAM prefix sums
        middle_blocks_sum = 0
        if block_i + 1 <= block_j - 1:
            middle_blocks_sum = self.block_prefix_sums[block_j - 1] - self.block_prefix_sums[block_i]
            
        # 3. Sum partial end block from start of block_j to j
        end_block_sum = 0
        end_block_start = block_j * self.B
        for idx in range(end_block_start, j + 1):
            end_block_sum += self.array_vf.read_record(idx)[0]
            
        return start_block_sum + middle_blocks_sum + end_block_sum
