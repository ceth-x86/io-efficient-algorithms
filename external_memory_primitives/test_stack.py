import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from external_memory_primitives import ExternalStack
from io_simulator import VirtualDisk, IOSimulator

class TestExternalStack:
    def test_basic_lifo_correctness(self):
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=3, cache_memory_size=30)
        
        stack = ExternalStack(sim)
        assert stack.is_empty()
        assert len(stack) == 0
        
        # Push 10 elements
        for i in range(10):
            stack.push(i)
            assert len(stack) == i + 1
            assert not stack.is_empty()
            
        # Pop all 10 and verify LIFO order
        for i in reversed(range(10)):
            assert stack.pop() == i
            assert len(stack) == i
            
        assert stack.is_empty()
        
        # Popping from empty stack should raise IndexError
        with pytest.raises(IndexError):
            stack.pop()
            
        stack.close()

    def test_stack_hysteresis_and_io_counts(self):
        """
        Verifies the block I/O optimization and hysteresis behavior of the stack.
        For a block size B = 3:
        1. Pushing up to 5 elements should occur purely in-memory (0 I/Os).
        2. Pushing the 6th element triggers a buffer limit of 2B = 6. 
           It flushes the bottom B = 3 elements to disk. This costs 1 Write I/O.
        3. Popping elements back. The top 3 are in the RAM buffer, so popping them costs 0 I/Os.
        4. Popping the 4th element leaves the RAM buffer empty (0 elements), which triggers
           a block read from disk of size B = 3 elements. This costs 1 Read I/O.
        """
        vd = VirtualDisk(size=1000)
        # block_size = 3, cache capacity = 10 blocks (30 elements)
        sim = IOSimulator(vd, block_size=3, cache_memory_size=30)
        
        stack = ExternalStack(sim)
        
        # Flush simulator I/O counters
        sim.io_count = 0
        
        # 1. Push 5 elements
        for i in range(5):
            stack.push(i)
        assert sim.io_count == 0  # Still in memory
        assert len(stack.buffer) == 5
        assert len(stack.disk_blocks) == 0
        
        # 2. Push 6th element
        stack.push(5)
        sim.flush_memory()  # Flush dirty cache blocks to disk to register I/O count
        # In the simulator, write_element has a read-before-write behavior:
        # it first reads the block from disk (1 Read I/O), then modifies it in cache.
        # Calling flush_memory() writes it back to disk (1 Write I/O).
        # Therefore, this triggers a total of 2 I/Os.
        assert sim.io_count == 2
        assert len(stack.buffer) == 3  # top 3 elements [3, 4, 5] remain in RAM
        assert len(stack.disk_blocks) == 1
        
        # Reset simulator I/O counter to focus on pop behavior
        sim.io_count = 0
        
        # 3. Pop 3 elements (from RAM buffer)
        assert stack.pop() == 5
        assert stack.pop() == 4
        assert stack.pop() == 3
        assert sim.io_count == 0  # No I/O needed as we read from RAM
        
        # 4. Pop the 4th element (triggers reading the block from disk)
        assert stack.pop() == 2
        assert sim.io_count == 1  # 1 Read I/O occurred to load block [0, 1, 2] from disk
        
        # Pop the rest from RAM buffer
        assert stack.pop() == 1
        assert stack.pop() == 0
        
        assert stack.is_empty()
        stack.close()

    def test_stack_large_operations(self):
        vd = VirtualDisk(size=10000)
        sim = IOSimulator(vd, block_size=5, cache_memory_size=50)
        
        stack = ExternalStack(sim)
        
        # Push 200 elements
        for i in range(200):
            stack.push(i)
            
        assert len(stack) == 200
        
        # Pop 200 elements and check order
        for i in reversed(range(200)):
            assert stack.pop() == i
            
        assert stack.is_empty()
        stack.close()
