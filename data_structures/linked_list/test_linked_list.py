import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from data_structures.linked_list import ExternalLinkedList
from io_simulator import VirtualDisk, IOSimulator

class TestExternalLinkedList:
    def test_basic_list_operations(self):
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=4, cache_memory_size=40)
        
        lst = ExternalLinkedList(sim)
        assert lst.is_empty()
        assert len(lst) == 0
        
        # Insert 10 at head
        p10 = lst.insert(None, 10)
        assert len(lst) == 1
        assert lst.lookup(p10) == 10
        
        # Insert 20 after 10
        p20 = lst.insert(p10, 20)
        assert len(lst) == 2
        assert lst.lookup(p20) == 20
        
        # Insert 30 after 20
        p30 = lst.insert(p20, 30)
        assert len(lst) == 3
        
        # Traverse list from p10
        assert lst.traverse(p10) == [10, 20, 30]
        
        # Remove 20
        lst.remove(p20)
        assert len(lst) == 2
        assert lst.traverse(p10) == [10, 30]
        
        # Check invalid pointer
        with pytest.raises(ValueError):
            lst.lookup(999)
            
        lst.close()

    def test_split_logic(self):
        """
        Verifies that blocks split correctly when the size exceeds 3B/2.
        For block size B = 4:
        - max_elements = 3B/2 = 6.
        - Pushing 7 elements should cause a split, creating two blocks.
        """
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=4, cache_memory_size=40)
        
        lst = ExternalLinkedList(sim)
        
        # Insert 6 elements at head (fits in 1 block)
        ptrs = []
        curr = None
        for i in range(6):
            curr = lst.insert(curr, i * 10)
            ptrs.append(curr)
            
        # Verify 1 block exists
        head_addr = lst.head_block_addr
        block = lst._read_block(head_addr)
        assert len(block['elements']) == 6
        assert block['next'] == -1
        
        # Insert 7th element, triggering a split
        lst.insert(curr, 60)
        
        # Verify split
        block_l = lst._read_block(head_addr)
        right_addr = block_l['next']
        assert right_addr != -1
        
        block_r = lst._read_block(right_addr)
        
        # The 7 elements should be divided roughly equally (3 and 4)
        assert len(block_l['elements']) == 3
        assert len(block_r['elements']) == 4
        assert block_r['prev'] == head_addr
        
        lst.close()

    def test_merge_and_redistribution_logic(self):
        """
        Verifies block merging and redistribution during deletions.
        """
        vd = VirtualDisk(size=1000)
        sim = IOSimulator(vd, block_size=4, cache_memory_size=40)
        
        lst = ExternalLinkedList(sim)
        
        # Insert elements to trigger a split (7 elements)
        ptrs = []
        curr = None
        for i in range(7):
            curr = lst.insert(curr, i)
            ptrs.append(curr)
            
        # Confirm split occurred
        head_addr = lst.head_block_addr
        block_l = lst._read_block(head_addr)
        right_addr = block_l['next']
        assert right_addr != -1
        
        # Now remove an element from the right block.
        # Right block size becomes 3. Left block is 3. B/2 = 2.
        # Still above B/2, no merge.
        lst.remove(ptrs[6])
        block_r = lst._read_block(right_addr)
        assert len(block_r['elements']) == 3
        
        # Remove another element from right block.
        # Right block size becomes 2. Left is 3. Still above B/2.
        lst.remove(ptrs[5])
        block_r = lst._read_block(right_addr)
        assert len(block_r['elements']) == 2
        
        # Remove a third element from right block.
        # Right block size becomes 1. This is less than B/2 (2).
        # We try to balance with left block (size 3).
        # Total elements = 1 (right) + 3 (left) = 4.
        # Since total <= max_elements (6), they should MERGE into a single block.
        lst.remove(ptrs[4])
        
        # Confirm they merged (right block is freed, left block has all 4 elements)
        block_merged = lst._read_block(head_addr)
        assert block_merged['next'] == -1
        assert len(block_merged['elements']) == 4
        
        lst.close()

    def test_traverse_io_complexity(self):
        """
        Verifies that sequential traversal takes O(K / B) block read operations,
        confirming that sequential nodes are grouped together on disk.
        """
        vd = VirtualDisk(size=5000)
        sim = IOSimulator(vd, block_size=10, cache_memory_size=100)
        
        lst = ExternalLinkedList(sim)
        
        # Insert 100 elements sequentially
        curr = None
        ptrs = []
        for i in range(100):
            curr = lst.insert(curr, i)
            ptrs.append(curr)
            
        # Flush simulation memory and reset I/O counter
        sim.flush_memory()
        sim.io_count = 0
        
        # Traverse the list starting from head
        results = lst.traverse(ptrs[0])
        assert len(results) == 100
        
        # With B=10, 100 elements are grouped in blocks of size between 5 and 15 elements.
        # Thus, there are around 10-15 list blocks on disk.
        # Since each list block on disk stores prev, next, elements count, and up to 15 elements,
        # its layout size is 3 + 2*15 = 33 elements, spanning 4 simulator blocks (of size 10).
        # Traversing them sequentially should trigger around 30-40 block reads.
        # This is still significantly fewer than 100 read operations (naive traversal of 100 elements).
        assert sim.io_count <= 45, f"Expected low block read count, got {sim.io_count}"
        
        lst.close()
