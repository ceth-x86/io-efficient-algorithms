import unittest
from io_simulator import VirtualDisk, VirtualFile, IOSimulator, VirtualMatrix, DiskBTree

class TestVirtualDisk(unittest.TestCase):
    def test_allocation_and_free(self):
        vd = VirtualDisk(size=100)
        # Allocate segments
        addr1 = vd.allocate(20)
        self.assertEqual(addr1, 0)
        
        addr2 = vd.allocate(30)
        self.assertEqual(addr2, 20)
        
        self.assertEqual(vd.free_segments, [(50, 50)])
        
        # Free segment and check merge
        vd.free(addr1, 20)
        self.assertEqual(vd.free_segments, [(0, 20), (50, 50)])
        
        vd.free(addr2, 30)
        self.assertEqual(vd.free_segments, [(0, 100)])

    def test_out_of_memory(self):
        vd = VirtualDisk(size=50)
        vd.allocate(40)
        with self.assertRaises(MemoryError):
            vd.allocate(20)


class TestVirtualFile(unittest.TestCase):
    def test_file_lifecycle_and_io(self):
        vd = VirtualDisk(size=200)
        sim = IOSimulator(vd, block_size=10, memory_size=50)
        
        vf1 = VirtualFile(sim, size=10, record_size=3) # requires 30 elements
        self.assertEqual(vf1.start_idx, 0)
        self.assertEqual(vd.free_segments, [(30, 170)])
        
        vf1.write_record(0, [10, 20, 30])
        vf1.write_record(1, [40, 50, 60])
        sim.flush_memory()
        
        self.assertEqual(vf1.read_record(0), [10, 20, 30])
        self.assertEqual(vf1.read_record(1), [40, 50, 60])
        
        vf1.close()
        self.assertEqual(vd.free_segments, [(0, 200)])


class TestVirtualMatrix(unittest.TestCase):
    def test_matrix_operations(self):
        vd = VirtualDisk(size=100)
        sim = IOSimulator(vd, block_size=10, memory_size=50)
        
        # 3x4 matrix = 12 elements
        mat = VirtualMatrix(sim, rows=3, cols=4)
        self.assertEqual(mat.start_idx, 0)
        
        # Set individual elements
        mat.set_element(0, 0, 1)
        mat.set_element(1, 2, 6)
        mat.set_element(2, 3, 12)
        
        self.assertEqual(mat.get_element(0, 0), 1)
        self.assertEqual(mat.get_element(1, 2), 6)
        self.assertEqual(mat.get_element(2, 3), 12)
        
        # Set submatrix
        sub = [
            [10, 20],
            [30, 40]
        ]
        mat.set_submatrix(0, 1, sub)
        
        # Check submatrix read
        # Row 0: [1, 10, 20, 0]
        # Row 1: [0, 30, 40, 6] (since (1,2)=6 was overwritten by 40)
        self.assertEqual(mat.get_submatrix(0, 2, 0, 3), [
            [1, 10, 20],
            [0, 30, 40]
        ])
        
        mat.close()
        self.assertEqual(vd.free_segments, [(0, 100)])


class TestDiskBTree(unittest.TestCase):
    def test_btree_insertion_and_search(self):
        vd = VirtualDisk(size=1000)
        # Block size = 9 elements -> max_keys = (9 - 3) // 2 = 3. t = 2 (2-3 Tree)
        sim = IOSimulator(vd, block_size=9, memory_size=90) # cache limit = 10 blocks
        
        btree = DiskBTree(sim)
        
        # Insert 3 keys (fills the root node)
        btree.insert(10, 100)
        btree.insert(20, 200)
        btree.insert(5, 50)
        
        # Root is node 0, keys should be [5, 10, 20]
        root = btree.read_node(btree.root_block_id)
        self.assertEqual(root.keys, [5, 10, 20])
        self.assertEqual(root.values, [50, 100, 200, 0])
        self.assertTrue(root.is_leaf)
        
        # Insert 4th key -> triggers split of root
        btree.insert(15, 150)
        
        # Root should now be a split internal node (not leaf)
        new_root = btree.read_node(btree.root_block_id)
        self.assertFalse(new_root.is_leaf)
        self.assertEqual(len(new_root.keys), 1) # one key should bubble up
        self.assertEqual(new_root.keys[0], 10)
        
        # Root should have two children
        self.assertEqual(len(new_root.values), 2)
        child1 = btree.read_node(new_root.values[0])
        child2 = btree.read_node(new_root.values[1])
        
        self.assertEqual(child1.keys, [5])
        self.assertEqual(child2.keys, [10, 15, 20])
        
        # Verify searches
        self.assertEqual(btree.search(5), (child1.block_id, 0, 50))
        self.assertEqual(btree.search(10), (child2.block_id, 0, 100)) # stored in leaf child2
        self.assertEqual(btree.search(15), (child2.block_id, 1, 150))
        self.assertEqual(btree.search(20), (child2.block_id, 2, 200))
        self.assertIsNone(btree.search(99))


class TestIOSimulatorCache(unittest.TestCase):
    def test_lru_caching(self):
        vd = VirtualDisk(size=100)
        sim = IOSimulator(vd, block_size=10, memory_size=30) # memory_limit = 3 blocks
        
        self.assertEqual(sim.io_count, 0)
        
        # Read from block 0
        sim.read_element(5)
        self.assertEqual(sim.io_count, 1)
        
        # Write to block 0 (cached, 0 I/O)
        sim.write_element(5, 42)
        self.assertEqual(sim.io_count, 1)
        
        # Fill cache with block 1 and block 2
        sim.write_element(15, 100) # block 1
        sim.write_element(25, 200) # block 2
        self.assertEqual(sim.io_count, 3)
        self.assertEqual(len(sim.memory), 3)
        
        # Access block 3 -> evicts block 0 (dirty, so it writes to disk + reads block 3)
        # Total +2 I/O
        sim.write_element(35, 300) # block 3
        self.assertEqual(sim.io_count, 5)
        
        # Flush and check persistence
        sim.flush_memory()
        self.assertEqual(vd.disk[5], 42)
        self.assertEqual(vd.disk[15], 100)
        self.assertEqual(vd.disk[25], 200)
        self.assertEqual(vd.disk[35], 300)

if __name__ == '__main__':
    unittest.main()
