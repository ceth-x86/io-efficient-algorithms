import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

class ExternalLinkedList:
    """
    An External Memory Linked List optimized for the Aggarwal-Vitter I/O model.
    
    Theoretical Context:
    A naive external memory linked list stores each node in a separate block on disk.
    While point updates (insert, remove, lookup) cost O(1) I/O, traversing the list
    of size N takes O(N) I/Os due to pointer chasing across scattered disk blocks.
    
    This implementation solves the traversal problem by grouping consecutive list
    elements into physical blocks on disk.
    
    Block Constraints:
    - Each block contains between B/2 and 3B/2 consecutive elements.
    - B is the block size of the system (elements per block).
    
    Balancing Strategy (Split, Merge, and Redistribute):
    1. Insert: When an insert causes a block size to exceed 3B/2, the block is SPLIT
       into two blocks of size ~3B/4.
    2. Remove: When a remove causes a block size to drop below B/2:
       - If a adjacent sibling block is small (total elements <= 3B/2), we MERGE them.
       - If the sibling block is large, we REDISTRIBUTE elements between them so both
         have size ~B.
    
    This guarantees that splits/merges happen at most once every O(B) modifications.
    The amortized cost of block reorganizations is O(1/B) I/O.
    
    Pointer Invalidation & External Hash Table:
    Because splits and merges physically relocate elements on disk, direct disk addresses
    cannot serve as stable pointers.
    - We use an external hash table mapping a stable unique `pointer_id` to its
      current physical location: `pointer_id -> (block_address, offset_within_block)`.
    - Lookup: Costs O(1) I/O on average (modeled as 1 Read I/O).
    - Block reorganization updates O(B) entries in the hash table, costing O(B) I/Os.
      Since this occurs once per O(B) operations, the amortized update cost is O(1) I/O.
    """
    def __init__(self, sim):
        """
        Initializes the external linked list.
        
        Args:
            sim (IOSimulator): The simulator handling I/O operations.
        """
        self.sim = sim
        self.B = sim.block_size
        
        # Block size bounds
        self.min_elements = max(1, self.B // 2)
        self.max_elements = (3 * self.B) // 2
        
        # Disk block layout size (elements):
        # 0: prev_block_addr
        # 1: next_block_addr
        # 2: elements_count
        # 3 to 3 + 2 * max_elements - 1: slots for (id, value) pairs
        self.block_disk_size = 3 + 2 * self.max_elements
        
        self.head_block_addr = -1
        self.next_pointer_id = 1
        
        # Simulated external hash table: pointer_id -> (block_addr, offset_within_block)
        self.hash_table = {}

    def _read_block(self, addr: int) -> dict:
        """
        Reads a block representation from disk.
        """
        prev_addr = self.sim.read_element(addr + 0)
        next_addr = self.sim.read_element(addr + 1)
        count = self.sim.read_element(addr + 2)
        
        elements = []
        for i in range(count):
            elem_id = self.sim.read_element(addr + 3 + 2 * i)
            elem_val = self.sim.read_element(addr + 3 + 2 * i + 1)
            elements.append((elem_id, elem_val))
            
        return {
            'prev': prev_addr,
            'next': next_addr,
            'elements': elements
        }

    def _write_block(self, addr: int, block: dict):
        """
        Writes a block representation to disk.
        """
        self.sim.write_element(addr + 0, block['prev'])
        self.sim.write_element(addr + 1, block['next'])
        count = len(block['elements'])
        self.sim.write_element(addr + 2, count)
        
        for i, (elem_id, elem_val) in enumerate(block['elements']):
            self.sim.write_element(addr + 3 + 2 * i, elem_id)
            self.sim.write_element(addr + 3 + 2 * i + 1, elem_val)

    def _update_hash_table_for_block(self, block_addr: int, elements: list):
        """
        Updates the physical offsets of all elements inside a block in the hash table.
        Costs 1 I/O per updated element in the simulated external hash table.
        """
        for i, (elem_id, _) in enumerate(elements):
            self.hash_table[elem_id] = (block_addr, i)
            self.sim.io_count += 1  # 1 Write I/O per hash table update

    def lookup(self, pointer: int) -> int:
        """
        Returns the value of the element pointed to by the unique pointer ID.
        
        Complexity: O(1) I/O.
        """
        if pointer not in self.hash_table:
            raise ValueError("Invalid pointer")
            
        # 1 Read I/O for looking up pointer location in the external hash table
        self.sim.io_count += 1
        
        block_addr, offset = self.hash_table[pointer]
        
        # Read the block (block read is cached by IOSimulator)
        block = self._read_block(block_addr)
        return block['elements'][offset][1]

    def insert(self, pointer: int, value: int) -> int:
        """
        Inserts a new value after the element pointed to by pointer.
        If pointer is None, inserts at the head of the list.
        
        Returns the new unique pointer ID.
        """
        new_id = self.next_pointer_id
        self.next_pointer_id += 1
        
        # Case 1: Inserting into an empty list
        if self.head_block_addr == -1:
            addr = self.sim.disk.allocate(self.block_disk_size)
            block = {'prev': -1, 'next': -1, 'elements': [(new_id, value)]}
            self._write_block(addr, block)
            self.head_block_addr = addr
            
            # Hash table insert
            self.hash_table[new_id] = (addr, 0)
            self.sim.io_count += 1
            return new_id

        # Case 2: Inserting at the head of the list (pointer is None)
        if pointer is None:
            addr = self.head_block_addr
            block = self._read_block(addr)
            block['elements'].insert(0, (new_id, value))
            
            # Check for split
            if len(block['elements']) > self.max_elements:
                self._split_block(addr, block)
            else:
                self._write_block(addr, block)
                self._update_hash_table_for_block(addr, block['elements'])
                
            return new_id

        # Case 3: Inserting after a specific pointer
        if pointer not in self.hash_table:
            raise ValueError("Invalid pointer")
            
        # 1 Read I/O for looking up pointer location in the external hash table
        self.sim.io_count += 1
        block_addr, offset = self.hash_table[pointer]
        
        block = self._read_block(block_addr)
        block['elements'].insert(offset + 1, (new_id, value))
        
        # Check for split
        if len(block['elements']) > self.max_elements:
            self._split_block(block_addr, block)
        else:
            self._write_block(block_addr, block)
            self._update_hash_table_for_block(block_addr, block['elements'])
            
        return new_id

    def _split_block(self, addr: int, block: dict):
        """
        Splits the block at addr into two blocks of approximately equal size.
        """
        elements = block['elements']
        mid = len(elements) // 2
        
        left_elements = elements[:mid]
        right_elements = elements[mid:]
        
        # Allocate new block on disk
        new_addr = self.sim.disk.allocate(self.block_disk_size)
        next_addr = block['next']
        
        # Write right block
        right_block = {'prev': addr, 'next': next_addr, 'elements': right_elements}
        self._write_block(new_addr, right_block)
        
        # Update left block
        block['next'] = new_addr
        block['elements'] = left_elements
        self._write_block(addr, block)
        
        # Update successor's prev pointer if it exists
        if next_addr != -1:
            next_block = self._read_block(next_addr)
            next_block['prev'] = new_addr
            self._write_block(next_addr, next_block)
            
        # Update hash table mapping for all relocated elements
        self._update_hash_table_for_block(addr, left_elements)
        self._update_hash_table_for_block(new_addr, right_elements)

    def remove(self, pointer: int):
        """
        Removes the element pointed to by pointer from the list.
        """
        if pointer not in self.hash_table:
            raise ValueError("Invalid pointer")
            
        # 1 Read I/O for looking up pointer location in the external hash table
        self.sim.io_count += 1
        block_addr, offset = self.hash_table[pointer]
        
        block = self._read_block(block_addr)
        block['elements'].pop(offset)
        
        # Remove from hash table
        del self.hash_table[pointer]
        self.sim.io_count += 1  # 1 Write I/O to delete from hash table
        
        # Check if block violates min constraint
        if len(block['elements']) < self.min_elements:
            self._balance_block_after_removal(block_addr, block)
        else:
            self._write_block(block_addr, block)
            self._update_hash_table_for_block(block_addr, block['elements'])

    def _balance_block_after_removal(self, addr: int, block: dict):
        """
        Restores block invariants by merging or redistributing with sibling blocks.
        """
        next_addr = block['next']
        prev_addr = block['prev']
        
        # Try to balance with successor block
        if next_addr != -1:
            next_block = self._read_block(next_addr)
            total_elements = len(block['elements']) + len(next_block['elements'])
            
            if total_elements <= self.max_elements:
                # Merge block and next_block
                block['elements'] += next_block['elements']
                block['next'] = next_block['next']
                self._write_block(addr, block)
                
                # Update successor's prev pointer
                if next_block['next'] != -1:
                    succ_block = self._read_block(next_block['next'])
                    succ_block['prev'] = addr
                    self._write_block(next_block['next'], succ_block)
                    
                # Free space of next_block
                self.sim.disk.free(next_addr, self.block_disk_size)
                
                # Update hash table mapping for all merged elements
                self._update_hash_table_for_block(addr, block['elements'])
            else:
                # Redistribute elements between block and next_block
                all_elems = block['elements'] + next_block['elements']
                mid = len(all_elems) // 2
                block['elements'] = all_elems[:mid]
                next_block['elements'] = all_elems[mid:]
                
                self._write_block(addr, block)
                self._write_block(next_addr, next_block)
                
                self._update_hash_table_for_block(addr, block['elements'])
                self._update_hash_table_for_block(next_addr, next_block['elements'])
                
        # Symmetrically try to balance with predecessor block
        elif prev_addr != -1:
            prev_block = self._read_block(prev_addr)
            total_elements = len(prev_block['elements']) + len(block['elements'])
            
            if total_elements <= self.max_elements:
                # Merge prev_block and block
                prev_block['elements'] += block['elements']
                prev_block['next'] = block['next']
                self._write_block(prev_addr, prev_block)
                
                # Free space of block
                self.sim.disk.free(addr, self.block_disk_size)
                
                self._update_hash_table_for_block(prev_addr, prev_block['elements'])
            else:
                # Redistribute elements between prev_block and block
                all_elems = prev_block['elements'] + block['elements']
                mid = len(all_elems) // 2
                prev_block['elements'] = all_elems[:mid]
                block['elements'] = all_elems[mid:]
                
                self._write_block(prev_addr, prev_block)
                self._write_block(addr, block)
                
                self._update_hash_table_for_block(prev_addr, prev_block['elements'])
                self._update_hash_table_for_block(addr, block['elements'])
                
        else:
            # Sibling-less block (only block in list)
            if not block['elements']:
                # List is now completely empty
                self.sim.disk.free(addr, self.block_disk_size)
                self.head_block_addr = -1
            else:
                self._write_block(addr, block)
                self._update_hash_table_for_block(addr, block['elements'])

    def traverse(self, pointer: int, count: int = None) -> list:
        """
        Starting from pointer, scans the list sequentially to collect up to `count` values.
        If count is None, traverses to the end of the list.
        
        Complexity: O(K / B) I/Os where K = count of elements traversed.
        """
        if self.head_block_addr == -1:
            return []
            
        if pointer is None:
            # Default to head block
            block_addr = self.head_block_addr
            offset = 0
        else:
            if pointer not in self.hash_table:
                raise ValueError("Invalid pointer")
                
            # 1 Read I/O for looking up pointer location in the external hash table
            self.sim.io_count += 1
            block_addr, offset = self.hash_table[pointer]
            
        results = []
        curr_addr = block_addr
        curr_offset = offset
        
        while curr_addr != -1:
            block = self._read_block(curr_addr)
            elements = block['elements']
            
            # Read elements from current block starting at offset
            for i in range(curr_offset, len(elements)):
                if count is not None and len(results) >= count:
                    return results
                results.append(elements[i][1])
                
            # Move to next block
            curr_addr = block['next']
            curr_offset = 0  # Start at beginning of next block
            
        return results

    def __len__(self) -> int:
        """
        Returns the total number of elements currently stored in the linked list.
        """
        return len(self.hash_table)

    def is_empty(self) -> bool:
        """
        Returns True if the list is empty, False otherwise.
        """
        return len(self) == 0

    def close(self):
        """
        Releases all allocated disk segments.
        """
        curr_addr = self.head_block_addr
        while curr_addr != -1:
            block = self._read_block(curr_addr)
            next_addr = block['next']
            self.sim.disk.free(curr_addr, self.block_disk_size)
            curr_addr = next_addr
            
        self.head_block_addr = -1
        self.hash_table.clear()
