import sys
from pathlib import Path

# Add project root to sys.path to allow importing from io_simulator
sys.path.append(str(Path(__file__).parent.parent.parent))

class ExternalStack:
    """
    An External Memory Stack implementing the LIFO (Last-In, First-Out) data structure.
    
    Theoretical Context:
    To minimize the number of block I/O operations, this stack maintains an internal
    memory (RAM) buffer of size up to 2 * B elements, where B is the block size of the
    virtual disk (sim.block_size).
    
    Hysteresis and Block I/O Optimization:
    1. RAM Buffer Limit: The RAM buffer can hold up to 2*B elements.
    2. Flush Strategy (Hysteresis): When a push operation increases the buffer size to 2*B,
       the bottom B elements (the oldest elements currently in RAM) are flushed as a single
       block to disk. This leaves the top B elements in the RAM buffer.
    3. Load Strategy: When a pop operation is called and the RAM buffer is completely empty,
       we read the last block of B elements from the disk and load it into the RAM buffer.
    
    This strategy prevents boundary oscillation (thrashing). If we used a buffer of size B,
    alternating push and pop operations on the boundary would cause a block read/write on
    every step (O(1) I/O per operation). With a 2*B buffer and hysteresis, at least B pushes
    or pops must occur between any two physical disk I/O operations, guaranteeing an
    amortized block I/O complexity of O(1/B) per operation.
    """
    def __init__(self, sim):
        """
        Initializes the external memory stack.
        
        Args:
            sim (IOSimulator): The simulator handling I/O operations and caching.
        """
        self.sim = sim
        self.B = sim.block_size
        self.buffer = []            # The RAM buffer (holds up to 2B elements)
        self.disk_blocks = []       # List of virtual disk addresses where blocks are saved

    def push(self, val):
        """
        Pushes an element onto the stack.
        
        If the RAM buffer reaches size 2B, the bottom B elements are written to disk.
        """
        self.buffer.append(val)
        
        # When the RAM buffer fills up to 2B elements, we flush the bottom B elements to disk.
        if len(self.buffer) == 2 * self.B:
            # Oldest B elements
            to_write = self.buffer[:self.B]
            # Keep the newest B elements in RAM
            self.buffer = self.buffer[self.B:]
            
            # Allocate space for B elements on the virtual disk
            disk_addr = self.sim.disk.allocate(self.B)
            
            # Write the block to the virtual disk. Since disk_addr is B-aligned,
            # these sequential writes map directly to a single cached disk block.
            for i, val in enumerate(to_write):
                self.sim.write_element(disk_addr + i, val)
                
            # Store the block's disk address
            self.disk_blocks.append(disk_addr)

    def pop(self):
        """
        Pops and returns the top element of the stack.
        
        If the RAM buffer is empty, it reads the latest block of B elements from disk.
        Raises IndexError if the stack is completely empty.
        """
        if not self.buffer:
            # If the RAM buffer is empty, we must fetch the latest block from disk.
            if not self.disk_blocks:
                raise IndexError("pop from empty stack")
                
            # Fetch the latest block address
            disk_addr = self.disk_blocks.pop()
            read_elements = []
            
            # Read the B elements from the virtual disk
            for i in range(self.B):
                read_elements.append(self.sim.read_element(disk_addr + i))
                
            # Free the virtual disk space occupied by the popped block
            self.sim.disk.free(disk_addr, self.B)
            
            # Repopulate the RAM buffer with the retrieved block
            self.buffer = read_elements
            
        # Pop the latest element from the top of the RAM buffer
        return self.buffer.pop()

    def is_empty(self) -> bool:
        """
        Returns True if the stack is empty, False otherwise.
        """
        return len(self) == 0

    def __len__(self) -> int:
        """
        Returns the total number of elements currently stored in the stack.
        """
        return len(self.buffer) + len(self.disk_blocks) * self.B

    def close(self):
        """
        Releases all allocated disk blocks on the virtual disk.
        """
        for disk_addr in self.disk_blocks:
            self.sim.disk.free(disk_addr, self.B)
        self.disk_blocks.clear()
