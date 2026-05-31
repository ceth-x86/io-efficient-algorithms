import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data_structures.stack import ExternalStack
from io_simulator import VirtualDisk, IOSimulator

def main():
    print("Initializing virtual disk and simulator...")
    vd = VirtualDisk(size=1000)
    # Use block size B = 3
    sim = IOSimulator(vd, block_size=3, cache_memory_size=30)
    
    print("\nCreating ExternalStack (B = 3)...")
    stack = ExternalStack(sim)
    
    print("\nPushing 5 elements: 10, 20, 30, 40, 50...")
    # Buffer capacity is up to 2B = 6 elements. Pushing 5 elements should take 0 I/Os.
    sim.io_count = 0
    for val in [10, 20, 30, 40, 50]:
        stack.push(val)
    print(f"Stack size: {len(stack)}")
    print(f"RAM Buffer contents: {stack.buffer}")
    print(f"I/O operations so far: {sim.io_count} (all elements are stored in RAM)")

    print("\nPushing the 6th element: 60...")
    # Buffer reaches 2B = 6, triggering a flush of the bottom B=3 elements to disk.
    stack.push(60)
    sim.flush_memory()  # Flush simulator cache to register physical writes
    print(f"Stack size: {len(stack)}")
    print(f"RAM Buffer contents: {stack.buffer} (contains the top 3 elements: 40, 50, 60)")
    print(f"Disk blocks allocated: {len(stack.disk_blocks)}")
    print(f"I/O operations so far: {sim.io_count} (1 Read, 1 Write due to read-before-write cache policy of the simulator)")

    print("\nPopping 3 elements (should read from RAM)...")
    sim.io_count = 0
    print("Popped:", stack.pop())  # 60
    print("Popped:", stack.pop())  # 50
    print("Popped:", stack.pop())  # 40
    print(f"RAM Buffer contents: {stack.buffer} (empty)")
    print(f"I/O operations for these pops: {sim.io_count} (0 I/Os, read entirely from memory)")

    print("\nPopping the 4th element (RAM is empty, must load from disk)...")
    # This will read the block [10, 20, 30] from disk into the RAM buffer.
    popped_val = stack.pop()
    print("Popped:", popped_val)  # 30
    print(f"RAM Buffer contents: {stack.buffer} (contains the remaining elements: 10, 20)")
    print(f"I/O operations: {sim.io_count} (1 Read I/O to fetch the block from disk)")

    print("\nPopping the remaining elements...")
    print("Popped:", stack.pop())  # 20
    print("Popped:", stack.pop())  # 10
    print(f"Stack is empty: {stack.is_empty()}")
    
    stack.close()

if __name__ == "__main__":
    main()
