import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data_structures.linked_list import ExternalLinkedList
from io_simulator import VirtualDisk, IOSimulator

def main():
    print("Initializing virtual disk and simulator...")
    vd = VirtualDisk(size=2000)
    # Using block size B = 4
    # min_elements = max(1, B // 2) = 2
    # max_elements = 3B // 2 = 6
    sim = IOSimulator(vd, block_size=4, cache_memory_size=40)
    
    print("\nCreating ExternalLinkedList (B = 4)...")
    lst = ExternalLinkedList(sim)
    
    print(f"List initial state: empty={lst.is_empty()}, length={len(lst)}")
    
    # 1. Pushing elements sequentially to trigger a split
    print("\n--- Phase 1: Sequential Insertion & Block Splitting ---")
    ptrs = []
    curr = None
    
    # Insert 6 elements (max_elements is 6)
    print("Inserting 6 elements (values 10, 20, 30, 40, 50, 60)...")
    for val in [10, 20, 30, 40, 50, 60]:
        curr = lst.insert(curr, val)
        ptrs.append(curr)
        
    head_addr = lst.head_block_addr
    block = lst._read_block(head_addr)
    print(f"Elements in head block: {[val for _, val in block['elements']]}")
    print(f"Next block address: {block['next']} (None)")
    
    # Insert 7th element: triggers a split because length > 6
    print("\nInserting 7th element (value 70) to trigger a split...")
    curr = lst.insert(curr, 70)
    ptrs.append(curr)
    
    block_left = lst._read_block(head_addr)
    right_addr = block_left['next']
    block_right = lst._read_block(right_addr)
    
    print(f"After split, left block size: {len(block_left['elements'])}")
    print(f"Left block elements: {[val for _, val in block_left['elements']]}")
    print(f"After split, right block size: {len(block_right['elements'])}")
    print(f"Right block elements: {[val for _, val in block_right['elements']]}")
    
    # 2. Traverse the list sequentially
    print("\n--- Phase 2: Sequential Traversal I/O Bounds ---")
    sim.flush_memory()
    sim.io_count = 0
    
    print("Traversing the entire list...")
    traversed = lst.traverse(ptrs[0])
    print(f"Traversed values: {traversed}")
    print(f"Total I/O operations for traversal: {sim.io_count}")
    
    # 3. Removing elements to trigger redistribution and merging
    print("\n--- Phase 3: Deletions, Redistribution & Block Merging ---")
    # Let's insert more elements to get 3 blocks, then remove elements.
    # Current elements: 10, 20, 30 in block_left (3 elements)
    # and 40, 50, 60, 70 in block_right (4 elements)
    # Let's insert 4 more elements to block_right: 80, 90, 100
    print("Inserting 80, 90, 100 to grow the list...")
    for val in [80, 90, 100]:
        curr = lst.insert(curr, val)
        ptrs.append(curr)
        
    # Print current blocks
    curr_addr = lst.head_block_addr
    block_index = 0
    while curr_addr != -1:
        blk = lst._read_block(curr_addr)
        print(f"Block {block_index} (address {curr_addr}): {[val for _, val in blk['elements']]}")
        curr_addr = blk['next']
        block_index += 1
        
    # We will now delete elements.
    # Let's delete the elements in the middle block to trigger redistribution or merging.
    # Let's locate and remove 70, 80, 90.
    print("\nRemoving elements to trigger block balancing...")
    # Get pointer for 70, 80, 90
    val_to_ptr = {}
    curr_addr = lst.head_block_addr
    while curr_addr != -1:
        blk = lst._read_block(curr_addr)
        for elem_id, val in blk['elements']:
            val_to_ptr[val] = elem_id
        curr_addr = blk['next']
        
    print("Removing 70...")
    lst.remove(val_to_ptr[70])
    print("Removing 80...")
    lst.remove(val_to_ptr[80])
    
    # Display the final block configuration
    curr_addr = lst.head_block_addr
    block_index = 0
    while curr_addr != -1:
        blk = lst._read_block(curr_addr)
        print(f"Final Block {block_index}: {[val for _, val in blk['elements']]}")
        curr_addr = blk['next']
        block_index += 1
        
    print(f"Final list traversal: {lst.traverse(ptrs[0])}")
    
    lst.close()
    print("\nFinished successfully.")

if __name__ == "__main__":
    main()
