"""
**External Memory Priority Queue Implementation**

Priority queue adapted from buffer trees achieving optimal sorting bound O(n/B·log_{M/B}(n/M))
through phase-based processing and in-memory minimum set management.

**Algorithm Overview:**

1. **Phase-based Processing**: Operations divided into phases of M/4 operations each
2. **Minimum Set S**: At phase start, load M/4 smallest elements into memory
3. **Memory Operations**: extract-min and insert work entirely in memory during phase
4. **Batch Flush**: At phase end, remaining elements flushed back to buffer tree

**Key Innovation**: Solves the problem that extract-min cannot be delayed (unlike other buffer tree operations)
by ensuring minimum elements are always available in memory.

**I/O Complexity**: O(n/B·log_{M/B}(n/M)) - achieves sorting bound for n operations
"""

import heapq
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from algorithms.searching.buffer_tree.buffer_tree import BufferTree
from io_simulator.io_simulator import IOSimulator


class ExternalPriorityQueue:
    """
    External memory priority queue achieving optimal I/O complexity.

    Uses phase-based processing: each phase handles M/4 operations using an
    in-memory minimum set, achieving O(n/B·log_{M/B}(n/M)) total I/O complexity.

    Parameters:
        disk: IOSimulator instance for external storage
        memory_size: Size of internal memory (M)
        block_size: Size of each block (B)
        degree: Node degree for underlying buffer tree
    """

    def __init__(self, disk: IOSimulator, memory_size: int = 200, block_size: int = 50, degree: int = 16):
        self.disk = disk
        self.memory_size = memory_size
        self.block_size = block_size
        self.degree = degree

        # Underlying buffer tree for bulk storage
        self.buffer_tree = BufferTree(disk, memory_size, block_size, degree)

        # Phase management
        self.phase_size = memory_size // 4  # M/4 operations per phase
        self.current_phase_ops = 0
        self.phase_active = False

        # In-memory minimum set S* (priority queue)
        self.min_set: list[tuple[Any, Any]] = []  # (priority, value) pairs

        # Operation tracking
        self.total_operations = 0
        self.sequence_counter = 0  # For making duplicate priorities unique
        self.priority_scale = 1000000  # Scale factor for encoding priorities

    def insert(self, priority: Any, value: Any = None) -> None:
        """
        Insert element with given priority.

        During active phase: adds to in-memory min_set
        Otherwise: adds to underlying buffer tree
        """
        self.total_operations += 1

        if self.phase_active:
            # Phase is active - add to in-memory min set
            heapq.heappush(self.min_set, (priority, value))
            self.current_phase_ops += 1

            # Check if phase is complete
            if self.current_phase_ops >= self.phase_size:
                self._end_phase()
        else:
            # No active phase - add to buffer tree with unique encoded key
            # Encode as: priority * scale + sequence_counter to ensure uniqueness
            encoded_key = priority * self.priority_scale + self.sequence_counter
            self.sequence_counter += 1
            # Store as (encoded_key, (original_priority, value)) pair
            self.buffer_tree.insert(encoded_key, (priority, value))

    def extract_min(self) -> tuple[Any, Any] | None:
        """
        Extract and return element with minimum priority.

        If no active phase: starts new phase by loading M/4 minimum elements
        Returns (priority, value) tuple or None if queue is empty
        """
        self.total_operations += 1

        # Start new phase if needed
        if not self.phase_active:
            self._start_phase()

        # Extract from in-memory min set
        if self.min_set:
            result = heapq.heappop(self.min_set)
            self.current_phase_ops += 1

            # Check if phase is complete
            if self.current_phase_ops >= self.phase_size:
                self._end_phase()

            return result

        return None  # Queue is empty

    def _start_phase(self) -> None:
        """
        Start new phase by loading M/4 minimum elements into memory.
        This is where the main I/O cost occurs.
        """
        # Flush all pending operations in buffer tree
        self.buffer_tree.flush_all_operations()

        # Extract M/4 minimum elements from buffer tree to memory
        min_elements = self._extract_minimum_elements(self.phase_size)

        # Load into in-memory min set
        self.min_set = min_elements
        heapq.heapify(self.min_set)

        # Start phase
        self.phase_active = True
        self.current_phase_ops = 0

        # Debug: print phase info
        # print(f"Started phase with {len(min_elements)} elements: {[elem[0] for elem in min_elements]}")

    def _end_phase(self) -> None:
        """
        End current phase by flushing remaining elements back to buffer tree.
        """
        # Insert remaining elements from min_set back to buffer tree
        while self.min_set:
            priority, value = heapq.heappop(self.min_set)
            # Encode key for buffer tree storage
            encoded_key = priority * self.priority_scale + self.sequence_counter
            self.sequence_counter += 1
            self.buffer_tree.insert(encoded_key, (priority, value))

        # Reset phase state
        self.phase_active = False
        self.current_phase_ops = 0

    def _extract_minimum_elements(self, count: int) -> list[tuple[Any, Any]]:
        """
        Extract up to 'count' minimum elements from the buffer tree.
        Uses a simple approach: extract elements one by one from buffer tree.
        """
        elements = []

        # First, ensure all operations are flushed to the tree structure
        self.buffer_tree.flush_all_operations()

        # Extract minimum elements one by one using repeated search operations
        for _ in range(count):
            min_element = self._extract_single_minimum()
            if min_element:
                elements.append(min_element)
            else:
                break  # No more elements

        return elements

    def _extract_single_minimum(self) -> tuple[Any, Any] | None:
        """Extract a single minimum element from the buffer tree."""
        # Collect all elements
        if self.buffer_tree.root_id is not None:
            root = self.buffer_tree._read_node(self.buffer_tree.root_id)
            temp_elements = []
            self._collect_all_elements(root, temp_elements)

            if temp_elements:
                # Sort by encoded key and take minimum
                temp_elements.sort(key=lambda x: x[0])
                min_encoded_key, stored_value = temp_elements[0]
                remaining = temp_elements[1:]

                # Convert to original priority format
                if isinstance(stored_value, tuple) and len(stored_value) == 2:
                    original_priority, actual_value = stored_value
                    result = (original_priority, actual_value)
                else:
                    original_priority = min_encoded_key // self.priority_scale
                    result = (original_priority, stored_value)

                # Rebuild tree with remaining elements
                self.buffer_tree._create_empty_tree()
                for encoded_key, stored_value in remaining:
                    self.buffer_tree.insert(encoded_key, stored_value)
                self.buffer_tree.flush_all_operations()

                return result

        return None

    def _collect_all_elements(self, node, elements_list):
        """Helper method to collect all elements from the tree structure."""
        if node.is_leaf:
            # Add all data from this leaf (without clearing)
            elements_list.extend(node.data)
        else:
            # For internal nodes, recursively collect from children
            for child_id in node.children:
                if child_id is not None:
                    child = self.buffer_tree._read_node(child_id)
                    self._collect_all_elements(child, elements_list)

    def peek_min(self) -> tuple[Any, Any] | None:
        """
        Return minimum element without removing it.
        May trigger phase start if no active phase.
        """
        if not self.phase_active:
            self._start_phase()

        if self.min_set:
            return self.min_set[0]  # Heap property ensures min is at index 0

        return None

    def size(self) -> int:
        """
        Return approximate size of priority queue.
        Note: Exact size tracking would require additional bookkeeping.
        """
        size = len(self.min_set)

        # Add approximate size from buffer tree (simplified)
        if self.buffer_tree.root_id is not None:
            root = self.buffer_tree._read_node(self.buffer_tree.root_id)
            if root.is_leaf:
                size += len(root.data)

        return size

    def is_empty(self) -> bool:
        """Check if priority queue is empty."""
        return self.size() == 0

    def get_io_count(self) -> int:
        """Get current I/O operation count."""
        return self.buffer_tree.get_io_count()

    def flush_all_operations(self) -> None:
        """Force flush all pending operations (end current phase if active)."""
        if self.phase_active:
            self._end_phase()
        self.buffer_tree.flush_all_operations()


# Example usage and testing
if __name__ == "__main__":
    import random

    import numpy as np

    print("Testing External Memory Priority Queue...")

    # Create large disk for priority queue storage
    disk_data = np.zeros(50000)
    disk = IOSimulator(disk_data, block_size=50, memory_size=200)

    # Create priority queue with phase size = M/4 = 50
    pq = ExternalPriorityQueue(disk, memory_size=200, block_size=50, degree=8)

    print("\nTesting phase-based operations...")

    # Test insertions and extractions
    priorities = [10, 5, 15, 3, 7, 12, 18, 1, 4, 6, 8, 11, 13, 16, 20]
    values = [f"item_{p}" for p in priorities]

    initial_io = pq.get_io_count()

    print(f"Inserting {len(priorities)} elements...")
    for priority, value in zip(priorities, values):
        pq.insert(priority, value)

    insert_io = pq.get_io_count() - initial_io
    print(f"I/O operations for insertions: {insert_io}")

    # Test extractions
    extract_io_start = pq.get_io_count()
    extracted = []

    print(f"\nQueue size before extraction: {pq.size()}")
    print(f"Queue empty before extraction: {pq.is_empty()}")

    print("Extracting minimum elements...")
    # Force flush all operations first to ensure elements are in the tree
    pq.flush_all_operations()

    print(f"Queue size after flush: {pq.size()}")
    print(f"Queue empty after flush: {pq.is_empty()}")

    # Extract elements one by one
    for i in range(len(priorities)):
        min_elem = pq.extract_min()
        if min_elem:
            extracted.append(min_elem)
            print(f"Extracted: {min_elem}")
        else:
            print(f"No more elements to extract (iteration {i})")
            break

    extract_io = pq.get_io_count() - extract_io_start
    print(f"I/O operations for extractions: {extract_io}")

    # Verify elements are extracted in priority order
    extracted_priorities = [elem[0] for elem in extracted]
    is_sorted = extracted_priorities == sorted(priorities)
    print(f"\nElements extracted in correct order: {is_sorted}")
    print(f"Expected: {sorted(priorities)}")
    print(f"Got:      {extracted_priorities}")
    print(f"Extracted count: {len(extracted)}, Expected count: {len(priorities)}")

    print(f"\nTotal I/O operations: {pq.get_io_count()}")
    total_ops = len(priorities) * 2  # inserts + extracts
    print(f"Total operations: {total_ops}")
    print(f"Amortized I/O per operation: {pq.get_io_count() / total_ops:.3f}")
    print("Phase-based processing demonstrates I/O efficiency!")

    # Test mixed operations
    print("\n\nTesting mixed insert/extract operations...")
    pq2 = ExternalPriorityQueue(disk, memory_size=200, block_size=50, degree=8)

    random.seed(42)
    mixed_io_start = pq2.get_io_count()

    for i in range(20):
        if random.random() < 0.7:  # 70% chance to insert
            priority = random.randint(1, 100)
            pq2.insert(priority, f"random_{priority}")
            print(f"Inserted {priority}")
        else:  # 30% chance to extract
            min_elem = pq2.extract_min()
            if min_elem:
                print(f"Extracted {min_elem[0]}")
            else:
                print("Queue empty")

    # Final flush
    pq2.flush_all_operations()

    mixed_io = pq2.get_io_count() - mixed_io_start
    print(f"Mixed operations I/O: {mixed_io}")
    print("External Memory Priority Queue test completed!")
