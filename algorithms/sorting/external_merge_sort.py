import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from io_simulator.io_simulator import IOSimulator


def external_merge_sort(disk: IOSimulator, array_size: int) -> tuple[np.ndarray, int]:
    """
    Optimal external memory sorting using k-way merge sort.

    This implementation achieves O((n/B) * log_{M/B}(n/M)) I/O complexity:
    - Divides array into M/B parts (not just 2)
    - Recursively sorts each part
    - Performs k-way merge of all parts simultaneously
    - Achieves optimal external memory sorting complexity

    Args:
        disk: IOSimulator instance containing the array to sort
        array_size: Number of elements in the array

    Returns:
        Tuple of (sorted_array, io_count)
    """
    # Reset I/O count to track only sorting operations
    initial_io_count = disk.io_count

    # Perform optimal k-way merge sort
    _multiway_merge_sort_recursive(disk, 0, array_size, array_size)

    # Calculate total I/O operations
    io_count = disk.io_count - initial_io_count

    # Return sorted array
    return disk.disk.copy(), io_count


def _multiway_merge_sort_recursive(disk: IOSimulator, start: int, end: int, total_length: int) -> None:
    """
    Optimal k-way external merge sort achieving O((n/B) * log_{M/B}(n/M)) complexity.

    Algorithm:
    1. Calculate optimal k = M/B (number of parts that fit in memory)
    2. If array fits in memory, sort it directly
    3. Otherwise, divide into k parts and recursively sort each
    4. Perform k-way merge of all sorted parts

    Args:
        disk: IOSimulator instance
        start: Start index of subarray
        end: End index of subarray (exclusive)
        total_length: Total length of array (for IOSimulator indexing)
    """
    length = end - start

    # Base case: arrays of size 1 or smaller are already sorted
    if length <= 1:
        return

    # Base case: if subarray fits in memory, sort it directly
    if _fits_in_memory(disk, length):
        _sort_in_memory(disk, start, end, total_length)
        return

    # Calculate optimal k = M/B for k-way merge
    # Use memory_limit as proxy for M/B (max blocks that fit in memory)
    k = max(2, disk.memory_limit)  # At least 2-way merge

    # Divide into k parts
    part_size = length // k
    remainder = length % k

    parts = []  # [(start_idx, end_idx), ...]
    current_start = start

    for i in range(k):
        # Distribute remainder among first few parts
        current_size = part_size + (1 if i < remainder else 0)
        current_end = current_start + current_size

        if current_size > 0:  # Only add non-empty parts
            parts.append((current_start, current_end))
            current_start = current_end

    # Recursively sort each part
    for part_start, part_end in parts:
        _multiway_merge_sort_recursive(disk, part_start, part_end, total_length)

    # Perform k-way merge of all sorted parts
    _k_way_merge(disk, parts, start, end, total_length)




def _fits_in_memory(disk: IOSimulator, length: int) -> bool:
    """
    Check if a subarray fits in memory.
    """
    # Simple heuristic: if length is small enough
    return length <= disk.memory_size // 2


def _sort_in_memory(disk: IOSimulator, start: int, end: int, total_length: int) -> None:
    """
    Sort a subarray that fits in memory.
    """
    # Force flush to ensure we read correct data
    disk.flush_memory()

    # Read subarray into memory
    subarray = []
    for i in range(start, end):
        element = disk.get_element(0, i, total_length)
        subarray.append(element)

    # Sort using standard algorithm
    subarray.sort()

    # Flush memory before writing to avoid conflicts
    disk.flush_memory()

    # Write back to disk
    for i, value in enumerate(subarray):
        disk.set_element(0, start + i, value, total_length)

    # Flush to ensure changes are persistent
    disk.flush_memory()




def _k_way_merge(disk: IOSimulator, parts: list[tuple[int, int]], start: int, end: int, total_length: int) -> None:
    """
    Perform k-way merge of sorted parts to achieve optimal I/O complexity.

    This is the key optimization: instead of merging 2 parts at a time (log₂ factor),
    we merge all k parts simultaneously (log_{k} factor where k = M/B).

    Algorithm:
    - Maintain one current element from each part
    - Repeatedly select the minimum element among all parts
    - Write selected element to output and advance that part's pointer
    - Continue until all parts are exhausted

    Args:
        disk: IOSimulator instance
        parts: List of (start, end) ranges of sorted parts to merge
        start: Start index of output range
        end: End index of output range (exclusive)
        total_length: Total array length for IOSimulator indexing
    """
    import heapq

    # Force flush to ensure clean state
    disk.flush_memory()

    if not parts:
        return

    # Initialize heap with first element from each part
    # Heap entry: (value, part_index, current_position)
    heap = []
    part_positions = []  # Current position for each part

    for i, (part_start, part_end) in enumerate(parts):
        if part_start < part_end:  # Non-empty part
            first_value = disk.get_element(0, part_start, total_length)
            heapq.heappush(heap, (first_value, i, part_start))
            part_positions.append(part_start)
        else:
            part_positions.append(part_end)  # Empty part

    # Merge process: repeatedly extract minimum and refill heap
    merged_result = []

    while heap:
        # Extract minimum element
        min_value, part_idx, current_pos = heapq.heappop(heap)
        merged_result.append(min_value)

        # Advance position in the selected part
        part_positions[part_idx] += 1
        next_pos = part_positions[part_idx]

        # If this part has more elements, add next element to heap
        part_start, part_end = parts[part_idx]
        if next_pos < part_end:
            next_value = disk.get_element(0, next_pos, total_length)
            heapq.heappush(heap, (next_value, part_idx, next_pos))

    # Write merged result back to disk
    for i, value in enumerate(merged_result):
        disk.set_element(0, start + i, value, total_length)

    # Force flush to ensure all writes are committed
    disk.flush_memory()


# Optimal k-way external merge sort implementation complete


# Example usage and testing
if __name__ == "__main__":
    print("Testing simplified external memory sorting...")

    # Test with a simple array
    test_array = np.array([5, 1, 3, 2, 8, 4, 7, 6])
    print(f"Simple test: {test_array}")

    disk = IOSimulator(test_array, block_size=2, memory_size=4)
    result, io_count = external_merge_sort(disk, len(test_array))

    expected = np.sort(test_array)
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    print(f"I/O count: {io_count}")

    if np.array_equal(result, expected):
        print("✓ Simple test passed!")
    else:
        print("✗ Simple test failed!")

    print()

    # Test with random array
    n = 16
    test_array = np.random.randint(0, 50, n)
    print(f"Random test: {test_array}")

    disk = IOSimulator(test_array, block_size=4, memory_size=8)
    result, io_count = external_merge_sort(disk, n)

    expected = np.sort(test_array)
    if np.array_equal(result, expected):
        print(f"✓ Random test passed! I/O count: {io_count}")
    else:
        print("✗ Random test failed!")
        print(f"Result: {result}")
        print(f"Expected: {expected}")
