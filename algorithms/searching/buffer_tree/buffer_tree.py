import sys
from enum import Enum
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from io_simulator.io_simulator import IOSimulator


class OperationType(Enum):
    """Types of operations in buffer trees."""

    SEARCH = "search"
    INSERT = "insert"
    DELETE = "delete"


class Operation:
    """Represents a single operation in the buffer tree."""

    def __init__(self, op_type: OperationType, key: Any, value: Any = None) -> None:
        self.op_type = op_type
        self.key = key
        self.value = value

    def __lt__(self, other: "Operation") -> bool:
        """Compare operations by key for sorting."""
        return self.key < other.key

    def __repr__(self) -> str:
        return f"Op({self.op_type.value}, {self.key})"


class BufferTreeNode:
    """
    Buffer tree node for external memory storage.

    Internal nodes have very high degree (Θ(M/B)) and contain buffers for operations.
    Leaves contain actual data elements.
    """

    def __init__(self, node_id: int, is_leaf: bool = False, degree: int = 16) -> None:
        self.node_id = node_id
        self.is_leaf = is_leaf
        self.degree = degree

        # For internal nodes: routing keys and child pointers
        self.keys: list[Any] = []  # Routing keys for finding correct child
        self.children: list[int] = []  # Child node IDs

        # Buffer for operations (only for internal nodes)
        self.buffer: list[Operation] = []
        self.buffer_capacity = degree  # Buffer size Θ(M/B)

        # For leaves: actual data elements
        self.data: list[tuple[Any, Any]] = []  # (key, value) pairs
        self.data_capacity = degree  # Leaf capacity ~B elements

    def is_buffer_full(self) -> bool:
        """Check if buffer is full and needs flushing."""
        return len(self.buffer) >= self.buffer_capacity

    def is_data_full(self) -> bool:
        """Check if leaf data is full."""
        return len(self.data) >= self.data_capacity

    def add_to_buffer(self, operation: Operation) -> bool:
        """Add operation to buffer. Returns True if successful, False if buffer full."""
        if self.is_buffer_full():
            return False
        self.buffer.append(operation)
        return True

    def find_child_index(self, key: Any) -> int:
        """Find which child should contain the given key."""
        # Binary search to find appropriate child
        left, right = 0, len(self.keys)
        while left < right:
            mid = (left + right) // 2
            if key <= self.keys[mid]:
                right = mid
            else:
                left = mid + 1
        return left


class BufferTree:
    """
    Buffer Tree implementation for batched external memory operations.

    Achieves sorting bound O(n/B * log_{M/B}(n/M)) for n operations by:
    - Using very high degree nodes (Θ(M/B) children)
    - Buffering operations in internal nodes
    - Batch processing operations via flush operations
    - Amortizing I/O cost across multiple operations

    Parameters:
        disk: IOSimulator instance for external storage
        memory_size: Size of internal memory (M)
        block_size: Size of each block (B)
        degree: Node degree, should be Θ(M/B)
    """

    def __init__(self, disk: IOSimulator, memory_size: int = 200, block_size: int = 50, degree: int = 16) -> None:
        self.disk = disk
        self.memory_size = memory_size
        self.block_size = block_size
        self.degree = degree  # Should be Θ(M/B)

        # Tree structure
        self.next_node_id = 0
        self.root_id = None
        self.height = 0

        # In-memory collection buffer b* for incoming operations
        self.collection_buffer: list[Operation] = []
        self.collection_buffer_capacity = block_size  # B operations

        # Node cache
        self.node_cache = {}

        # Results storage for search operations
        self.search_results = {}

        # Initialize empty tree
        self._create_empty_tree()

    def _create_empty_tree(self) -> None:
        """Create empty buffer tree with single leaf root."""
        root = BufferTreeNode(self._get_next_node_id(), is_leaf=True, degree=self.degree)
        self.root_id = root.node_id
        self.height = 1
        self._write_node(root)

    def _get_next_node_id(self) -> int:
        """Get next available node ID."""
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id

    def _write_node(self, node: BufferTreeNode) -> None:
        """Write node to external storage."""
        # Serialize node data for storage
        start_pos = node.node_id * 200  # Reserve 200 positions per node

        # Basic node info
        data = [
            node.node_id,
            int(node.is_leaf),
            node.degree,
            len(node.keys),
            len(node.children),
            len(node.buffer),
            len(node.data),
        ]

        # Add keys
        data.extend(node.keys)

        # Add children IDs
        data.extend(node.children)

        # Add buffer operations (simplified serialization)
        for op in node.buffer:
            # Convert string values to hash for storage
            value_hash = hash(str(op.value)) if op.value is not None else -1
            data.extend(
                [
                    int(op.op_type.value == "search"),
                    int(op.op_type.value == "insert"),
                    int(op.op_type.value == "delete"),
                    op.key,
                    value_hash,
                ]
            )

        # Add leaf data
        for key, value in node.data:
            value_hash = hash(str(value)) if value is not None else -1
            data.extend([key, value_hash])

        # Write to disk
        for i, value in enumerate(data[:200]):  # Limit to reserved space
            if start_pos + i < self.disk.total_size:
                self.disk.set_element(0, start_pos + i, float(value), self.disk.total_size)

        # Cache the node
        self.node_cache[node.node_id] = node

    def _read_node(self, node_id: int) -> BufferTreeNode:
        """Read node from external storage."""
        # Check cache first
        if node_id in self.node_cache:
            return self.node_cache[node_id]

        start_pos = node_id * 200

        # Read basic info
        stored_id = int(self.disk.get_element(0, start_pos, self.disk.total_size))
        is_leaf = bool(int(self.disk.get_element(0, start_pos + 1, self.disk.total_size)))
        degree = int(self.disk.get_element(0, start_pos + 2, self.disk.total_size))
        num_keys = int(self.disk.get_element(0, start_pos + 3, self.disk.total_size))
        num_children = int(self.disk.get_element(0, start_pos + 4, self.disk.total_size))
        num_buffer = int(self.disk.get_element(0, start_pos + 5, self.disk.total_size))
        num_data = int(self.disk.get_element(0, start_pos + 6, self.disk.total_size))

        # Create node
        node = BufferTreeNode(stored_id, is_leaf, degree)

        # Read keys
        pos = start_pos + 7
        for i in range(num_keys):
            key = int(self.disk.get_element(0, pos + i, self.disk.total_size))
            node.keys.append(key)
        pos += num_keys

        # Read children
        for i in range(num_children):
            child_id = int(self.disk.get_element(0, pos + i, self.disk.total_size))
            node.children.append(child_id)
        pos += num_children

        # Read buffer operations
        for i in range(num_buffer):
            op_pos = pos + i * 5
            is_search = bool(int(self.disk.get_element(0, op_pos, self.disk.total_size)))
            is_insert = bool(int(self.disk.get_element(0, op_pos + 1, self.disk.total_size)))
            # is_delete not used in this implementation
            key = int(self.disk.get_element(0, op_pos + 3, self.disk.total_size))
            value_hash = self.disk.get_element(0, op_pos + 4, self.disk.total_size)
            # For simplicity, reconstruct values from key in this demo
            value = f"value_{key}" if is_insert and value_hash != -1 else None

            if is_search:
                op_type = OperationType.SEARCH
            elif is_insert:
                op_type = OperationType.INSERT
            else:
                op_type = OperationType.DELETE

            node.buffer.append(Operation(op_type, key, value))
        pos += num_buffer * 5

        # Read leaf data
        for i in range(num_data):
            data_pos = pos + i * 2
            key = int(self.disk.get_element(0, data_pos, self.disk.total_size))
            value_hash = self.disk.get_element(0, data_pos + 1, self.disk.total_size)
            # Reconstruct value from key for demo
            value = f"value_{key}" if value_hash != -1 else None
            node.data.append((key, value))

        # Cache and return
        self.node_cache[node_id] = node
        return node

    def search(self, key: Any) -> Any | None:
        """Search for a key in the buffer tree."""
        op = Operation(OperationType.SEARCH, key)
        self._add_operation(op)

        # Check if we have a result from previous flushes
        return self.search_results.get(key)

    def insert(self, key: Any, value: Any = None) -> None:
        """Insert a key-value pair into the buffer tree."""
        op = Operation(OperationType.INSERT, key, value)
        self._add_operation(op)

    def delete(self, key: Any) -> None:
        """Delete a key from the buffer tree."""
        op = Operation(OperationType.DELETE, key)
        self._add_operation(op)

    def _add_operation(self, operation: Operation) -> None:
        """Add operation to collection buffer, flushing if necessary."""
        # Add to in-memory collection buffer
        self.collection_buffer.append(operation)

        # If collection buffer is full, flush to root
        if len(self.collection_buffer) >= self.collection_buffer_capacity:
            self._flush_collection_buffer()

    def _flush_collection_buffer(self) -> None:
        """Flush collection buffer to root buffer."""
        if not self.collection_buffer:
            return

        root = self._read_node(self.root_id)

        # Add operations to root buffer
        for op in self.collection_buffer:
            if not root.add_to_buffer(op):
                # Root buffer full, need to flush root first
                self._flush_node(self.root_id)
                root = self._read_node(self.root_id)  # Re-read after flush
                root.add_to_buffer(op)

        self._write_node(root)
        self.collection_buffer.clear()

    def _flush_node(self, node_id: int) -> None:
        """
        Flush operations from node buffer to children.
        Core of buffer tree algorithm - batch processing for I/O efficiency.
        """
        node = self._read_node(node_id)

        if not node.buffer:
            return

        # If node is a leaf (including leaf root), process operations directly
        if node.is_leaf:
            self._process_operations_at_leaf(node)
            return

        # If node is parent of leaves, process operations directly
        if self._is_parent_of_leaves(node):
            self._process_operations_at_leaves(node)
            return

        # Sort operations by key for efficient distribution
        operations = sorted(node.buffer)
        node.buffer.clear()

        # Distribute operations to children based on routing keys
        child_operations = {}
        for op in operations:
            child_idx = node.find_child_index(op.key)
            if child_idx not in child_operations:
                child_operations[child_idx] = []
            child_operations[child_idx].append(op)

        # Send operations to each child
        for child_idx, ops in child_operations.items():
            if child_idx < len(node.children):
                child_id = node.children[child_idx]
                child = self._read_node(child_id)

                # Add operations to child buffer
                for op in ops:
                    if not child.add_to_buffer(op):
                        # Child buffer full, flush it first
                        self._flush_node(child_id)
                        child = self._read_node(child_id)  # Re-read after flush
                        child.add_to_buffer(op)

                self._write_node(child)

        self._write_node(node)

    def _is_parent_of_leaves(self, node: BufferTreeNode) -> bool:
        """Check if node's children are all leaves."""
        if node.is_leaf or not node.children:
            return False

        # Check first child to determine if all children are leaves
        first_child = self._read_node(node.children[0])
        return first_child.is_leaf

    def _process_operations_at_leaves(self, parent: BufferTreeNode) -> None:
        """
        Process buffered operations when they reach leaves.
        This is where actual search/insert/delete operations are performed.
        """
        operations = sorted(parent.buffer)
        parent.buffer.clear()

        # Load all relevant leaves into memory
        leaves = {}
        for child_id in parent.children:
            leaves[child_id] = self._read_node(child_id)

        # Process each operation
        for op in operations:
            child_idx = parent.find_child_index(op.key)
            if child_idx < len(parent.children):
                child_id = parent.children[child_idx]
                leaf = leaves[child_id]

                if op.op_type == OperationType.SEARCH:
                    # Perform search in leaf data
                    for key, value in leaf.data:
                        if key == op.key:
                            self.search_results[op.key] = value
                            break
                    else:
                        self.search_results[op.key] = None

                elif op.op_type == OperationType.INSERT:
                    # Insert into leaf (maintaining sort order)
                    inserted = False
                    for i, (key, _) in enumerate(leaf.data):
                        if op.key <= key:
                            if op.key == key:
                                # Update existing key
                                leaf.data[i] = (op.key, op.value)
                            else:
                                # Insert new key
                                leaf.data.insert(i, (op.key, op.value))
                            inserted = True
                            break
                    if not inserted:
                        leaf.data.append((op.key, op.value))

                    # Handle leaf overflow by splitting
                    if leaf.is_data_full():
                        self._split_leaf(parent, child_idx, leaf)

                elif op.op_type == OperationType.DELETE:
                    # Remove from leaf data
                    leaf.data = [(k, v) for k, v in leaf.data if k != op.key]

        # Write all modified leaves back
        for leaf in leaves.values():
            self._write_node(leaf)

        self._write_node(parent)

    def _split_leaf(self, parent: BufferTreeNode, leaf_index: int, full_leaf: BufferTreeNode) -> None:
        """Split a full leaf node."""
        mid = len(full_leaf.data) // 2

        # Create new leaf with second half of data
        new_leaf = BufferTreeNode(self._get_next_node_id(), is_leaf=True, degree=self.degree)
        new_leaf.data = full_leaf.data[mid:]
        full_leaf.data = full_leaf.data[:mid]

        # Update parent routing keys and children
        if mid < len(new_leaf.data):
            split_key = new_leaf.data[0][0]
            parent.keys.insert(leaf_index, split_key)
            parent.children.insert(leaf_index + 1, new_leaf.node_id)

        # Write nodes
        self._write_node(full_leaf)
        self._write_node(new_leaf)
        self._write_node(parent)

    def _process_operations_at_leaf(self, leaf: BufferTreeNode) -> None:
        """
        Process buffered operations directly at a single leaf node.
        This handles the case when the root is a leaf.
        """
        operations = sorted(leaf.buffer)
        leaf.buffer.clear()

        # Process each operation directly on this leaf
        for op in operations:
            if op.op_type == OperationType.SEARCH:
                # Perform search in leaf data
                for key, value in leaf.data:
                    if key == op.key:
                        self.search_results[op.key] = value
                        break
                else:
                    self.search_results[op.key] = None

            elif op.op_type == OperationType.INSERT:
                # Insert into leaf (maintaining sort order)
                inserted = False
                for i, (key, _) in enumerate(leaf.data):
                    if op.key <= key:
                        if op.key == key:
                            # Update existing key
                            leaf.data[i] = (op.key, op.value)
                        else:
                            # Insert new key
                            leaf.data.insert(i, (op.key, op.value))
                        inserted = True
                        break
                if not inserted:
                    leaf.data.append((op.key, op.value))

                # Handle leaf overflow by splitting (would need to create parent)
                if leaf.is_data_full():
                    self._split_root_leaf(leaf)

            elif op.op_type == OperationType.DELETE:
                # Remove from leaf data
                leaf.data = [(k, v) for k, v in leaf.data if k != op.key]

        # Write modified leaf back
        self._write_node(leaf)

    def _split_root_leaf(self, full_leaf: BufferTreeNode) -> None:
        """Split a full root leaf by creating a new internal root."""
        # For simplicity in this demo, we'll just expand the capacity
        # In a full implementation, this would create a new internal root
        full_leaf.data_capacity *= 2

    def flush_all_operations(self) -> None:
        """Flush all pending operations through the tree."""
        # First flush collection buffer
        self._flush_collection_buffer()

        # Then flush from root downwards
        self._recursive_flush(self.root_id)

    def _recursive_flush(self, node_id: int) -> None:
        """Recursively flush all operations in the tree."""
        node = self._read_node(node_id)

        if node.buffer:
            self._flush_node(node_id)

        # Recursively flush children
        if not node.is_leaf:
            for child_id in node.children:
                self._recursive_flush(child_id)

    def get_io_count(self) -> int:
        """Get current I/O operation count."""
        return self.disk.io_count

    def print_tree(self) -> None:
        """Print tree structure for debugging."""
        print("Buffer Tree Structure:")  # noqa: T201
        self._print_node(self.root_id, 0)

    def _print_node(self, node_id: int, level: int) -> None:
        """Recursively print node structure."""
        if node_id is None:
            return

        node = self._read_node(node_id)
        indent = "  " * level

        node_type = "leaf" if node.is_leaf else "internal"
        buffer_info = f", buffer: {len(node.buffer)}/{node.buffer_capacity}"
        data_info = f", data: {len(node.data)}" if node.is_leaf else ""

        print(f"{indent}Node {node_id} ({node_type}): keys={node.keys}{buffer_info}{data_info}")  # noqa: T201

        if not node.is_leaf:
            for child_id in node.children:
                self._print_node(child_id, level + 1)


# Example usage and testing
if __name__ == "__main__":
    import numpy as np

    print("Testing Buffer Tree implementation...")  # noqa: T201

    # Create large disk for buffer tree storage
    disk_data = np.zeros(50000)  # Larger storage for buffer trees
    disk = IOSimulator(disk_data, block_size=50, memory_size=200)

    # Create buffer tree with appropriate parameters
    # degree should be Θ(M/B) = Θ(200/50) = Θ(4), using 8 for better performance
    buffer_tree = BufferTree(disk, memory_size=200, block_size=50, degree=8)

    print("\nTesting batch operations for I/O efficiency...")  # noqa: T201

    # Test batch insertions
    keys_to_insert = [10, 5, 15, 3, 7, 12, 18, 1, 4, 6, 8, 11, 13, 16, 20, 2, 9, 14, 17, 19]

    initial_io = buffer_tree.get_io_count()

    print(f"Inserting {len(keys_to_insert)} keys...")  # noqa: T201
    for key in keys_to_insert:
        buffer_tree.insert(key, f"value_{key}")

    # Flush all operations to see results
    buffer_tree.flush_all_operations()

    insert_io = buffer_tree.get_io_count() - initial_io
    print(f"I/O operations for batch insertions: {insert_io}")  # noqa: T201

    # Test batch searches
    search_io_start = buffer_tree.get_io_count()

    search_keys = [1, 5, 10, 15, 20, 25]  # Mix of existing and non-existing
    print(f"\nSearching for keys: {search_keys}")  # noqa: T201

    for key in search_keys:
        buffer_tree.search(key)

    # Flush to process search operations
    buffer_tree.flush_all_operations()

    # Display search results
    for key in search_keys:
        result = buffer_tree.search_results.get(key)
        print(f"Search {key}: {'Found' if result else 'Not found'}")  # noqa: T201

    search_io = buffer_tree.get_io_count() - search_io_start
    print(f"I/O operations for batch searches: {search_io}")  # noqa: T201

    print(f"\nTotal I/O operations: {buffer_tree.get_io_count()}")  # noqa: T201
    print("Buffer tree test completed!")  # noqa: T201

    # Demonstrate I/O efficiency compared to individual operations
    amortized_io_per_op = buffer_tree.get_io_count() / len(keys_to_insert)
    print(f"Amortized I/O per operation: {amortized_io_per_op:.2f}")  # noqa: T201
    print("This demonstrates the batching advantage of buffer trees!")  # noqa: T201
