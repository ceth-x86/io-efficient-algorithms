import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from io_simulator.io_simulator import IOSimulator


class BTreeNode:
    """
    B-tree node for external memory storage.

    Each node stores:
    - keys: List of keys in sorted order
    - children: List of child node IDs (for internal nodes)
    - is_leaf: Whether this is a leaf node
    - node_id: Unique identifier for this node
    """

    def __init__(self, node_id: int, is_leaf: bool = False) -> None:
        self.node_id = node_id
        self.keys: list[Any] = []
        self.children: list[int] = []  # Store node IDs, not actual nodes
        self.is_leaf = is_leaf

    def is_full(self, max_keys: int) -> bool:
        """Check if node is full (has maximum allowed keys)."""
        return len(self.keys) >= max_keys

    def is_minimal(self, min_keys: int) -> bool:
        """Check if node has minimum required keys."""
        return len(self.keys) <= min_keys

    def to_dict(self) -> dict:
        """Convert node to dictionary for serialization."""
        return {"node_id": self.node_id, "keys": self.keys, "children": self.children, "is_leaf": self.is_leaf}

    @classmethod
    def from_dict(cls, data: dict) -> "BTreeNode":
        """Create node from dictionary."""
        node = cls(data["node_id"], data["is_leaf"])
        node.keys = data["keys"]
        node.children = data["children"]
        return node


class BTree:
    """
    B-tree implementation optimized for external memory access.

    Key properties:
    - All leaves are at the same level
    - Each internal node has between d_min and 2*d_min children
    - Each node has between d_min-1 and 2*d_min-1 keys
    - All node data fits in a single disk block
    - Operations achieve O(log_B n) I/O complexity

    Parameters:
        disk: IOSimulator instance for external storage
        d_min: Minimum degree (approximately B/4 where B is block size)
    """

    def __init__(self, disk: IOSimulator, d_min: int = 2) -> None:
        self.disk = disk
        self.d_min = d_min
        self.max_keys = 2 * d_min - 1
        self.min_keys = d_min - 1

        # Node management
        self.next_node_id = 0
        self.root_id = None
        self.node_cache = {}  # Simple cache for frequently accessed nodes

        # Initialize with empty root
        self._create_empty_tree()

    def _create_empty_tree(self) -> None:
        """Create an empty B-tree with a single leaf root."""
        root = BTreeNode(self._get_next_node_id(), is_leaf=True)
        self.root_id = root.node_id
        self._write_node(root)

    def _get_next_node_id(self) -> int:
        """Get next available node ID."""
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id

    def _write_node(self, node: BTreeNode) -> None:
        """Write node to external storage."""
        # Serialize node to a flat array format for IOSimulator
        # Using a simple encoding: [node_id, is_leaf, num_keys, keys..., num_children, children...]
        data = [node.node_id, int(node.is_leaf), len(node.keys)]
        data.extend(node.keys)
        data.append(len(node.children))
        data.extend(node.children)

        # Write to disk at position based on node_id
        start_pos = node.node_id * 100  # Reserve 100 positions per node
        for i, value in enumerate(data):
            if start_pos + i < self.disk.total_size:
                self.disk.set_element(0, start_pos + i, float(value), self.disk.total_size)

        # Cache the node
        self.node_cache[node.node_id] = node

    def _read_node(self, node_id: int) -> BTreeNode:
        """Read node from external storage."""
        # Check cache first
        if node_id in self.node_cache:
            return self.node_cache[node_id]

        # Read from disk
        start_pos = node_id * 100

        # Read basic info
        stored_node_id = int(self.disk.get_element(0, start_pos, self.disk.total_size))
        is_leaf = bool(int(self.disk.get_element(0, start_pos + 1, self.disk.total_size)))
        num_keys = int(self.disk.get_element(0, start_pos + 2, self.disk.total_size))

        # Create node
        node = BTreeNode(stored_node_id, is_leaf)

        # Read keys
        for i in range(num_keys):
            key = self.disk.get_element(0, start_pos + 3 + i, self.disk.total_size)
            node.keys.append(int(key))  # Assuming integer keys for simplicity

        # Read children count and children
        children_pos = start_pos + 3 + num_keys
        num_children = int(self.disk.get_element(0, children_pos, self.disk.total_size))

        for i in range(num_children):
            child_id = int(self.disk.get_element(0, children_pos + 1 + i, self.disk.total_size))
            node.children.append(child_id)

        # Cache the node
        self.node_cache[node_id] = node
        return node

    def search(self, key: Any) -> bool:
        """
        Search for a key in the B-tree.

        Returns True if key is found, False otherwise.
        Achieves O(log_B n) I/O complexity.
        """
        return self._search_recursive(self.root_id, key) is not None

    def _search_recursive(self, node_id: int, key: Any) -> int | None:
        """Recursive search implementation."""
        node = self._read_node(node_id)

        # Find position where key could be
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        # Check if key is found
        if i < len(node.keys) and key == node.keys[i]:
            return node_id

        # If leaf and not found, key doesn't exist
        if node.is_leaf:
            return None

        # Continue search in appropriate child
        return self._search_recursive(node.children[i], key)

    def insert(self, key: Any) -> None:
        """
        Insert a key into the B-tree.

        Maintains B-tree properties and achieves O(log_B n) I/O complexity.
        """
        root = self._read_node(self.root_id)

        # If root is full, create new root
        if root.is_full(self.max_keys):
            new_root = BTreeNode(self._get_next_node_id())
            new_root.children.append(self.root_id)
            self.root_id = new_root.node_id
            self._split_child(new_root, 0)
            self._write_node(new_root)

        # Insert into non-full root
        self._insert_non_full(self.root_id, key)

    def _insert_non_full(self, node_id: int, key: Any) -> None:
        """Insert key into a non-full node."""
        node = self._read_node(node_id)
        i = len(node.keys) - 1

        if node.is_leaf:
            # Insert into leaf
            node.keys.append(None)  # Make space
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
            self._write_node(node)
        else:
            # Find child to insert into
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1

            child = self._read_node(node.children[i])
            if child.is_full(self.max_keys):
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1

            self._insert_non_full(node.children[i], key)

    def _split_child(self, parent: BTreeNode, child_index: int) -> None:
        """Split a full child node."""
        full_child = self._read_node(parent.children[child_index])
        new_child = BTreeNode(self._get_next_node_id(), full_child.is_leaf)

        mid_index = self.d_min - 1

        # Save the middle key before modifying the arrays
        middle_key = full_child.keys[mid_index]

        # Move keys to new node
        new_child.keys = full_child.keys[mid_index + 1 :].copy()
        full_child.keys = full_child.keys[:mid_index].copy()

        # Move children if internal node
        if not full_child.is_leaf:
            new_child.children = full_child.children[mid_index + 1 :].copy()
            full_child.children = full_child.children[: mid_index + 1].copy()

        # Insert new child into parent
        parent.children.insert(child_index + 1, new_child.node_id)
        parent.keys.insert(child_index, middle_key)

        # Write updated nodes
        self._write_node(full_child)
        self._write_node(new_child)
        self._write_node(parent)

    def delete(self, key: Any) -> None:
        """
        Delete a key from the B-tree.

        Maintains B-tree properties and achieves O(log_B n) I/O complexity.
        """
        self._delete_recursive(self.root_id, key)

        # If root becomes empty, make its only child the new root
        root = self._read_node(self.root_id)
        if len(root.keys) == 0 and not root.is_leaf:
            self.root_id = root.children[0]

    def _delete_recursive(self, node_id: int, key: Any) -> None:
        """Recursive deletion implementation."""
        node = self._read_node(node_id)

        # Find key position
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            # Key found in this node
            if node.is_leaf:
                # Simple deletion from leaf
                node.keys.pop(i)
                self._write_node(node)
            else:
                # Key in internal node - more complex
                self._delete_from_internal_node(node, i, key)
        else:
            # Key not in this node
            if node.is_leaf:
                # Key not found
                return

            # Find child that could contain key
            child = self._read_node(node.children[i])
            if child.is_minimal(self.min_keys):
                self._fix_minimal_child(node, i)
                # Retry after fixing
                self._delete_recursive(node_id, key)
            else:
                self._delete_recursive(node.children[i], key)

    def _delete_from_internal_node(self, node: BTreeNode, key_index: int, key: Any) -> None:
        """Delete key from internal node."""
        left_child = self._read_node(node.children[key_index])
        right_child = self._read_node(node.children[key_index + 1])

        if not left_child.is_minimal(self.min_keys):
            # Replace with predecessor
            predecessor = self._get_predecessor(left_child)
            node.keys[key_index] = predecessor
            self._write_node(node)
            self._delete_recursive(left_child.node_id, predecessor)
        elif not right_child.is_minimal(self.min_keys):
            # Replace with successor
            successor = self._get_successor(right_child)
            node.keys[key_index] = successor
            self._write_node(node)
            self._delete_recursive(right_child.node_id, successor)
        else:
            # Merge children
            self._merge_children(node, key_index)
            self._delete_recursive(left_child.node_id, key)

    def _get_predecessor(self, node: BTreeNode) -> Any:
        """Get the predecessor key (rightmost key in subtree)."""
        while not node.is_leaf:
            node = self._read_node(node.children[-1])
        return node.keys[-1]

    def _get_successor(self, node: BTreeNode) -> Any:
        """Get the successor key (leftmost key in subtree)."""
        while not node.is_leaf:
            node = self._read_node(node.children[0])
        return node.keys[0]

    def _fix_minimal_child(self, parent: BTreeNode, child_index: int) -> None:
        """Fix a child with minimal keys by borrowing or merging."""
        # Read child to check its state, but we don't need to store it
        self._read_node(parent.children[child_index])

        # Try to borrow from left sibling
        if child_index > 0:
            left_sibling = self._read_node(parent.children[child_index - 1])
            if not left_sibling.is_minimal(self.min_keys):
                self._borrow_from_left_sibling(parent, child_index)
                return

        # Try to borrow from right sibling
        if child_index < len(parent.children) - 1:
            right_sibling = self._read_node(parent.children[child_index + 1])
            if not right_sibling.is_minimal(self.min_keys):
                self._borrow_from_right_sibling(parent, child_index)
                return

        # Must merge with a sibling
        if child_index > 0:
            self._merge_children(parent, child_index - 1)
        else:
            self._merge_children(parent, child_index)

    def _borrow_from_left_sibling(self, parent: BTreeNode, child_index: int) -> None:
        """Borrow a key from left sibling."""
        child = self._read_node(parent.children[child_index])
        left_sibling = self._read_node(parent.children[child_index - 1])

        # Move parent key down to child
        child.keys.insert(0, parent.keys[child_index - 1])

        # Move left sibling's largest key to parent
        parent.keys[child_index - 1] = left_sibling.keys.pop()

        # Move child pointer if internal node
        if not child.is_leaf:
            child.children.insert(0, left_sibling.children.pop())

        self._write_node(child)
        self._write_node(left_sibling)
        self._write_node(parent)

    def _borrow_from_right_sibling(self, parent: BTreeNode, child_index: int) -> None:
        """Borrow a key from right sibling."""
        child = self._read_node(parent.children[child_index])
        right_sibling = self._read_node(parent.children[child_index + 1])

        # Move parent key down to child
        child.keys.append(parent.keys[child_index])

        # Move right sibling's smallest key to parent
        parent.keys[child_index] = right_sibling.keys.pop(0)

        # Move child pointer if internal node
        if not child.is_leaf:
            child.children.append(right_sibling.children.pop(0))

        self._write_node(child)
        self._write_node(right_sibling)
        self._write_node(parent)

    def _merge_children(self, parent: BTreeNode, left_index: int) -> None:
        """Merge two adjacent children."""
        left_child = self._read_node(parent.children[left_index])
        right_child = self._read_node(parent.children[left_index + 1])

        # Move parent key and right child's keys to left child
        left_child.keys.append(parent.keys[left_index])
        left_child.keys.extend(right_child.keys)

        # Move right child's children to left child
        if not left_child.is_leaf:
            left_child.children.extend(right_child.children)

        # Remove key and child from parent
        parent.keys.pop(left_index)
        parent.children.pop(left_index + 1)

        self._write_node(left_child)
        self._write_node(parent)

    def find_min(self) -> Any | None:
        """Find minimum key in the tree. O(log_B n) I/O complexity."""
        if self.root_id is None:
            return None

        node = self._read_node(self.root_id)
        while not node.is_leaf:
            node = self._read_node(node.children[0])

        return node.keys[0] if node.keys else None

    def find_max(self) -> Any | None:
        """Find maximum key in the tree. O(log_B n) I/O complexity."""
        if self.root_id is None:
            return None

        node = self._read_node(self.root_id)
        while not node.is_leaf:
            node = self._read_node(node.children[-1])

        return node.keys[-1] if node.keys else None

    def get_io_count(self) -> int:
        """Get current I/O operation count from disk simulator."""
        return self.disk.io_count

    def print_tree(self) -> None:
        """Print tree structure for debugging."""
        print("B-Tree Structure:")  # noqa: T201
        self._print_node(self.root_id, 0)

    def _print_node(self, node_id: int, level: int) -> None:
        """Recursively print node structure."""
        if node_id is None:
            return

        node = self._read_node(node_id)
        indent = "  " * level

        print(f"{indent}Node {node_id}: {node.keys} {'(leaf)' if node.is_leaf else ''}")  # noqa: T201

        if not node.is_leaf:
            for child_id in node.children:
                self._print_node(child_id, level + 1)


# Example usage and testing
if __name__ == "__main__":
    import numpy as np

    print("Testing B-tree implementation...")  # noqa: T201

    # Create a large enough disk for B-tree storage
    # Each node needs ~100 positions, so for testing we'll use 10000 positions
    disk_data = np.zeros(10000)
    disk = IOSimulator(disk_data, block_size=50, memory_size=200)

    # Create B-tree with d_min=3 (max 5 keys per node)
    btree = BTree(disk, d_min=3)

    print("\nInserting keys: 10, 20, 5, 6, 12, 30, 7, 17...")  # noqa: T201
    keys_to_insert = [10, 20, 5, 6, 12, 30, 7, 17]

    initial_io = btree.get_io_count()
    for key in keys_to_insert:
        btree.insert(key)
        print(f"Inserted {key}")  # noqa: T201

    insert_io = btree.get_io_count() - initial_io
    print(f"\nI/O operations for insertions: {insert_io}")  # noqa: T201

    btree.print_tree()

    print("\nTesting searches...")  # noqa: T201
    search_io_start = btree.get_io_count()

    test_keys = [10, 15, 5, 30, 100]
    for key in test_keys:
        found = btree.search(key)
        print(f"Search {key}: {'Found' if found else 'Not found'}")  # noqa: T201

    search_io = btree.get_io_count() - search_io_start
    print(f"I/O operations for searches: {search_io}")  # noqa: T201

    print(f"\nMin key: {btree.find_min()}")  # noqa: T201
    print(f"Max key: {btree.find_max()}")  # noqa: T201

    print(f"\nTotal I/O operations: {btree.get_io_count()}")  # noqa: T201
    print("B-tree test completed successfully!")  # noqa: T201
