from .io_simulator import IOSimulator

class BTreeNode:
    """
    Represents a node in the External Memory B+ Tree.
    """
    def __init__(self, block_id: int, is_leaf: bool = True):
        self.block_id = block_id
        self.is_leaf = is_leaf
        self.keys = []
        self.values = []  # child block_ids (internal) or values (leaf)


class DiskBTree:
    """
    An I/O-efficient B+ Tree implementation storing nodes as physical blocks on virtual disk.
    In B+ Tree, all keys and values are stored in leaves. Internal nodes contain only keys as splitters.
    """
    def __init__(self, sim: IOSimulator, root_block_id: int = None):
        self.sim = sim
        self.block_size = sim.block_size
        
        # Format of node layout in block:
        # data[0]: is_leaf (0 or 1)
        # data[1]: num_keys
        # data[2 ... 2+max_keys-1]: keys
        # data[2+max_keys ... 2+2*max_keys]: children/values
        # Total elements = 2*max_keys + 3 <= block_size
        self.max_keys = (self.block_size - 3) // 2
        if self.max_keys < 3:
            raise ValueError(f"Block size {self.block_size} is too small for B-Tree! Must be >= 9.")
            
        self.t = (self.max_keys + 1) // 2  # Minimum degree
        self.root_block_id = root_block_id

    def read_node(self, block_id: int) -> BTreeNode:
        flat_start = block_id * self.block_size
        is_leaf = self.sim.read_element(flat_start) == 1
        num_keys = self.sim.read_element(flat_start + 1)
        
        node = BTreeNode(block_id, is_leaf)
        
        # Read keys
        for i in range(num_keys):
            node.keys.append(self.sim.read_element(flat_start + 2 + i))
            
        # Read values/children (num_keys + 1 elements)
        for i in range(num_keys + 1):
            node.values.append(self.sim.read_element(flat_start + 2 + self.max_keys + i))
            
        return node

    def write_node(self, node: BTreeNode) -> None:
        flat_start = node.block_id * self.block_size
        self.sim.write_element(flat_start, 1 if node.is_leaf else 0)
        self.sim.write_element(flat_start + 1, len(node.keys))
        
        # Write keys
        for i in range(self.max_keys):
            val = node.keys[i] if i < len(node.keys) else 0
            self.sim.write_element(flat_start + 2 + i, val)
            
        # Write values/children
        for i in range(self.max_keys + 1):
            val = node.values[i] if i < len(node.values) else 0
            self.sim.write_element(flat_start + 2 + self.max_keys + i, val)

    def search(self, key: int) -> tuple:
        """
        Searches for a key in the B+ Tree.
        Returns (node_id, val_idx, value) if found, else None.
        """
        if self.root_block_id is None:
            return None
        return self._search_rec(self.root_block_id, key)

    def _search_rec(self, block_id: int, key: int) -> tuple:
        node = self.read_node(block_id)
        if node.is_leaf:
            for i, k in enumerate(node.keys):
                if k == key:
                    return (block_id, i, node.values[i])
            return None
            
        # Internal node traversal: find index to go down
        i = 0
        while i < len(node.keys) and key >= node.keys[i]:
            i += 1
        return self._search_rec(node.values[i], key)

    def insert(self, key: int, value: int) -> None:
        """
        Inserts a key-value pair into the B+ Tree.
        """
        if self.root_block_id is None:
            # Allocate root node
            addr = self.sim.disk.allocate(self.block_size)
            self.root_block_id = addr // self.block_size
            root = BTreeNode(self.root_block_id, is_leaf=True)
            root.keys.append(key)
            root.values.append(value)
            self.write_node(root)
            return

        root = self.read_node(self.root_block_id)
        if len(root.keys) == self.max_keys:
            # Root is full, split root
            addr = self.sim.disk.allocate(self.block_size)
            new_root_id = addr // self.block_size
            new_root = BTreeNode(new_root_id, is_leaf=False)
            new_root.values.append(self.root_block_id)
            
            self._split_child(new_root, 0, root)
            self.root_block_id = new_root_id
            self._insert_non_full(new_root, key, value)
        else:
            self._insert_non_full(root, key, value)

    def _split_child(self, parent: BTreeNode, i: int, child: BTreeNode) -> None:
        """
        Splits a full child node of parent.
        """
        addr = self.sim.disk.allocate(self.block_size)
        z_id = addr // self.block_size
        z = BTreeNode(z_id, child.is_leaf)
        
        split_idx = self.t - 1
        
        if child.is_leaf:
            # B+ Tree leaf node split: split keys and values, copy up split key
            z.keys = child.keys[split_idx:]
            z.values = child.values[split_idx:]
            
            child_mid_key = z.keys[0]
            
            child.keys = child.keys[:split_idx]
            child.values = child.values[:split_idx]
        else:
            # Standard B-Tree internal node split: move split key up to parent
            z.keys = child.keys[self.t:]
            child_mid_key = child.keys[split_idx]
            child.keys = child.keys[:split_idx]
            
            z.values = child.values[self.t:]
            child.values = child.values[:self.t]
            
        parent.values.insert(i + 1, z_id)
        parent.keys.insert(i, child_mid_key)
        
        self.write_node(child)
        self.write_node(z)
        self.write_node(parent)

    def _insert_non_full(self, node: BTreeNode, key: int, value: int) -> None:
        i = len(node.keys) - 1
        if node.is_leaf:
            # Find place to insert
            node.keys.append(0)
            node.values.append(0)
            while i >= 0 and key < node.keys[i]:
                node.keys[i+1] = node.keys[i]
                node.values[i+1] = node.values[i]
                i -= 1
            node.keys[i+1] = key
            node.values[i+1] = value
            self.write_node(node)
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            child = self.read_node(node.values[i])
            if len(child.keys) == self.max_keys:
                self._split_child(node, i, child)
                if key >= node.keys[i]:
                    i += 1
                child = self.read_node(node.values[i])
            self._insert_non_full(child, key, value)
