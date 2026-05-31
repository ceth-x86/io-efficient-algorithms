from collections import OrderedDict
from .virtual_disk import VirtualDisk

class IOSimulator:
    """
    Simulates block-based disk I/O operations with an LRU memory cache.
    """
    def __init__(self, virtual_disk: VirtualDisk, block_size: int, memory_size: int):
        self.disk = virtual_disk
        self.total_size = len(virtual_disk.disk)
        self.block_size = block_size
        self.memory_size = memory_size
        self.memory_limit = max(1, memory_size // block_size)

        self.io_count = 0
        self.memory = OrderedDict()  # LRU cache: block_id -> block_data (list)
        self.dirty_blocks = set()

    def _get_block_id(self, flat_index: int) -> int:
        return flat_index // self.block_size

    def _get_index_in_block(self, flat_index: int) -> int:
        return flat_index % self.block_size

    def _read_block(self, block_id: int) -> list:
        if block_id in self.memory:
            self.memory.move_to_end(block_id)
            return self.memory[block_id]

        if len(self.memory) >= self.memory_limit:
            self._evict_lru_block()

        block = self.disk.read_block(block_id, self.block_size)
        self.memory[block_id] = block
        self.io_count += 1
        return block

    def _evict_lru_block(self) -> None:
        if not self.memory:
            return
        lru_block_id, lru_block_data = self.memory.popitem(last=False)
        if lru_block_id in self.dirty_blocks:
            self.disk.write_block(lru_block_id, self.block_size, lru_block_data)
            self.dirty_blocks.remove(lru_block_id)
            self.io_count += 1

    def _write_block(self, block_id: int) -> None:
        if block_id not in self.memory:
            return
        block_data = self.memory[block_id]
        self.disk.write_block(block_id, self.block_size, block_data)
        self.io_count += 1

    def read_element(self, flat_index: int) -> int:
        block_id = self._get_block_id(flat_index)
        block = self._read_block(block_id)
        index_in_block = self._get_index_in_block(flat_index)
        if index_in_block >= len(block):
            return 0
        return block[index_in_block]

    def write_element(self, flat_index: int, value: int) -> None:
        block_id = self._get_block_id(flat_index)
        block = self._read_block(block_id)
        index_in_block = self._get_index_in_block(flat_index)
        block[index_in_block] = value
        self.memory[block_id] = block
        self.dirty_blocks.add(block_id)
        self.memory.move_to_end(block_id)

    def flush_memory(self) -> None:
        for block_id in list(self.memory.keys()):
            self._write_block(block_id)
        self.memory.clear()
        self.dirty_blocks.clear()
