class VirtualDisk:
    """
    Simulates raw disk storage with a flat list and handles space allocation.
    """
    def __init__(self, size=5 * 10**6):
        self.disk = [0] * size
        self.free_segments = [(0, size)]  # list of (start, length)

    def allocate(self, length):
        for idx, (start, seg_len) in enumerate(self.free_segments):
            if seg_len >= length:
                self.free_segments[idx] = (start + length, seg_len - length)
                if self.free_segments[idx][1] == 0:
                    self.free_segments.pop(idx)
                return start
        raise MemoryError("Out of virtual disk space!")

    def free(self, start, length):
        self.free_segments.append((start, length))
        self.free_segments.sort()
        merged = []
        for s, l in self.free_segments:
            if not merged:
                merged.append((s, l))
            else:
                last_s, last_l = merged[-1]
                if last_s + last_l == s:
                    merged[-1] = (last_s, last_l + l)
                else:
                    merged.append((s, l))
        self.free_segments = merged

    def read_block(self, block_id, block_size):
        start = block_id * block_size
        end = start + block_size
        if start >= len(self.disk):
            return [0] * block_size
        return list(self.disk[start:end])

    def write_block(self, block_id, block_size, block_data):
        start = block_id * block_size
        end = start + block_size
        disk_slice_size = end - start
        if len(block_data) >= disk_slice_size:
            self.disk[start:end] = block_data[:disk_slice_size]
        else:
            self.disk[start : start + len(block_data)] = block_data
