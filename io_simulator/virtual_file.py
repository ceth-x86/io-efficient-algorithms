from .io_simulator import IOSimulator

class VirtualFile:
    """
    Represents a contiguous file of fixed-size records on the virtual disk.
    """
    def __init__(self, sim: IOSimulator, size: int, record_size: int):
        self.sim = sim
        self.size = size
        self.record_size = record_size
        self.start_idx = sim.disk.allocate(size * record_size)

    def close(self):
        if self.start_idx is not None:
            self.sim.disk.free(self.start_idx, self.size * self.record_size)
            self.start_idx = None

    def read_record(self, rec_idx: int) -> list:
        if self.start_idx is None:
            raise ValueError("File is closed!")
        rec = []
        for i in range(self.record_size):
            flat_idx = self.start_idx + rec_idx * self.record_size + i
            rec.append(self.sim.read_element(flat_idx))
        return rec

    def write_record(self, rec_idx: int, record: list) -> None:
        if self.start_idx is None:
            raise ValueError("File is closed!")
        for i in range(self.record_size):
            flat_idx = self.start_idx + rec_idx * self.record_size + i
            self.sim.write_element(flat_idx, record[i])
