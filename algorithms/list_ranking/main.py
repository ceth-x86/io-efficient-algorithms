import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from io_simulator import IOSimulator, VirtualDisk, VirtualFile

if __package__ is None or __package__ == "":
    # Direct execution: e.g. python3 main.py (from within directory)
    from list_ranking import list_ranking
else:
    # Module execution: e.g. python3 -m algorithms.list_ranking.main (from root)
    from .list_ranking import list_ranking


def generate_random_linked_list(N):
    """
    Generates a linked list of size N in memory.
    Each node gets a unique random ID.
    The output records are shuffled to simulate disk disorder.
    """
    ids = list(range(1, N + 1))
    random.shuffle(ids)

    records = []
    for i in range(N):
        curr_id = ids[i]
        next_id = ids[i + 1] if i < N - 1 else -1
        weight = random.randint(1, 10)
        records.append([curr_id, next_id, weight])

    random.shuffle(records)

    # Ground truth traversal
    adj = {r[0]: (r[1], r[2]) for r in records}
    destinations = {r[1] for r in records if r[1] != -1}
    head = [node for node in adj if node not in destinations][0]

    expected_ranks = {}
    curr = head
    curr_rank = 0
    while curr != -1:
        expected_ranks[curr] = curr_rank
        next_id, w = adj[curr]
        curr_rank += w
        curr = next_id

    return records, expected_ranks


def verify_results(vf_out, expected_ranks):
    calculated_ranks = {}
    for i in range(vf_out.size):
        rec = vf_out.read_record(i)
        calculated_ranks[rec[0]] = rec[1]

    if len(calculated_ranks) != len(expected_ranks):
        print(
            f"Error: Size mismatch! Expected {len(expected_ranks)}, got {len(calculated_ranks)}"
        )
        return False

    for node, r in expected_ranks.items():
        if node not in calculated_ranks:
            print(f"Error: Node {node} is missing in results!")
            return False
        if calculated_ranks[node] != r:
            print(
                f"Error: Rank mismatch for node {node}! Expected {r}, got {calculated_ranks[node]}"
            )
            return False

    return True


def main():
    N = 1000  # Number of nodes
    M_records = 50  # Memory size in records

    # Define sizes in simulator elements (integers)
    # Each record has size 3 (node_id, next_id, weight)
    record_size = 3
    block_size_elements = 9  # B = 3 records per block
    memory_size_elements = M_records * record_size  # M = 150 elements

    print(f"Generating random linked list of size N={N}...")
    records, expected_ranks = generate_random_linked_list(N)

    # Initialize Virtual Disk and Simulator
    vd = VirtualDisk(size=5 * 10**6)
    sim = IOSimulator(
        vd, block_size=block_size_elements, cache_memory_size=memory_size_elements
    )

    print("Writing input list to virtual disk...")
    vf_in = VirtualFile(sim, N, record_size)
    for i, r in enumerate(records):
        vf_in.write_record(i, r)

    # Flush memory cache and reset I/O stats to measure the algorithm only
    sim.flush_memory()
    sim.io_count = 0

    print(
        f"Running external memory list ranking in IO Simulator (M={M_records} records)..."
    )
    vf_out = list_ranking(sim, vd, vf_in, M=M_records)

    # Flush remaining writes
    sim.flush_memory()

    print("List ranking finished.")
    print(f"Total Block I/O Operations: {sim.io_count}")

    print("Verifying results...")
    # Verify results does read operations, we don't count them towards algorithm complexity
    if verify_results(vf_out, expected_ranks):
        print(
            "Success! Ranks calculated by external memory algorithm in IO Simulator match ground truth."
        )
    else:
        print("Failure: Ranks do not match.")

    # Clean up virtual files
    vf_in.close()
    vf_out.close()


if __name__ == "__main__":
    main()
