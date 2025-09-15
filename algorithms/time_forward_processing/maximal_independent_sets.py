"""
Maximal Independent Sets using Time-Forward Processing

Implementation of I/O-efficient algorithm for computing maximal independent sets
on undirected graphs using external memory priority queues and time-forward processing.

Algorithm Overview:
1. Convert undirected graph to DAG by directing edges from lower to higher index
2. Use time-forward processing with priority queue to compute local function values
3. Local function f(v) determines if vertex v is in the maximal independent set

I/O Complexity: O(sort(V+E)) = O((V+E)/B * log_{M/B}((V+E)/B))
"""

import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent.parent))

from algorithms.searching.priority_queue.priority_queue import ExternalPriorityQueue
from io_simulator.io_simulator import IOSimulator


class Graph:
    """
    Graph representation for external memory processing.

    Stores undirected graph as adjacency lists with vertex indexing.
    Supports conversion to DAG by directing edges from lower to higher indices.
    """

    def __init__(self, num_vertices: int) -> None:
        self.num_vertices = num_vertices
        self.adjacency_list: dict[int, list[int]] = {i: [] for i in range(num_vertices)}
        self.vertex_labels: dict[int, Any] = {}

    def add_edge(self, u: int, v: int) -> None:
        """Add undirected edge between vertices u and v."""
        if u != v:  # No self-loops
            self.adjacency_list[u].append(v)
            self.adjacency_list[v].append(u)

    def set_vertex_label(self, v: int, label: Any) -> None:
        """Set label for vertex v."""
        self.vertex_labels[v] = label

    def get_outgoing_neighbors(self, v: int) -> list[int]:
        """Get outgoing neighbors in DAG representation (neighbors with higher index)."""
        return [u for u in self.adjacency_list[v] if u > v]

    def get_incoming_neighbors(self, v: int) -> list[int]:
        """Get incoming neighbors in DAG representation (neighbors with lower index)."""
        return [u for u in self.adjacency_list[v] if u < v]

    def get_all_neighbors(self, v: int) -> list[int]:
        """Get all neighbors of vertex v."""
        return self.adjacency_list[v]


class TimeForwardProcessor:
    """
    Time-forward processing engine for computing local functions on DAG.

    Uses external memory priority queue to efficiently process vertices
    in topological order while forwarding computed values to future vertices.
    """

    def __init__(self, disk: IOSimulator, memory_size: int = 200, block_size: int = 50) -> None:
        self.disk = disk
        self.memory_size = memory_size
        self.block_size = block_size

        # External priority queue for time-forward processing
        self.priority_queue = ExternalPriorityQueue(disk, memory_size=memory_size, block_size=block_size, degree=8)

        # Results storage
        self.computed_values: dict[int, Any] = {}

    def process_dag(self, graph: Graph, local_function: callable) -> dict[int, Any]:
        """
        Process DAG using time-forward processing to compute local function values.

        Args:
            graph: Graph in DAG representation
            local_function: Function that computes f(v) given vertex v and incoming values

        Returns:
            Dictionary mapping vertices to their computed function values
        """
        # Process vertices in topological order (0, 1, 2, ..., n-1)
        for v in range(graph.num_vertices):
            # Extract all values forwarded to this vertex
            incoming_values = []
            while True:
                item = self.priority_queue.extract_min()
                if item is None:
                    break

                priority, value = item
                if priority == v:
                    # This value is for current vertex
                    incoming_values.append(value)
                elif priority > v:
                    # This value is for a future vertex, put it back
                    self.priority_queue.insert(priority, value)
                    break
                # priority < v should not happen in correct topological processing

            # Compute local function value
            vertex_label = graph.vertex_labels.get(v, None)
            f_value = local_function(v, vertex_label, incoming_values)
            self.computed_values[v] = f_value

            # Forward this value to all outgoing neighbors
            outgoing_neighbors = graph.get_outgoing_neighbors(v)
            for neighbor in outgoing_neighbors:
                # Only forward to neighbors that exist in the graph
                if neighbor < graph.num_vertices:
                    self.priority_queue.insert(neighbor, f_value)

        # Ensure all operations are flushed
        self.priority_queue.flush_all_operations()

        return self.computed_values

    def get_io_count(self) -> int:
        """Get total I/O operations performed."""
        return self.priority_queue.get_io_count()


class MaximalIndependentSetSolver:
    """
    Solver for maximal independent set problem using time-forward processing.

    Computes maximal independent set by:
    1. Converting undirected graph to DAG by vertex ordering
    2. Defining local function for independence property
    3. Using time-forward processing for I/O-efficient computation
    """

    def __init__(self, disk: IOSimulator, memory_size: int = 200, block_size: int = 50) -> None:
        self.processor = TimeForwardProcessor(disk, memory_size, block_size)

    def solve(self, graph: Graph) -> tuple[set[int], dict[int, int]]:
        """
        Compute maximal independent set for given graph.

        Args:
            graph: Undirected graph

        Returns:
            Tuple of (independent_set, vertex_values) where:
            - independent_set: Set of vertices in maximal independent set
            - vertex_values: Dictionary mapping each vertex to 0 or 1
        """

        # Define local function for maximal independent set
        def mis_local_function(_vertex: int, _label: Any, incoming_values: list[int]) -> int:
            """
            Local function for maximal independent set computation.

            Args:
                _vertex: Current vertex index (unused)
                _label: Vertex label (unused for MIS)
                incoming_values: List of f-values from incoming neighbors

            Returns:
                1 if vertex is in independent set, 0 otherwise
            """
            # If no incoming neighbors, include vertex in set
            if not incoming_values:
                return 1

            # If any incoming neighbor is included (value = 1), exclude this vertex
            if any(value == 1 for value in incoming_values):
                return 0

            # If all incoming neighbors are excluded (value = 0), include this vertex
            return 1

        # Process graph using time-forward processing
        vertex_values = self.processor.process_dag(graph, mis_local_function)

        # Extract independent set (vertices with value 1)
        # Only include vertices that actually exist in the graph
        independent_set = {v for v, value in vertex_values.items() if value == 1 and v < graph.num_vertices}

        return independent_set, vertex_values

    def get_io_count(self) -> int:
        """Get total I/O operations performed."""
        return self.processor.get_io_count()

    def verify_independence(self, graph: Graph, independent_set: set[int]) -> bool:
        """
        Verify that the given set is indeed independent.

        Args:
            graph: Original graph
            independent_set: Set of vertices to verify

        Returns:
            True if set is independent, False otherwise
        """
        for v in independent_set:
            # Check if vertex exists in graph
            if v >= graph.num_vertices:
                return False  # Invalid vertex

            neighbors = graph.get_all_neighbors(v)
            for neighbor in neighbors:
                if neighbor in independent_set:
                    return False  # Found edge within independent set
        return True

    def verify_maximality(self, graph: Graph, independent_set: set[int]) -> bool:
        """
        Verify that the independent set is maximal (cannot be extended).

        Args:
            graph: Original graph
            independent_set: Set of vertices to verify

        Returns:
            True if set is maximal, False otherwise
        """
        # For each vertex not in the set, check if it can be added
        for v in range(graph.num_vertices):
            if v not in independent_set:
                # Check if v is connected to any vertex in the independent set
                neighbors = graph.get_all_neighbors(v)
                if not any(neighbor in independent_set for neighbor in neighbors):
                    return False  # Found vertex that can be added
        return True


# Example usage and testing
if __name__ == "__main__":
    import random

    import numpy as np

    print("Testing Maximal Independent Set with Time-Forward Processing...")  # noqa: T201

    # Create disk simulator for external memory
    disk_data = np.zeros(50000)
    disk = IOSimulator(disk_data, block_size=50, memory_size=200)

    # Create MIS solver
    solver = MaximalIndependentSetSolver(disk, memory_size=200, block_size=50)

    print("\nTest 1: Simple path graph")
    # Path: 0-1-2-3-4
    graph1 = Graph(5)
    for i in range(4):
        graph1.add_edge(i, i + 1)

    initial_io = solver.get_io_count()
    independent_set1, values1 = solver.solve(graph1)
    io_used = solver.get_io_count() - initial_io

    print("Graph: Path 0-1-2-3-4")
    print(f"Independent set: {sorted(independent_set1)}")
    print(f"Vertex values: {values1}")
    print(f"I/O operations: {io_used}")
    print(f"Is independent: {solver.verify_independence(graph1, independent_set1)}")
    print(f"Is maximal: {solver.verify_maximality(graph1, independent_set1)}")

    print("\nTest 2: Complete graph K4")
    # Complete graph on 4 vertices
    graph2 = Graph(4)
    for i in range(4):
        for j in range(i + 1, 4):
            graph2.add_edge(i, j)

    independent_set2, values2 = solver.solve(graph2)

    print("Graph: Complete K4")
    print(f"Independent set: {sorted(independent_set2)}")
    print(f"Vertex values: {values2}")
    print(f"Is independent: {solver.verify_independence(graph2, independent_set2)}")
    print(f"Is maximal: {solver.verify_maximality(graph2, independent_set2)}")

    print("\nTest 3: Star graph")
    # Star: center vertex 0 connected to 1,2,3,4
    graph3 = Graph(5)
    for i in range(1, 5):
        graph3.add_edge(0, i)

    independent_set3, values3 = solver.solve(graph3)

    print("Graph: Star with center 0")
    print(f"Independent set: {sorted(independent_set3)}")
    print(f"Vertex values: {values3}")
    print(f"Is independent: {solver.verify_independence(graph3, independent_set3)}")
    print(f"Is maximal: {solver.verify_maximality(graph3, independent_set3)}")

    print("\nTest 4: Random graph")
    # Random graph with 10 vertices
    random.seed(42)
    graph4 = Graph(10)

    # Add random edges
    for _ in range(15):
        u = random.randint(0, 9)
        v = random.randint(0, 9)
        if u != v:
            graph4.add_edge(u, v)

    initial_io = solver.get_io_count()
    independent_set4, values4 = solver.solve(graph4)
    io_used = solver.get_io_count() - initial_io

    print("Graph: Random 10 vertices, ~15 edges")
    print(f"Independent set: {sorted(independent_set4)}")
    print(f"Independent set size: {len(independent_set4)}")
    print(f"I/O operations: {io_used}")
    print(f"Is independent: {solver.verify_independence(graph4, independent_set4)}")
    print(f"Is maximal: {solver.verify_maximality(graph4, independent_set4)}")

    print("\nTest 5: Large cycle")
    # Cycle: 0-1-2-...-n-1-0
    n = 20
    graph5 = Graph(n)
    for i in range(n):
        graph5.add_edge(i, (i + 1) % n)

    initial_io = solver.get_io_count()
    independent_set5, values5 = solver.solve(graph5)
    io_used = solver.get_io_count() - initial_io

    print(f"Graph: Cycle with {n} vertices")
    print(f"Independent set size: {len(independent_set5)}")
    print(f"Expected size: ~{n // 2} (for cycle)")
    print(f"I/O operations: {io_used}")
    print(f"I/O per vertex+edge: {io_used / (n + n):.3f}")
    print(f"Is independent: {solver.verify_independence(graph5, independent_set5)}")
    print(f"Is maximal: {solver.verify_maximality(graph5, independent_set5)}")

    print(f"\nTotal I/O operations across all tests: {solver.get_io_count()}")
    print("Time-forward processing demonstrates I/O efficiency for graph algorithms!")
