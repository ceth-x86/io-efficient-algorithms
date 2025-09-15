import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.time_forward_processing.maximal_independent_sets import Graph
from algorithms.time_forward_processing.maximal_independent_sets import MaximalIndependentSetSolver
from algorithms.time_forward_processing.maximal_independent_sets import TimeForwardProcessor
from io_simulator.io_simulator import IOSimulator


class TestGraph:
    """Test cases for Graph representation."""

    def test_empty_graph(self):
        """Test empty graph creation."""
        graph = Graph(3)
        assert graph.num_vertices == 3
        assert len(graph.adjacency_list) == 3
        assert all(len(neighbors) == 0 for neighbors in graph.adjacency_list.values())

    def test_add_edges(self):
        """Test adding edges to graph."""
        graph = Graph(4)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)

        assert 1 in graph.adjacency_list[0]
        assert 0 in graph.adjacency_list[1]
        assert 2 in graph.adjacency_list[1]
        assert 1 in graph.adjacency_list[2]

    def test_dag_neighbors(self):
        """Test DAG neighbor extraction."""
        graph = Graph(4)
        graph.add_edge(0, 2)
        graph.add_edge(1, 3)
        graph.add_edge(0, 3)

        # Outgoing neighbors (higher index)
        assert set(graph.get_outgoing_neighbors(0)) == {2, 3}
        assert set(graph.get_outgoing_neighbors(1)) == {3}
        assert set(graph.get_outgoing_neighbors(2)) == set()
        assert set(graph.get_outgoing_neighbors(3)) == set()

        # Incoming neighbors (lower index)
        assert set(graph.get_incoming_neighbors(0)) == set()
        assert set(graph.get_incoming_neighbors(1)) == set()
        assert set(graph.get_incoming_neighbors(2)) == {0}
        assert set(graph.get_incoming_neighbors(3)) == {0, 1}

    def test_self_loops_ignored(self):
        """Test that self-loops are ignored."""
        graph = Graph(3)
        graph.add_edge(0, 0)  # Self-loop should be ignored
        graph.add_edge(1, 2)

        assert len(graph.adjacency_list[0]) == 0
        assert len(graph.adjacency_list[1]) == 1
        assert len(graph.adjacency_list[2]) == 1


class TestTimeForwardProcessor:
    """Test cases for TimeForwardProcessor."""

    @pytest.fixture
    def disk_simulator(self) -> IOSimulator:
        """Create disk simulator for testing."""
        disk_data = np.zeros(10000)
        return IOSimulator(disk_data, block_size=50, memory_size=200)

    @pytest.fixture
    def processor(self, disk_simulator: IOSimulator) -> TimeForwardProcessor:
        """Create time-forward processor for testing."""
        return TimeForwardProcessor(disk_simulator, memory_size=200, block_size=50)

    def test_simple_dag_processing(self, processor: TimeForwardProcessor):
        """Test processing simple DAG."""
        # Create simple DAG: 0 -> 1 -> 2
        graph = Graph(3)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)

        # Define simple local function: f(v) = sum of incoming values + 1
        def simple_function(vertex: int, label, incoming_values):
            return sum(incoming_values) + 1

        result = processor.process_dag(graph, simple_function)

        assert result[0] == 1  # No incoming neighbors: 0 + 1 = 1
        assert result[1] == 2  # One incoming neighbor with value 1: 1 + 1 = 2
        assert result[2] == 3  # One incoming neighbor with value 2: 2 + 1 = 3

    def test_empty_graph_processing(self, processor: TimeForwardProcessor):
        """Test processing empty graph."""
        graph = Graph(3)

        def constant_function(vertex: int, label, incoming_values):
            return 42

        result = processor.process_dag(graph, constant_function)

        assert all(value == 42 for value in result.values())
        assert len(result) == 3


class TestMaximalIndependentSetSolver:
    """Test cases for MaximalIndependentSetSolver."""

    @pytest.fixture
    def disk_simulator(self) -> IOSimulator:
        """Create disk simulator for testing."""
        disk_data = np.zeros(10000)
        return IOSimulator(disk_data, block_size=50, memory_size=200)

    @pytest.fixture
    def solver(self, disk_simulator: IOSimulator) -> MaximalIndependentSetSolver:
        """Create MIS solver for testing."""
        return MaximalIndependentSetSolver(disk_simulator, memory_size=200, block_size=50)

    def test_path_graph(self, solver: MaximalIndependentSetSolver):
        """Test MIS on path graph."""
        # Path: 0-1-2-3-4
        graph = Graph(5)
        for i in range(4):
            graph.add_edge(i, i + 1)

        independent_set, values = solver.solve(graph)

        # Expected: alternating vertices like {0, 2, 4}
        assert solver.verify_independence(graph, independent_set)
        assert solver.verify_maximality(graph, independent_set)
        assert len(independent_set) == 3  # For path of 5 vertices

    def test_complete_graph(self, solver: MaximalIndependentSetSolver):
        """Test MIS on complete graph."""
        # Complete graph K4
        graph = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                graph.add_edge(i, j)

        independent_set, values = solver.solve(graph)

        assert solver.verify_independence(graph, independent_set)
        assert solver.verify_maximality(graph, independent_set)
        assert len(independent_set) == 1  # Only one vertex can be selected

    def test_star_graph(self, solver: MaximalIndependentSetSolver):
        """Test MIS on star graph."""
        # Star: center 0 connected to leaves 1,2,3,4
        graph = Graph(5)
        for i in range(1, 5):
            graph.add_edge(0, i)

        independent_set, values = solver.solve(graph)

        assert solver.verify_independence(graph, independent_set)
        assert solver.verify_maximality(graph, independent_set)

        # Either center alone or all leaves (depending on algorithm ordering)
        assert len(independent_set) in [1, 4]

    def test_cycle_graph(self, solver: MaximalIndependentSetSolver):
        """Test MIS on cycle graph."""
        # Cycle: 0-1-2-3-0
        graph = Graph(4)
        for i in range(4):
            graph.add_edge(i, (i + 1) % 4)

        independent_set, values = solver.solve(graph)

        assert solver.verify_independence(graph, independent_set)
        assert solver.verify_maximality(graph, independent_set)
        assert len(independent_set) == 2  # For 4-cycle

    def test_disconnected_graph(self, solver: MaximalIndependentSetSolver):
        """Test MIS on disconnected graph."""
        # Two separate edges: 0-1 and 2-3
        graph = Graph(4)
        graph.add_edge(0, 1)
        graph.add_edge(2, 3)

        independent_set, values = solver.solve(graph)

        assert solver.verify_independence(graph, independent_set)
        assert solver.verify_maximality(graph, independent_set)
        assert len(independent_set) == 2  # One from each component

    def test_single_vertex(self, solver: MaximalIndependentSetSolver):
        """Test MIS on single vertex."""
        graph = Graph(1)

        independent_set, values = solver.solve(graph)

        assert independent_set == {0}
        assert solver.verify_independence(graph, independent_set)
        assert solver.verify_maximality(graph, independent_set)

    def test_empty_graph(self, solver: MaximalIndependentSetSolver):
        """Test MIS on graph with no edges."""
        graph = Graph(5)  # 5 isolated vertices

        independent_set, values = solver.solve(graph)

        assert independent_set == {0, 1, 2, 3, 4}  # All vertices
        assert solver.verify_independence(graph, independent_set)
        assert solver.verify_maximality(graph, independent_set)

    def test_triangle_graph(self, solver: MaximalIndependentSetSolver):
        """Test MIS on triangle graph."""
        # Triangle: 0-1-2-0
        graph = Graph(3)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 0)

        independent_set, values = solver.solve(graph)

        assert solver.verify_independence(graph, independent_set)
        assert solver.verify_maximality(graph, independent_set)
        assert len(independent_set) == 1  # Only one vertex in triangle

    def test_verification_methods(self, solver: MaximalIndependentSetSolver):
        """Test independence and maximality verification methods."""
        # Simple path: 0-1-2
        graph = Graph(3)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)

        # Independent set
        assert solver.verify_independence(graph, {0, 2})
        assert not solver.verify_independence(graph, {0, 1})  # Adjacent vertices

        # Maximal set
        assert solver.verify_maximality(graph, {0, 2})  # Cannot add vertex 1
        assert not solver.verify_maximality(graph, {0})  # Can add vertex 2

    def test_io_efficiency(self, solver: MaximalIndependentSetSolver):
        """Test I/O efficiency for larger graphs."""
        # Create larger cycle
        n = 20
        graph = Graph(n)
        for i in range(n):
            graph.add_edge(i, (i + 1) % n)

        initial_io = solver.get_io_count()
        independent_set, values = solver.solve(graph)
        io_used = solver.get_io_count() - initial_io

        assert solver.verify_independence(graph, independent_set)
        assert solver.verify_maximality(graph, independent_set)

        # I/O should be efficient
        total_operations = n + n  # vertices + edges
        amortized_io = io_used / total_operations if total_operations > 0 else 0
        assert amortized_io < 1.0, f"I/O should be efficient, got {amortized_io:.3f} per operation"

    @pytest.mark.parametrize("size", [5, 10, 15])
    def test_different_graph_sizes(self, solver: MaximalIndependentSetSolver, size: int):
        """Test MIS computation on different graph sizes."""
        # Create path graph of given size
        graph = Graph(size)
        for i in range(size - 1):
            graph.add_edge(i, i + 1)

        independent_set, values = solver.solve(graph)

        assert solver.verify_independence(graph, independent_set)
        assert solver.verify_maximality(graph, independent_set)

        # For path graph, independent set size should be ceil(size/2)
        expected_size = (size + 1) // 2
        assert len(independent_set) == expected_size
