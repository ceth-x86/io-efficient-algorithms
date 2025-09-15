# Makefile for transpose_cache_aware project

.PHONY: help test test-io test-transpose test-sorting test-btree test-buffer-tree test-priority-queue test-time-forward test-maximal-independent-sets example-transpose-cache-aware example-small example-large example-transpose-cache-oblivious example-sorting example-btree example-buffer-tree example-priority-queue example-time-forward example-maximal-independent-sets examples clean install

# Default target
help:
	@echo "Available targets:"
	@echo "  help        - Show this help message"
	@echo "  test        - Run all tests"
	@echo "  test-io     - Run IOSimulator tests only"
	@echo "  test-transpose - Run transpose tests only"
	@echo "  test-sorting - Run sorting tests only"
	@echo "  test-btree  - Run B-tree tests only"
	@echo "  test-buffer-tree - Run Buffer tree tests only"
	@echo "  test-priority-queue - Run Priority queue tests only"
	@echo "  test-time-forward - Run Time-forward processing tests only"
	@echo "  test-maximal-independent-sets - Run Maximal independent sets tests only"
	@echo "  test-cache-oblivious - Run cache-oblivious tests only"
	@echo "  example-transpose-cache-aware - Run the built-in transpose example (4x4 matrix)"
	@echo "  example-small - Run small matrix example (2x2)"
	@echo "  example-large - Run large matrix example (8x8)"
	@echo "  example-transpose-cache-oblivious - Run cache-oblivious algorithm example"
	@echo "  example-sorting - Run external memory sorting example"
	@echo "  example-btree - Run B-tree example"
	@echo "  example-buffer-tree - Run Buffer tree example"
	@echo "  example-priority-queue - Run Priority queue example"
	@echo "  example-time-forward - Run Time-forward processing example"
	@echo "  example-maximal-independent-sets - Run Maximal independent sets example"
	@echo "  examples    - Run all examples"
	@echo "  clean       - Clean up __pycache__ directories"
	@echo "  install     - Install dependencies"

# Run all tests
test:
	@echo "Running all tests..."
	pytest tests/ -v

# Run cache-oblivious tests only
test-cache-oblivious:
	@echo "Running cache-oblivious tests..."
	pytest tests/test_transpose_cache_oblivious.py -v

# Run cache-oblivious algorithm example
example-transpose-cache-oblivious:
	@echo "Running cache-oblivious algorithm example..."
	python algorithms/transpose/cache_oblivious.py

# Run IOSimulator tests only
test-io:
	@echo "Running IOSimulator tests..."
	pytest tests/test_io_simulator.py -v

# Run transpose tests only
test-transpose:
	@echo "Running transpose tests..."
	pytest tests/test_transpose_cache_aware.py tests/test_transpose_cache_oblivious.py -v

# Run sorting tests only
test-sorting:
	@echo "Running sorting tests..."
	pytest tests/test_external_merge_sort.py -v

# Run B-tree tests only
test-btree:
	@echo "Running B-tree tests..."
	pytest tests/test_btree.py -v

# Run Buffer tree tests only
test-buffer-tree:
	@echo "Running Buffer tree tests..."
	pytest tests/test_buffer_tree.py -v

# Run Priority queue tests only
test-priority-queue:
	@echo "Running Priority queue tests..."
	pytest tests/test_priority_queue.py -v

# Run Time-forward processing tests only
test-time-forward:
	@echo "Running Time-forward processing tests..."
	pytest tests/test_maximal_independent_sets.py -v

# Run Maximal independent sets tests only
test-maximal-independent-sets:
	@echo "Running Maximal independent sets tests..."
	pytest tests/test_maximal_independent_sets.py -v

# Run the transpose example (built-in example in transpose_cache_aware.py)
example-transpose-cache-aware:
	@echo "Running transpose example..."
	python algorithms/transpose/cache_aware.py

# Run additional examples
example-small:
	@echo "Running small matrix example (2x2)..."
	python -c "from io_simulator import IOSimulator; from algorithms.transpose import transpose_cache_aware; import numpy as np; A = np.array([[1, 2], [3, 4]]); print('Original:'); print(A); disk = IOSimulator(A, block_size=1, memory_size=4); result_flat, io_count = transpose_cache_aware(disk, 2, 2); result = result_flat.reshape(2, 2); print('Transposed:'); print(result); print(f'I/O count: {io_count}')"

example-large:
	@echo "Running large matrix example (8x8)..."
	python -c "from io_simulator import IOSimulator; from algorithms.transpose import transpose_cache_aware; import numpy as np; A = np.arange(64).reshape(8, 8); print('Original (8x8):'); print(A); disk = IOSimulator(A, block_size=2, memory_size=16); result_flat, io_count = transpose_cache_aware(disk, 8, 8); result = result_flat.reshape(8, 8); print('Transposed:'); print(result); print(f'I/O count: {io_count}')"

# Run external memory sorting example
example-sorting:
	@echo "Running external memory sorting example..."
	python algorithms/sorting/external_merge_sort.py

# Run B-tree example
example-btree:
	@echo "Running B-tree example..."
	python algorithms/searching/btree/btree.py

# Run Buffer tree example
example-buffer-tree:
	@echo "Running Buffer tree example..."
	python algorithms/searching/buffer_tree/buffer_tree.py

# Run Priority queue example
example-priority-queue:
	@echo "Running Priority queue example..."
	python algorithms/searching/priority_queue/priority_queue.py

# Run Time-forward processing example
example-time-forward:
	@echo "Running Time-forward processing example..."
	python algorithms/time_forward_processing/maximal_independent_sets.py

# Run Maximal independent sets example
example-maximal-independent-sets:
	@echo "Running Maximal independent sets example..."
	python algorithms/time_forward_processing/maximal_independent_sets.py

# Run all examples
examples: example-transpose-cache-aware example-small example-large example-transpose-cache-oblivious example-sorting example-btree example-buffer-tree example-priority-queue example-time-forward example-maximal-independent-sets

# Clean up __pycache__ directories
clean:
	@echo "Cleaning up __pycache__ directories..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Install dependencies (if needed)
install:
	@echo "Installing dependencies..."
	pip install numpy

# Run tests with coverage (if coverage is installed)
test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=. --cov-report=html --cov-report=term

# Run a quick test to verify everything works
quick-test:
	@echo "Running quick verification test..."
	python -c "from io_simulator import IOSimulator; from algorithms.transpose import transpose_cache_aware; import numpy as np; A = np.arange(4).reshape(2, 2); disk = IOSimulator(A, block_size=1, memory_size=4); result_flat, io_count = transpose_cache_aware(disk, 2, 2); result = result_flat.reshape(2, 2); print('✓ Quick test passed!'); print(f'Result: {result}'); print(f'I/O count: {io_count}')"
