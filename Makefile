# Makefile for transpose_cache_aware project

.PHONY: help test test-io test-transpose example example-small example-large examples clean install

# Default target
help:
	@echo "Available targets:"
	@echo "  help        - Show this help message"
	@echo "  test        - Run all tests"
	@echo "  test-io     - Run IOSimulator tests only"
	@echo "  test-transpose - Run transpose tests only"
	@echo "  example     - Run the built-in transpose example (4x4 matrix)"
	@echo "  example-small - Run small matrix example (2x2)"
	@echo "  example-large - Run large matrix example (8x8)"
	@echo "  examples    - Run all examples"
	@echo "  clean       - Clean up __pycache__ directories"
	@echo "  install     - Install dependencies"

# Run all tests
test:
	@echo "Running all tests..."
	python -m unittest tests.test_io_simulator tests.test_transpose -v

# Run IOSimulator tests only
test-io:
	@echo "Running IOSimulator tests..."
	python -m unittest tests.test_io_simulator -v

# Run transpose tests only
test-transpose:
	@echo "Running transpose tests..."
	python -m unittest tests.test_transpose -v

# Run the transpose example (built-in example in transpose_cache_aware.py)
example:
	@echo "Running transpose example..."
	python algorithms/transpose_cache_aware.py

# Run additional examples
example-small:
	@echo "Running small matrix example (2x2)..."
	python -c "from io_simulator import IOSimulator; from algorithms import transpose_cache_aware; import numpy as np; A = np.array([[1, 2], [3, 4]]); print('Original:'); print(A); sim = IOSimulator(A, block_size=1, memory_size=4); result, io_count = transpose_cache_aware(sim); print('Transposed:'); print(result); print(f'I/O count: {io_count}')"

example-large:
	@echo "Running large matrix example (8x8)..."
	python -c "from io_simulator import IOSimulator; from algorithms import transpose_cache_aware; import numpy as np; A = np.arange(64).reshape(8, 8); print('Original (8x8):'); print(A); sim = IOSimulator(A, block_size=2, memory_size=16); result, io_count = transpose_cache_aware(sim); print('Transposed:'); print(result); print(f'I/O count: {io_count}')"

# Run all examples
examples: example example-small example-large

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
	python -m coverage run -m unittest tests.test_io_simulator tests.test_transpose
	python -m coverage report
	python -m coverage html

# Run a quick test to verify everything works
quick-test:
	@echo "Running quick verification test..."
	python -c "from io_simulator import IOSimulator; from algorithms import transpose_cache_aware; import numpy as np; A = np.arange(4).reshape(2, 2); sim = IOSimulator(A, block_size=1, memory_size=4); result, io_count = transpose_cache_aware(sim); print('✓ Quick test passed!'); print(f'Result: {result}'); print(f'I/O count: {io_count}')"
