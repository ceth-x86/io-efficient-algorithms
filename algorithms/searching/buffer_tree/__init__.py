"""
Buffer tree implementation for batched external memory operations.

Provides advanced data structure achieving sorting bound O(n/B·log_{M/B}(n/M))
through operation buffering and batch processing.
"""

from .buffer_tree import BufferTree, Operation, OperationType

__all__ = ['BufferTree', 'Operation', 'OperationType']