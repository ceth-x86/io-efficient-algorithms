"""
External memory searching algorithms.

This module implements data structures optimized for external memory access,
where I/O operations are the primary performance bottleneck.
"""

from .btree import BTree
from .buffer_tree import BufferTree, Operation, OperationType
from .priority_queue import ExternalPriorityQueue

__all__ = ['BTree', 'BufferTree', 'Operation', 'OperationType', 'ExternalPriorityQueue']