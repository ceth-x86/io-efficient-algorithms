"""
B-tree implementation for external memory dictionary operations.

Provides classic balanced tree structure optimized for block-based I/O
with O(log_B(n/B)) complexity per operation.
"""

from .btree import BTree

__all__ = ['BTree']