"""
External memory searching algorithms.

This module implements data structures optimized for external memory access,
where I/O operations are the primary performance bottleneck.
"""

from .btree import BTree

__all__ = ['BTree']