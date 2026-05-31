"""
Matrix transpose algorithms for I/O efficient computation.

This module contains implementations of cache-aware and cache-oblivious
matrix transposition algorithms optimized for external memory.
"""

from .cache_aware import transpose_cache_aware
from .cache_oblivious import transpose_cache_oblivious

__all__ = ["transpose_cache_aware", "transpose_cache_oblivious"]
