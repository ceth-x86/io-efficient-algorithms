"""
External Memory Priority Queue Module

Phase-based priority queue achieving optimal I/O complexity O(n/B·log_{M/B}(n/M))
for batch operations through clever phase management and in-memory minimum sets.
"""

from .priority_queue import ExternalPriorityQueue

__all__ = ['ExternalPriorityQueue']