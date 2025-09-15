"""
Time-Forward Processing Module

Implementation of I/O-efficient algorithms for computing local functions on DAGs
using external memory priority queues. Includes maximal independent set computation.
"""

from .maximal_independent_sets import (
    Graph,
    TimeForwardProcessor, 
    MaximalIndependentSetSolver
)

__all__ = [
    'Graph',
    'TimeForwardProcessor',
    'MaximalIndependentSetSolver'
]