"""
fd2d – standalone 2-D finite-difference forward / adjoint solver.

Public API
----------
FD2D    main solver class
ricker  Ricker source-time function
"""

from .fd2d   import FD2D
from .ricker import ricker

__all__ = ['FD2D', 'ricker']
