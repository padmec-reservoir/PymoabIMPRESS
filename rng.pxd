"""Implements range functionality."""

from pymoab cimport moab, eh


cdef class Range:

    cdef moab.Range * inst
