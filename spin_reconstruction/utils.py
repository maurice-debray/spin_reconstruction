"""
Utilities for mapping between array indices, lattice coordinates, and physical positions.

This module provides functions to convert between:
  - Lattice indices and (x, y, z, site) lattice coordinates.
  - 3D indices and displacement vectors in physical units.
  - Lattice indices and real-space position vectors using the lattice parameters from :py:mod:`.constants`.

Functions:
  - :py:func:`coord_to_index` — Convert (x, y, z, site) to lattice index.
  - :py:func:`index_to_coord` — Convert lattice index to (x, y, z, site).
  - :py:func:`compute_dr` — Map 3D index to real-space position
  - :py:func:`index_to_position` — Map lattice index to real-space position.
  - :py:func:`exchange_columns` — Exchange spins indeces in a couplings matrix
"""

import numpy as np
from numba import jit

from .constants import lattice_s, lattice_x, lattice_y, lattice_z, site_nb


@jit
def coord_to_index(vec, max_distance, site_nb):
    """
    Maps (x, y, z, site) to a lattice index.

    This is the inverse operation of :py:func:`index_to_coord`.

    :param tuple vec: (x, y, z, site) coordinates.
    :param int max_distance: Spatial grid size per dimension.
    :param int site_nb: The number of sites per lattice cell.
    :returns: Lattice index.
    :rtype: int
    """
    center = max_distance // 2
    return (
        ((vec[0] + center) * max_distance + (vec[1] + center)) * max_distance
        + (vec[2] + center)
    ) * site_nb + vec[3]


@jit
def index_to_coord(index, max_distance, site_nb):
    """
    Maps a lattice index to (x, y, z, site).

    This is the inverse operation of :py:func:`coord_to_index`.

    :param int index: Lattice index
    :param int max_distance: The number of lattice cells along each axis.
    :param int site_nb: The number of sites per lattice cell.
    :returns: (x, y, z, site) where ``x, y, z`` are the spatial coordinates in the range ``[-max_distance//2, max_distance//2]`` and ``site`` is the site index in the range ``[0, site_nb-1]``
    :rtype: tuple of int
    """

    center = max_distance // 2
    return (
        index // (max_distance**2 * site_nb) - center,
        index // (max_distance * site_nb) % max_distance - center,
        index // site_nb % max_distance - center,
        index % site_nb,
    )


@jit
def compute_dr(index, size, distance):
    """
    Computes a small displacement vector corresponding to a 3D index.

    :param int i: 3D index inside ``[0, size**3)``.
    :param int size: Number of points along each spatial dimension.
    :param float distance: Physical spacing between grid points.
    :returns: 3D displacement vector in physical units.
    :rtype: :py:class:`numpy.ndarray`
    """
    center = size // 2
    x = index // (size**2) - center
    y = (index // size) % size - center
    z = index % size - center
    return np.array([x, y, z]) * distance


@jit
def origin_index(size):
    """
    Computes the 3D index of the (0,0,0) vector

    :param int size: Number of points along each spatial dimension.
    :returns: 3D index of the 0 vector
    :rtype: int
    """
    center = size // 2
    return (size**2) * center + size * center + center


@jit
def index_to_position(i, max_distance):
    """
    Map a lattice index to a 3D position vector in physical space.

    Uses :py:func:`index_to_coord` and then convert to 3D space using the
    lattice parameters in :py:mod:`.constants`

    :param int i: Lattice index.
    :param int max_distance: Grid size per dimension.
    :returns: 3D position vector.
    :rtype: :py:class:`numpy.ndarray`
    """
    c = index_to_coord(i, max_distance, site_nb)
    return c[0] * lattice_x + c[1] * lattice_y + c[2] * lattice_z + lattice_s[c[3]]


@jit
def exchange_columns(couplings, permutation, a, b):
    """
    Swap columns and corresponding rows ``a`` and ``b`` in a spin couplings matrix (similar to the one in ``spin_couplings.csv``). The operation is done in-place.

    Also updates the `permutation` array to reflect the swap.

    :param numpy.ndarray couplings: 2D square matrix of couplings.
    :param numpy.ndarray permutation: 1D array tracking the current order of columns/rows.
    :param int a: Index of the first column/row to swap.
    :param int b: Index of the second column/row to swap.

    :notes:
        Only the upper triangle of the matrix is explicitly modified.
    """

    a, b = min(a, b), max(a, b)
    permutation[a], permutation[b] = permutation[b], permutation[a]
    for i in range(a):
        couplings[i, a], couplings[i, b] = couplings[i, b], couplings[i, a]
    for i in range(a + 1, b):
        couplings[a, i], couplings[i, b] = couplings[i, b], couplings[a, i]
    for i in range(b + 1, couplings.shape[0]):
        couplings[a, i], couplings[b, i] = couplings[b, i], couplings[a, i]


def set_placing_order(couplings, first=0):
    """
    Reorders a symmetric coupling matrix by iteratively swapping columns and rows
    to place the largest remaining off-diagonal element at each step.

    Also tracks the permutation applied to the columns/rows.

    :param numpy.ndarray couplings: 2D square matrix of couplings to reorder.
    :returns: Tuple containing the reordered coupling matrix and the final permutation array.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """

    n_tot = couplings.shape[0]
    permutation = np.arange(n_tot)
    if first != 0:
        exchange_columns(couplings, permutation, 0, first)

    for i in range(1, n_tot):
        next_index = np.nanargmax(np.abs(couplings[:i, i:])) % (n_tot - i) + i
        if next_index != i:
            exchange_columns(couplings, permutation, i, next_index)
    return couplings, permutation
