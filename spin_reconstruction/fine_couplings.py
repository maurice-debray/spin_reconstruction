"""
Routines to compute spatially-resolved couplings (dipolar or hyperfine
couplings) between spins in the lattice.  Couplings are evaluated over
displacement grids around reference positions.

Functions:
- :py:func:`WW_couplings` : Computes ZZ-coupling matrices between two tungsten nuclei.
- :py:func:`nb_couplings` : Computes ZZ-coupling matrices between a tungsten nucleus and a niobium site.
- :py:func:`erbium_tungsten_couplings` : Computes hyperfine couplings between an erbium and a displaced tungsten.
"""

import numpy as np
from numba import jit, prange

from .constants import gamma_ratio, lattice_s
from .hamiltonian import nuclei_nuclei_coupling, tungsten_erbium_coupling
from .utils import compute_dr


@jit(parallel=True)
def erbium_tungsten_couplings(vec, size, distance, B):
    """
    Compute the hyperfine coupling between an erbium and a tungsten spin
    over a displacement grid around the tungsten position.

    The tungsten position is displaced around ``vec`` by a small grid of
    positions computed via :py:func:`.utils.compute_dr`.

    The parallel hyperfine coupling is computed via :py:func:`.hamiltonian.tungsten_erbium_coupling`
    for each displaced position.

    :param array-like vec: Reference position of the tungsten spin.
    :param int size: Number of points along each spatial dimension for the displacement grid.
    :param float distance: Physical spacing between grid points.
    :param array-like B: Static magnetic field vector (in T).
    :returns: Array of hyperfine couplings (in Hz), shape ``(size**3,)``.
    :rtype: :py:class:`numpy.ndarray`
    """
    a_par = np.empty((size**3,))
    for i in prange(size**3):
        r = compute_dr(i, size, distance) + vec
        a_par[i] = tungsten_erbium_coupling(r, B)
    return a_par


# TODO: Rename this to stick with naming convention (full names)
@jit(parallel=True)
def WW_couplings(vec1, vec2, size, distance, B):
    """
    Compute the ZZ-type couplings between two tungsten spins.

    Each tungsten is displaced around its nominal position ``vec1`` and ``vec2``
    by a small grid of positions computed via :py:func:`.utils.compute_dr`.
    For each pair of displacements, the ZZ-coupling is computed using
    :py:func:`.hamiltonian.nuclei_nuclei_coupling`.

    :param array-like vec1: Reference position of the first tungsten spin.
    :param array-like vec2: Reference position of the second tungsten spin.
    :param int size: Number of points along each spatial dimension for the displacement grid.
    :param float distance: Physical spacing between grid points.
    :param array-like B: Static magnetic field vector (in T).
    :returns: Matrix of ZZ-couplings (in Hz) of shape ``(size**3, size**3)``.
    :rtype: :py:class:`numpy.ndarray`
    """
    couplings = np.empty((size**3, size**3))
    for i in prange(size**3):
        for j in prange(size**3):
            r1 = compute_dr(i, size, distance) + vec1
            r2 = compute_dr(j, size, distance) + vec2
            c = nuclei_nuclei_coupling(r1, r2, B, 1.0)
            couplings[i, j] = c
    return couplings


# TODO: Rename this to stick with naming convention (full names)
@jit(parallel=True)
def nb_couplings(vec1, size, distance, B):
    """
    Compute the ZZ-type coupling matrix between a tungsten spin and a niobium site.

    The tungsten is displaced around its reference position ``vec1``, and the niobium
    is fixed at the first site position ``lattice_s[0]``. For each pair of displacements,
    the ZZ-coupling is computed via :py:func:`nuclei_nuclei_coupling`.

    :param array-like vec1: Reference position of the tungsten spin.
    :param int size: Number of points along each spatial dimension for the displacement grid.
    :param float distance: Physical spacing between grid points.
    :param array-like B: Static magnetic field vector (in T).
    :returns: Matrix of ZZ-couplings (in Hz) of shape ``(size**3, size**3)``.
    :rtype: :py:class:`numpy.ndarray`
    """
    couplings = np.empty((size**3, size**3))
    for i in prange(size**3):
        for j in prange(size**3):
            r1 = compute_dr(i, size, distance) + vec1
            r2 = compute_dr(j, size, distance) + lattice_s[0]
            c = nuclei_nuclei_coupling(r1, r2, B, 1 / gamma_ratio)
            couplings[i, j] = c
    return couplings
