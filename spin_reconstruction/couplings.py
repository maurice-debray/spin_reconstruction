import numpy as np
from numba import jit, prange

from .constants import gamma_ratio, site_nb
from .hamiltonian import nuclei_nuclei_coupling, tungsten_erbium_coupling
from .utils import coord_to_index, index_to_coord, index_to_position


# TODO: Rename to stick to convention of not using abbreviations for spin names
@jit(parallel=True)
def WW_couplings(max_distance, B):
    couplings = np.empty((max_distance**3 * site_nb, max_distance**3 * site_nb))
    for i in prange(max_distance**3 * site_nb):
        couplings[i, i] = np.nan
    for i in prange(max_distance**3 * site_nb):
        for j in prange(i + 1, max_distance**3 * site_nb):
            vec1 = index_to_position(i, max_distance)
            vec2 = index_to_position(j, max_distance)
            c = nuclei_nuclei_coupling(vec1, vec2, B, 1.0)
            couplings[i, j] = c
            couplings[j, i] = c

    return couplings


# TODO: Rename to stick to convention of not using abbreviations for spin names
@jit(parallel=True)
def nb_couplings(max_distance, B):
    nb_par = np.empty(max_distance**3 * site_nb)
    for i in prange(max_distance**3 * site_nb):
        vec1 = index_to_coord(i, max_distance, site_nb)
        if vec1 == (0, 0, 0, 0):
            nb_par[i] = np.nan
        else:
            nb_par[i] = nuclei_nuclei_coupling(
                index_to_position(i, max_distance),
                index_to_position(
                    coord_to_index((0, 0, 0, 0), max_distance, site_nb), max_distance
                ),
                B,
                1 / gamma_ratio,
            )

    return nb_par


@jit(parallel=True)
def full_nb_couplings(max_distance, B):
    couplings = np.empty((max_distance**3 * site_nb, max_distance**3 * site_nb))
    for i in prange(max_distance**3 * site_nb):
        couplings[i, i] = np.nan
    for i in prange(max_distance**3 * site_nb):
        for j in prange(i + 1, max_distance**3 * site_nb):
            vec1 = index_to_position(i, max_distance)
            vec2 = index_to_position(j, max_distance)
            c = nuclei_nuclei_coupling(vec1, vec2, B, 1 / gamma_ratio)
            couplings[i, j] = c
            couplings[j, i] = c

    return couplings


@jit(parallel=True)
def erbium_tungsten_couplings(max_distance, B):
    a_par = np.empty(max_distance**3 * site_nb)
    for i in prange(max_distance**3 * site_nb):
        vec1 = index_to_position(i, max_distance)
        a_par[i] = tungsten_erbium_coupling(vec1, B)
    return a_par


def all_couplings(max_distance, B):
    return (
        WW_couplings(max_distance, B),
        erbium_tungsten_couplings(max_distance, B),
        nb_couplings(max_distance, B),
    )
