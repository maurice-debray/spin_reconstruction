"""
Cost functions for the reconstruction.

TODO: Remode code duplication
"""

import numpy as np
from numba import jit, prange


@jit(parallel=True)
def compute_new_possible_config(
    possible_configurations, len_all_couplings, len_config, n_placed
):
    new_possible_configurations = np.zeros(
        (len(possible_configurations) * len_all_couplings, len_config), dtype=np.uint64
    )
    for c in prange(len(possible_configurations)):
        config = possible_configurations[c]
        # Get candidates
        for site in prange(len_all_couplings):
            for i in prange(n_placed + 1):
                new_possible_configurations[c * len_all_couplings + site, i] = config[i]
            new_possible_configurations[c * len_all_couplings + site, n_placed] = site
    return new_possible_configurations


@jit(parallel=True)
def cost(
    configs,
    sedor_data,
    a_par_data,
    nb_par_data,
    n_max,
    all_couplings,
    a_par,
    nb_par,
    a_par_weight,
    nb_par_weight,
    tolerance,
    nb_tolerance,
):
    """
    Cost function where all spins share the same pool of sites
    """
    errors = np.zeros(len(configs))
    for k in prange(len(configs)):
        err = 0.0
        config = configs[k]
        for i in range(n_max):
            # A parallel
            if not np.isnan(a_par_data[i]):
                err += a_par_weight * (a_par[config[i]] - a_par_data[i]) ** 2
            if (
                np.isnan(nb_par[config[i]])
                or np.abs(nb_par[config[i]] - nb_par_data[i]) > nb_tolerance
            ):
                err = np.inf
                break
            # Nb couplings
            if not np.isnan(nb_par_data[i]):
                err += nb_par_weight * (nb_par[config[i]] - nb_par_data[i]) ** 2
            for j in range(i + 1, n_max):
                # WW couplings
                if (
                    np.isnan(all_couplings[config[i], config[j]])
                    or np.abs(all_couplings[config[i], config[j]] - sedor_data[i, j])
                    > tolerance
                ):
                    err = np.inf
                    break
                if not np.isnan(sedor_data[i, j]):
                    err += (sedor_data[i, j] - all_couplings[config[i], config[j]]) ** 2
            if err == np.inf:
                break
        errors[k] = err
    return errors


@jit(parallel=True)
def site_resolved_cost(
    configs,
    coupl,
    a_par_data,
    nb_par_data,
    n_max,
    all_couplings_index,
    all_couplings,
    a_par,
    nb_par,
    a_par_weight,
    nb_par_weight,
    tolerance,
    nb_tolerance,
):
    """
    Cost function where each spin has its own pool of possible sites.
    """
    errors = np.zeros(len(configs))
    for k in prange(len(configs)):
        err = 0.0
        config = configs[k]
        for i in range(n_max):
            current_nb_par = nb_par[i, config[i]]
            # A parallel
            if not np.isnan(a_par_data[i]):
                err += a_par_weight * (a_par[i, config[i]] - a_par_data[i]) ** 2
            if (
                np.isnan(current_nb_par)
                or np.abs(current_nb_par - nb_par_data[i]) > nb_tolerance
            ):  # current_nb_par can be nan if nb site and atom site are the same
                err = np.inf
                break
            # Nb couplings
            if not np.isnan(nb_par_data[i]):
                err += nb_par_weight * (current_nb_par - nb_par_data[i]) ** 2
            for j in range(i + 1, n_max):
                # WW couplings
                if all_couplings_index[i, j] != -1:
                    coupling = all_couplings[
                        all_couplings_index[i, j], config[i], config[j]
                    ]
                    if np.isnan(coupling) or np.abs(coupling - coupl[i, j]) > tolerance:
                        err = np.inf
                        break
                    if not np.isnan(coupl[i, j]):
                        err += (coupl[i, j] - coupling) ** 2
            if err == np.inf:
                break
        errors[k] = err
    return errors


@jit(parallel=True)
def site_resolved_cost_with_nb(
    configs,
    coupl,
    a_par_data,
    nb_par_data,
    n_max,
    all_couplings_index,
    all_couplings,
    a_par,
    nb_par,
    a_par_weight,
    nb_par_weight,
    tolerance,
    nb_tolerance,
):
    """
    Cost function with niobium relaxed
    """
    errors = np.zeros(len(configs))
    for k in prange(len(configs)):
        err = 0.0
        niobium_site = configs[k, 0]
        config = configs[k, 1:]
        for i in range(n_max):
            current_nb_par = nb_par[i, config[i], niobium_site]
            # A parallel
            if not np.isnan(a_par_data[i]):
                err += a_par_weight * (a_par[i, config[i]] - a_par_data[i]) ** 2
            if (
                np.isnan(current_nb_par)
                or np.abs(current_nb_par - nb_par_data[i]) > nb_tolerance
            ):  # current_nb_par can be nan if nb site and atom site are the same
                err = np.inf
                break
            # Nb couplings
            if not np.isnan(nb_par_data[i]):
                err += nb_par_weight * (current_nb_par - nb_par_data[i]) ** 2
            for j in range(i + 1, n_max):
                # WW couplings
                if all_couplings_index[i, j] != -1:
                    coupling = all_couplings[
                        all_couplings_index[i, j], config[i], config[j]
                    ]
                    if np.isnan(coupling) or np.abs(coupling - coupl[i, j]) > tolerance:
                        err = np.inf
                        break
                    if not np.isnan(coupl[i, j]):
                        err += (coupl[i, j] - coupling) ** 2
            if err == np.inf:
                break
        errors[k] = err
    return errors
