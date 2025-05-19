#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path

import h5py
import numpy as np
from numba import jit, prange
from tqdm import tqdm

from constants import gamma_ratio, site_nb
from gitlock import get_commit_hash, get_config
from measurement_data import a_par_data, nb_par_data, renormalized_data

# In[2]:


# In[3]:

a_par_weights = np.array(get_config("reconstruction", ["cost", "a_par_refs"])) ** (-2)
tolerance = get_config("reconstruction", ["cost", "tolerance"])
nb_par_weight = gamma_ratio**2
nb_tolerance = tolerance / gamma_ratio
cutoff = get_config("reconstruction", ["cost", "cutoff"])

file = get_config("reconstruction", ["filename"])
couplings_file = get_config("reconstruction", ["couplings_file"])


# # State reconstruction

# In[4]:


@jit
def index_to_coord(index, max_distance, site_nb):
    center = max_distance // 2
    return (
        index // (max_distance**2 * site_nb) - center,
        index // (max_distance * site_nb) % max_distance - center,
        index // site_nb % max_distance - center,
        index % site_nb,
    )


@jit
def coord_to_index(vec, max_distance, site_nb):
    center = max_distance // 2
    return (
        ((vec[0] + center) * max_distance + (vec[1] + center)) * max_distance
        + (vec[2] + center)
    ) * site_nb + vec[3]


# In[5]:


@jit
def exchange_columns(couplings, permutation, a, b):
    a, b = min(a, b), max(a, b)
    permutation[a], permutation[b] = permutation[b], permutation[a]
    for i in range(a):
        couplings[i, a], couplings[i, b] = couplings[i, b], couplings[i, a]
    for i in range(a + 1, b):
        couplings[a, i], couplings[i, b] = couplings[i, b], couplings[a, i]
    for i in range(b + 1, couplings.shape[0]):
        couplings[a, i], couplings[b, i] = couplings[b, i], couplings[a, i]


def set_placing_order(couplings):
    """
    First spin will always be niobium. Then we sort all other spins
    """
    n_tot = couplings.shape[0]
    permutation = np.arange(n_tot)
    for i in range(1, n_tot):
        next_index = np.nanargmax(np.abs(couplings[:i, i:])) % (n_tot - i) + i
        if next_index != i:
            exchange_columns(couplings, permutation, i, next_index)
    return couplings, permutation


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
def all_error_cost(
    configs,
    coupl,
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
                    or np.abs(all_couplings[config[i], config[j]] - coupl[i, j])
                    > tolerance
                ):
                    err = np.inf
                    break
                if not np.isnan(coupl[i, j]):
                    err += (coupl[i, j] - all_couplings[config[i], config[j]]) ** 2
            if err == np.inf:
                break
        errors[k] = err
    return errors


def compute_sites(
    couplings,
    a_par_data,
    nb_par_data,
    site_nb,
    all_couplings,
    tolerance,
    nb_tolerance,
    a_par,
    nb_par,
    a_par_weight,
    nb_par_weight,
    cutoff,
    verbose=True,
):
    n_placed = 1
    max_distance = round((all_couplings.shape[0] // site_nb) ** (1 / 3))
    if max_distance**3 * site_nb != all_couplings.shape[0]:
        raise ValueError("Impossible to get the right max_distance")
    n_tot = couplings.shape[0]
    couplings, permutation = set_placing_order(couplings.copy())
    a_par_data = a_par_data[permutation]
    possible_configurations = np.array(
        [[i] + [0] * (n_tot - 1) for i in range(max_distance**3 * site_nb)],
        dtype=np.uint64,
    )
    inf_index: np.intp = max_distance**3 * site_nb
    errors = np.zeros(max_distance**3 * site_nb)
    argsort_error = np.arange(max_distance**3 * site_nb)
    while n_placed < n_tot:
        # Be careful, position relative to edge_spin
        edge_spin = np.nanargmax(couplings[:n_placed, n_placed])

        if verbose:
            print(
                f"Placing {n_placed} (linked to {edge_spin}). {len(possible_configurations)}*{len(all_couplings)} cases to process."
            )
        new_possible_configurations = compute_new_possible_config(
            possible_configurations, len(all_couplings), n_tot, n_placed
        )
        checkpoint = (
            possible_configurations,
            permutation,
            errors[argsort_error[: np.minimum(cutoff, inf_index)]],
            True,
        )
        errors = all_error_cost(
            new_possible_configurations,
            couplings,
            a_par_data,
            nb_par_data,
            n_placed + 1,
            all_couplings,
            a_par,
            nb_par,
            a_par_weight,
            nb_par_weight,
            tolerance,
            nb_tolerance,
        )
        argsort_error = np.argsort(errors)
        inf_index = np.searchsorted(errors, np.inf, sorter=argsort_error)
        if inf_index == 0:
            if verbose:
                print("Ending prematurely")
            return checkpoint
        del checkpoint
        possible_configurations = new_possible_configurations[
            argsort_error[: np.minimum(cutoff, inf_index)]
        ].copy()
        n_placed += 1
    return (
        possible_configurations,
        permutation,
        errors[argsort_error[: np.minimum(cutoff, inf_index)]],
        False,
    )


if os.path.isfile(file):
    pass
    # raise ValueError(f"A file named {file} already exists")

git_commit = get_commit_hash()


with h5py.File(couplings_file, "r") as f:
    with h5py.File(file, "w") as g:
        for k, val in f.attrs.items():
            g.attrs[k] = val
        g.attrs["git_commit"] = git_commit
        mdata = g.create_group("measured_data")
        mdata.create_dataset(name="WW_couplings", data=renormalized_data)
        mdata.create_dataset(name="A_par_couplings", data=a_par_data)
        for a_par_weight in tqdm(a_par_weights):
            gr = g.create_group(f"Reconstructed_weight_{a_par_weight}")
            gr.attrs["cutoff"] = cutoff
            gr.attrs["git_commit"] = git_commit
            gr.attrs["tolerance"] = tolerance
            gr.attrs["nb_tolerance"] = nb_tolerance
            gr.attrs["a_par_weight"] = a_par_weight
            gr.attrs["nb_par_weight"] = nb_par_weight
            n_partial = 0
            for key, v in f.items():
                all_couplings = v["SEDOR_couplings"][:]
                a_parallel = v["A_par_couplings"][:]
                nb_par = v["NB_couplings"][:]
                final_sites, permutation, errors, ended_prematurely = compute_sites(
                    renormalized_data,
                    a_par_data,
                    nb_par_data,
                    site_nb=site_nb,
                    all_couplings=all_couplings,
                    tolerance=tolerance,
                    nb_tolerance=nb_tolerance,
                    a_par=a_parallel,
                    nb_par=nb_par,
                    a_par_weight=a_par_weight,
                    nb_par_weight=nb_par_weight,
                    cutoff=cutoff,
                    verbose=False,
                )
                if ended_prematurely:
                    n_partial += 1
                d = gr.create_group(f"Reconstructed_from_{k}")
                d.attrs["ended_prematurely"] = ended_prematurely
                for k, val in v.attrs.items():
                    d.attrs[k] = val
                d.attrs["couplings_key"] = key

                d.create_dataset(name="sites", data=final_sites, dtype=np.int64)
                d.create_dataset(name="permutation", data=permutation, dtype=np.uint64)
                d.create_dataset(name="errors", data=errors)
            gr.attrs["n_partial_solutions"] = n_partial
