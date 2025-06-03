#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path

import h5py
import numpy as np
from numba import jit, prange

from constants import gamma_ratio
from gitlock import get_commit_hash, get_config
from measurement_data import a_par_data, nb_par_data, renormalized_data

# In[2]:


# In[3]:

a_par_weight = np.array(
    get_config("fine_reconstruction", ["cost", "a_par_reference"])
) ** (-2)
tolerance = get_config("fine_reconstruction", ["cost", "tolerance"])
nb_par_weight = gamma_ratio**2
nb_tolerance = tolerance / gamma_ratio
cutoff = get_config("fine_reconstruction", ["cost", "cutoff"])

file = get_config("fine_reconstruction", ["filename"])
couplings_file = get_config("fine_reconstruction", ["couplings_file"])

selected_sites = get_config("fine_reconstruction", ["selected_sites"])

if len(selected_sites) == 0:
    selected_sites = list(range(len(a_par_data)))


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
    exchange_columns(couplings, permutation, 7, 8)
    return couplings, permutation


@jit(parallel=True)
def compute_new_possible_config(possible_configurations, size, len_config, n_placed):
    new_possible_configurations = np.zeros(
        (len(possible_configurations) * size, len_config), dtype=np.uint64
    )
    for c in prange(len(possible_configurations)):
        config = possible_configurations[c]
        # Get candidates
        for site in prange(size):
            for i in prange(n_placed + 1):
                new_possible_configurations[c * size + site, i] = config[i]
            new_possible_configurations[c * size + site, n_placed] = site
    return new_possible_configurations


@jit(parallel=True)
def all_error_cost(
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
    errors = np.zeros(len(configs))
    for k in prange(len(configs)):
        err = 0.0
        config = configs[k]
        for i in range(n_max):
            # A parallel
            if not np.isnan(a_par_data[i]):
                err += a_par_weight * (a_par[i, config[i]] - a_par_data[i]) ** 2
            if (
                np.isnan(nb_par[i, config[i]])
                or np.abs(nb_par[i, config[i]] - nb_par_data[i]) > nb_tolerance
            ):
                err = np.inf
                break
            # Nb couplings
            if not np.isnan(nb_par_data[i]):
                err += nb_par_weight * (nb_par[i, config[i]] - nb_par_data[i]) ** 2
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


def compute_sites(
    couplings,
    a_par_data,
    nb_par_data,
    WW_couplings,
    WW_couplings_index,
    a_par,
    nb_par,
    tolerance,
    nb_tolerance,
    a_par_weight,
    nb_par_weight,
    cutoff,
    verbose=True,
):
    n_placed = 1
    n_tot = couplings.shape[0]
    # Here size is the number of position in the displacment cube
    size = a_par.shape[1]
    couplings, permutation = set_placing_order(couplings.copy())
    a_par_data = a_par_data[permutation]
    a_par = a_par[permutation]
    nb_par_data = nb_par_data[permutation]
    nb_par = nb_par[permutation]
    WW_couplings_index_permuted = np.array(  # Maybe there is a fancier way to do it with strange numpy indexing. Let's leave this for latter
        [
            [
                WW_couplings_index[permutation[i], permutation[j]]
                for j in range(len(permutation))
            ]
            for i in range(len(permutation))
        ]
    )
    possible_configurations = np.array(
        [[i] + [0] * (n_tot - 1) for i in range(size)],
        dtype=np.uint64,
    )
    print(WW_couplings.shape, a_par.shape, nb_par.shape, permutation)

    # initialize some reconstruction
    inf_index: np.intp = size
    errors = np.zeros(size)
    argsort_error = np.arange(size)
    while n_placed < n_tot:
        edge_spin = np.nanargmax(couplings[:n_placed, n_placed])

        if verbose:
            print(
                f"Placing {n_placed} (linked to {edge_spin}). {len(possible_configurations)}*{len(WW_couplings)} cases to process."
            )
        new_possible_configurations = compute_new_possible_config(
            possible_configurations, size, n_tot, n_placed
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
            WW_couplings_index_permuted,
            WW_couplings,
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
    raise ValueError(f"A file named {file} already exists")

git_commit = get_commit_hash()


with h5py.File(couplings_file, "r") as f:
    with h5py.File(file, "w") as g:
        for k, val in f.attrs.items():
            g.attrs[f"couplings_file_{k}"] = val
            # This line is kept for backward compatibility of the file format
            g.attrs[k] = val
        g.attrs["git_commit"] = git_commit
        mdata = g.create_group("measured_data")
        mdata.create_dataset(name="WW_couplings", data=renormalized_data)
        mdata.create_dataset(name="A_par_couplings", data=a_par_data)
        g.attrs["cutoff"] = cutoff
        g.attrs["git_commit"] = git_commit
        g.attrs["tolerance"] = tolerance
        g.attrs["nb_tolerance"] = nb_tolerance
        g.attrs["a_par_weight"] = a_par_weight
        g.attrs["nb_par_weight"] = nb_par_weight
        g.attrs["selected_sites"] = selected_sites

        n_tot = len(a_par_data)
        a_par = np.array([f[f"A_par_couplings/{i}"][:] for i in range(n_tot)])
        nb_par = np.array([f[f"Nb_par_couplings/{i}"][:] for i in range(n_tot)])

        print("Allocating")
        size = a_par.shape[1]
        print(n_tot, size)
        WW_couplings_index = np.full((n_tot, n_tot), -1, dtype=np.int64)
        WW_couplings = np.empty((len(f["/SEDOR_couplings"].keys()), size, size))

        print("Allocated")

        k = 0
        # TODO Improve this: It can be done with a single loop over the hdf5 keys
        for j in range(n_tot):
            for i in range(n_tot):
                if f"/SEDOR_couplings/{i}_{j}" in f:
                    WW_couplings_index[i, j] = k
                    WW_couplings[k] = np.array(
                        f[f"/SEDOR_couplings/{i}_{j}"], dtype=np.float64
                    )
                    k += 1

        final_sites, permutation, errors, ended_prematurely = compute_sites(
            renormalized_data[
                np.meshgrid(selected_sites, selected_sites, indexing="ij")
            ],
            a_par_data[selected_sites],
            nb_par_data[selected_sites],
            WW_couplings,
            WW_couplings_index[
                np.meshgrid(selected_sites, selected_sites, indexing="ij")
            ],
            a_par=a_par[selected_sites],
            nb_par=nb_par[selected_sites],
            tolerance=tolerance,
            nb_tolerance=nb_tolerance,
            a_par_weight=a_par_weight,
            nb_par_weight=nb_par_weight,
            cutoff=cutoff,
            verbose=True,
        )
        g.attrs["partial_solution"] = ended_prematurely
        g.create_dataset(name="sites", data=final_sites, dtype=np.uint64)
        g.create_dataset(name="permutation", data=permutation, dtype=np.uint64)
        g.create_dataset(name="errors", data=errors)
