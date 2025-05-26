#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path

import h5py
import numpy as np
from numba import jit, prange
from qutip import jmat
from tqdm import trange

from constants import (erbium_gamma, gamma_ratio, gamma_w, h, lattice_s,
                       lattice_x, lattice_y, lattice_z, mu_0, omega_I, omega_S,
                       site_nb)
from gitlock import get_commit_hash, get_config

x_start = get_config("couplings", ["range", "x_start"])
y_start = get_config("couplings", ["range", "y_start"])
x_end = get_config("couplings", ["range", "x_end"])
y_end = get_config("couplings", ["range", "y_end"])


x_max = get_config("couplings", ["angle", "x_max"]) / 180 * np.pi
y_max = get_config("couplings", ["angle", "y_max"]) / 180 * np.pi
x_size = get_config("couplings", ["range", "x_size"])
y_size = get_config("couplings", ["range", "y_size"])

filename = get_config("couplings", ["filename"])
max_distance = get_config("couplings", ["lattice", "max_distance"])


def get_full_H_matrices(spin, pre_dim, post_dim):
    """
    Construct full spin operator matrices (Sx, Sy, Sz) embedded in a larger Hilbert space.

    This function generates the spin operators Sx, Sy, and Sz for a given spin value,
    and embeds them into a larger Hilbert space. The embedding
    places the spin operator between identity matrices of dimension `pre_dim` and `post_dim`,
    representing dimensions before and after the target spin in a composite system.

    Parameters
    ----------
    spin : float or int
        Spin value of the particle (e.g., 0.5 for spin-1/2, 1 for spin-1).

    pre_dim : int
        Dimension of the Hilbert space before the spin operator (e.g., for preceding spins).

    post_dim : int
        Dimension of the Hilbert space after the spin operator (e.g., for succeeding spins).

    Returns
    -------
    np.ndarray
        A NumPy array containing the full spin operator matrices [Sx, Sy, Sz], each of shape
        (`pre_dim` * dim_spin * `post_dim`, `pre_dim` * dim_spin * `post_dim`), where `dim_spin = int(2 * spin + 1)`.
    """
    if pre_dim == 0:
        pre_dim = 1
    if post_dim == 0:
        post_dim = 1
    eye_pre = np.eye(pre_dim)
    eye_post = np.eye(post_dim)
    Ix = jmat(spin, "x").full()
    Iy = jmat(spin, "y").full()
    Iz = jmat(spin, "z").full()
    Sx = np.kron(np.kron(eye_pre, Ix), eye_post)
    Sy = np.kron(np.kron(eye_pre, Iy), eye_post)
    Sz = np.kron(np.kron(eye_pre, Iz), eye_post)
    return np.array([Sx, Sy, Sz])


# In[5]:


@jit
def dipolar_hamiltonian(mu_1, mu_2, xyz):
    """
    Computes the full dipole hamiltonian of two nuclear spins magnetic moment
    """
    r = np.linalg.norm(xyz)
    return (
        mu_0
        / 4
        / np.pi
        / r**3
        * (
            # mu_1 . mu_2
            mu_1[0] @ mu_2[0]
            + mu_1[1] @ mu_2[1]
            + mu_1[2] @ mu_2[2]
            # -3/r² * (mu_1 . r) (mu_2 . r)
            - 3
            / r**2
            * (xyz[0] * mu_1[0] + xyz[1] * mu_1[1] + xyz[2] * mu_1[2])
            @ (xyz[0] * mu_2[0] + xyz[1] * mu_2[1] + xyz[2] * mu_2[2])
        )
    )


# In[6]:


S = get_full_H_matrices(1 / 2, 0, 4)
I1 = get_full_H_matrices(1 / 2, 2, 2)
I2 = get_full_H_matrices(1 / 2, 4, 0)

mu_S = erbium_gamma[:, None, None] * S
mu_I1 = gamma_w * I1
mu_I2 = gamma_w * I2


@jit
def get_zeeman(B0):
    # Zeeman for each atom in the 8dim Hailtonian
    H_zeeman_erbium = (
        -h  # Minus sign to compensate mu_S minus sign!
        / 2
        / np.pi
        * omega_S
        / np.linalg.norm(B0 * erbium_gamma)
        * (B0[0] * mu_S[0] + B0[1] * mu_S[1] + B0[2] * mu_S[2])
    )
    H_zeeman_I1 = (
        h / 2 / np.pi * omega_I * (B0[0] * I1[0] + B0[1] * I1[1] + B0[2] * I1[2])
    )
    H_zeeman_I2 = (
        h / 2 / np.pi * omega_I * (B0[0] * I2[0] + B0[1] * I2[1] + B0[2] * I2[2])
    )
    return H_zeeman_erbium, H_zeeman_I1, H_zeeman_I2


erbium_position = lattice_x * 0.5 + lattice_y * 0.5 + lattice_z * 0.5


@jit
def get_hamiltonian2(r1, r2, B0, gamma_ratio):
    H_zeeman_erbium, H_zeeman_I1, H_zeeman_I2 = get_zeeman(B0)
    H_0 = (
        H_zeeman_erbium
        + H_zeeman_I1
        + gamma_ratio * H_zeeman_I2
        + dipolar_hamiltonian(mu_I1, gamma_ratio * mu_I2, r1 - r2)
        + dipolar_hamiltonian(mu_I1, mu_S, r1 - erbium_position)
        + dipolar_hamiltonian(gamma_ratio * mu_I2, mu_S, r2 - erbium_position)
    )
    return H_0


@jit
def get_hamiltonian(r, B0):
    H_zeeman_erbium, H_zeeman_I1, _ = get_zeeman(B0)
    H_0 = (
        H_zeeman_erbium
        + H_zeeman_I1
        + dipolar_hamiltonian(mu_I1, mu_S, r - erbium_position)
    )
    return H_0


# In[7]:


@jit
def compute_coupling_full(vec1, vec2, B, gamma_ratio):
    r1 = (
        lattice_x * vec1[0]
        + lattice_y * vec1[1]
        + lattice_z * vec1[2]
        + lattice_s[vec1[3]]
    )
    r2 = (
        lattice_x * vec2[0]
        + lattice_y * vec2[1]
        + lattice_z * vec2[2]
        + lattice_s[vec2[3]]
    )

    H = get_hamiltonian2(r1, r2, B, gamma_ratio)

    eig = np.linalg.eigvalsh(H)
    return eig


@jit
def compute_coupling(vec1, vec2, B, gamma_ratio):
    eig = compute_coupling_full(vec1, vec2, B, gamma_ratio)
    return (eig[0] + eig[3] - eig[1] - eig[2]) / h


@jit
def compute_a_par_full(vec1, B):
    r = (
        lattice_x * vec1[0]
        + lattice_y * vec1[1]
        + lattice_z * vec1[2]
        + lattice_s[vec1[3]]
    )

    H = get_hamiltonian(r, B)

    eig = np.linalg.eigvalsh(H)
    return eig


@jit
def compute_a_par(vec1, B):
    eig = compute_a_par_full(vec1, B)
    return (
        (eig[0] + eig[1] + eig[7] + eig[6] - eig[2] - eig[4] - eig[3] - eig[5]) / h / 2
    )


# # Matrix diagonalization

# In[8]:


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


@jit(parallel=True)
def get_all_couplings(max_distance, site_nb, B):
    """
    It is ok to place the niobium in 0,0,0,0 since our system is invariant under B_field + spins inversion
    """
    couplings = np.empty((max_distance**3 * site_nb, max_distance**3 * site_nb))
    a_par = np.empty(max_distance**3 * site_nb)
    nb_par = np.empty(max_distance**3 * site_nb)
    for i in prange(max_distance**3 * site_nb):
        couplings[i, i] = np.nan
        vec1 = index_to_coord(i, max_distance, site_nb)
        a_par[i] = compute_a_par(vec1, B)
        if vec1 == (0, 0, 0, 0):
            nb_par[i] = np.nan
        else:
            nb_par[i] = compute_coupling(vec1, (0, 0, 0, 0), B, 1 / gamma_ratio)
    for i in prange(max_distance**3 * site_nb):
        for j in prange(i + 1, max_distance**3 * site_nb):
            vec1 = index_to_coord(i, max_distance, site_nb)
            vec2 = index_to_coord(j, max_distance, site_nb)
            c = compute_coupling(vec1, vec2, B, 1.0)
            couplings[i, j] = c
            couplings[j, i] = c

    return couplings, a_par, nb_par


def vector_couplings(max_distance, site_nb, B):
    couplings, a_par, nb_par = get_all_couplings(max_distance, site_nb, B)
    return couplings, a_par, nb_par


# In[9]:


# Generate all couplings !

if os.path.isfile(filename):
    raise ValueError(f"A file named {filename} already exists")

git_commit = get_commit_hash()

with h5py.File(filename, "w") as f:

    attrs = {
        "git_commit": git_commit,
        "max_distance": max_distance,
        "lattice_x": lattice_x,
        "lattice_y": lattice_y,
        "lattice_z": lattice_z,
        "lattice_s": lattice_s,
        "erbium_position": erbium_position,
        "erbium_gamma": erbium_gamma,
        "omega_I": omega_I,
        "omega_S": omega_S,
        "gamma_w": gamma_w,
        "x_max": x_max,
        "y_max": y_max,
        "x_size": x_size,
        "y_size": y_size,
        "x_start": x_start,
        "x_end": x_end,
        "y_start": y_start,
        "y_end": y_end,
    }

    for k, v in attrs.items():
        f.attrs[k] = v
    for x in trange(x_start, x_end):
        for y in trange(y_start, y_end):
            grp_name = f"B_sweep_{x}_{y}"
            if grp_name in f:
                print(f"Skipping {grp_name}")
                continue
            B = np.array([x * x_max / x_size, y * y_max / y_size, 1])
            B_0 = B / np.linalg.norm(B)
            all_couplings, a_parallel, nb_par = vector_couplings(
                max_distance=max_distance, site_nb=site_nb, B=B_0
            )

            g = f.create_group(grp_name)
            d1 = g.create_dataset(name="SEDOR_couplings", data=all_couplings)
            d2 = g.create_dataset(name="A_par_couplings", data=a_parallel)
            d3 = g.create_dataset(name="NB_couplings", data=nb_par)

            attrs = {
                "B": B_0,
                "x": x,
                "y": y,
            }

            for k, v in attrs.items():
                g.attrs[k] = v
