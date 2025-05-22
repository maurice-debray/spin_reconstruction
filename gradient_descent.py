#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numba import jit, prange
from qutip import jmat
from scipy.optimize import minimize

from constants import (erbium_gamma, gamma_ratio, gamma_w, h, lattice_s,
                       lattice_x, lattice_y, lattice_z, mu_0, omega_I, omega_S,
                       site_nb)
from gitlock import get_config
from measurement_data import a_par_data, nb_par_data, renormalized_data

B_x = get_config("gradient_descent", ["angle", "B_x"]) / 180 * np.pi
B_y = get_config("gradient_descent", ["angle", "B_y"]) / 180 * np.pi

config = get_config("gradient_descent", ["config"])
max_distance = get_config("gradient_descent", ["max_distance"])

a_par_weight = 0
"""np.array(
    get_config("gradient_descent", ["cost", "a_par_reference"])
) ** (-2)"""
nb_par_weight = gamma_ratio**2

filename = get_config("gradient_descent", ["filename"])


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


@jit
def dipolar_diff_hamiltonian(mu_1, mu_2, xyz):
    """
    Computes the full dipole hamiltonian of two nuclear spins magnetic moment
    """
    r = np.linalg.norm(xyz)

    # Derivative of 1/r**3
    H = (
        xyz[:, None, None]
        * (
            -3
            / r**5
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
        )[None, :, :]
    )

    # Derivative of u_r (unit vector)
    du = (
        np.array(
            [
                [
                    xyz[1] ** 2 + xyz[2] ** 2,
                    -xyz[0] * xyz[1],
                    -xyz[0] * xyz[2],
                ],
                [
                    -xyz[1] * xyz[0],
                    xyz[0] ** 2 + xyz[2] ** 2,
                    -xyz[1] * xyz[2],
                ],
                [
                    -xyz[2] * xyz[0],
                    -xyz[2] * xyz[1],
                    xyz[0] ** 2 + xyz[1] ** 2,
                ],
            ]
        )
        / r**3
    )

    for grad_direction in range(3):

        H[grad_direction] += (
            -3
            / r**4
            * (
                (
                    (
                        du[grad_direction, 0] * mu_1[0]
                        + du[grad_direction, 1] * mu_1[1]
                        + du[grad_direction, 2] * mu_1[2]
                    )
                    @ (xyz[0] * mu_2[0] + xyz[1] * mu_2[1] + xyz[2] * mu_2[2])
                )
                + (
                    (xyz[0] * mu_1[0] + xyz[1] * mu_1[1] + xyz[2] * mu_1[2])
                    @ (
                        du[grad_direction, 0] * mu_2[0]
                        + du[grad_direction, 1] * mu_2[1]
                        + du[grad_direction, 2] * mu_2[2]
                    )
                )
            )
        )

    return mu_0 / 4 / np.pi * H


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
        + H_zeeman_I2
        + dipolar_hamiltonian(mu_I1, gamma_ratio * mu_I2, r1 - r2)
        + dipolar_hamiltonian(mu_I1, mu_S, r1 - erbium_position)
        + dipolar_hamiltonian(gamma_ratio * mu_I2, mu_S, r2 - erbium_position)
    )
    return H_0


@jit
def get_diff_hamiltonian2(r1, r2, B0, gamma_ratio):
    H_1 = dipolar_diff_hamiltonian(
        mu_I1, gamma_ratio * mu_I2, r1 - r2
    ) + dipolar_diff_hamiltonian(mu_I1, mu_S, r1 - erbium_position)
    H_2 = -dipolar_diff_hamiltonian(
        mu_I1, gamma_ratio * mu_I2, r1 - r2
    ) + dipolar_diff_hamiltonian(gamma_ratio * mu_I2, mu_S, r2 - erbium_position)
    return H_1, H_2


@jit
def get_hamiltonian(r, B0):
    H_zeeman_erbium, H_zeeman_I1, _ = get_zeeman(B0)
    H_0 = (
        H_zeeman_erbium
        + H_zeeman_I1
        + dipolar_hamiltonian(mu_I1, mu_S, r - erbium_position)
    )
    return H_0


@jit
def get_diff_hamiltonian(r, B0):
    return dipolar_diff_hamiltonian(mu_I1, mu_S, r - erbium_position)


# In[7]:


def printrel(a, b):
    print(np.abs(a - b) / b)


# @jit
def compute_coupling(r1, r2, B, gamma_ratio):
    H = get_hamiltonian2(r1, r2, B, gamma_ratio)
    dH1, dH2 = get_diff_hamiltonian2(r1, r2, B, gamma_ratio)

    eig, v = np.linalg.eigh(H)
    eigv = np.asmatrix(v)

    dE1 = np.empty(3)
    dE2 = np.empty(3)

    for grad_dir in range(3):
        dE1[grad_dir] = np.real(
            np.vdot(eigv[:, 0], dH1[grad_dir] @ eigv[:, 0])
            + np.vdot(eigv[:, 3].H, dH1[grad_dir] @ eigv[:, 3])
            - np.vdot(eigv[:, 1].H, dH1[grad_dir] @ eigv[:, 1])
            - np.vdot(eigv[:, 2].H, dH1[grad_dir] @ eigv[:, 2])
        ).item()

    for grad_dir in range(3):
        dE2[grad_dir] = np.real(
            np.vdot(eigv[:, 0].H, (dH2[grad_dir]) @ eigv[:, 0])
            + np.vdot(eigv[:, 3].H, (dH2[grad_dir]) @ eigv[:, 3])
            - np.vdot(eigv[:, 1].H, (dH2[grad_dir]) @ eigv[:, 1])
            - np.vdot(eigv[:, 2].H, (dH2[grad_dir]) @ eigv[:, 2])
        ).item()

    return (eig[0] + eig[3] - eig[1] - eig[2]) / h, dE1 / h, dE2 / h, eig, eigv


# @jit
def compute_a_par(r, B):
    H = get_hamiltonian(r, B)
    dH = get_diff_hamiltonian(r, B)

    eig, v = np.linalg.eigh(H)
    eigv = np.asmatrix(v)

    dE = np.empty(3)

    for grad_dir in range(3):
        dE[grad_dir] = np.real(
            eigv[:, 0].H * np.asmatrix(dH[grad_dir]) * eigv[:, 0]
            + eigv[:, 1].H * np.asmatrix(dH[grad_dir]) * eigv[:, 1]
            + eigv[:, 7].H * np.asmatrix(dH[grad_dir]) * eigv[:, 7]
            + eigv[:, 6].H * np.asmatrix(dH[grad_dir]) * eigv[:, 6]
            - eigv[:, 2].H * np.asmatrix(dH[grad_dir]) * eigv[:, 2]
            - eigv[:, 3].H * np.asmatrix(dH[grad_dir]) * eigv[:, 3]
            - eigv[:, 4].H * np.asmatrix(dH[grad_dir]) * eigv[:, 4]
            - eigv[:, 5].H * np.asmatrix(dH[grad_dir]) * eigv[:, 5]
        ).item()

    return (
        eig[0] + eig[1] + eig[7] + eig[6] - eig[2] - eig[4] - eig[3] - eig[5]
    ) / h / 2, dE / h / 2


# @jit
def index_to_coord(index, max_distance, site_nb):
    center = max_distance // 2
    return (
        index // (max_distance**2 * site_nb) - center,
        index // (max_distance * site_nb) % max_distance - center,
        index // site_nb % max_distance - center,
        index % site_nb,
    )


def index_to_position(i, max_distance, site_nb):
    c = index_to_coord(i, max_distance, site_nb)
    return c[0] * lattice_x + c[1] * lattice_y + c[2] * lattice_z + lattice_s[c[3]]


# @jit(parallel=True)
def cost(
    positions,
    B,
    couplings_data,
    a_par_data,
    nb_par_data,
    a_par_weight,
    nb_par_weight,
    shape,
):
    err = np.zeros((len(a_par_data), len(a_par_data)))
    jac = np.zeros((len(a_par_data), len(a_par_data), 3 * len(a_par_data)))
    for i in prange(len(a_par_data)):
        a_par, d_apar = compute_a_par(positions[3 * i : 3 * i + 3], B)
        err[i, i] += a_par_weight * (a_par - a_par_data[i]) ** 2
        jac[i, i, 3 * i : 3 * i + 3] += (
            2 * a_par_weight * (a_par - a_par_data[i]) * d_apar
        )
        nb_par, d_nb_par, _, _, _ = compute_coupling(
            positions[3 * i : 3 * i + 3],
            0.5 * lattice_x + 0.5 * lattice_y,
            B,
            1 / gamma_ratio,
        )
        err[i, i] += nb_par_weight * (nb_par - nb_par_data[i]) ** 2
        jac[i, i, 3 * i : 3 * i + 3] += (
            2 * nb_par_weight * (nb_par - nb_par_data[i]) * d_nb_par
        )
    for i in prange(len(a_par_data)):
        for j in prange(i + 1, len(a_par_data)):
            if not np.isnan(couplings_data[i, j]):
                cpl, d_cpl1, d_cpl2, _, _ = compute_coupling(
                    positions[3 * i : 3 * i + 3], positions[3 * j : 3 * j + 3], B, 1
                )

                err[i, j] += (couplings_data[i, j] - cpl) ** 2
                jac[i, j, 3 * i : 3 * i + 3] = 2 * (cpl - couplings_data[i, j]) * d_cpl1
                jac[i, j, 3 * j : 3 * j + 3] = 2 * (cpl - couplings_data[i, j]) * d_cpl2
    jac_sum = np.zeros(3 * len(a_par_data))
    for i in range(len(a_par_data)):
        for j in range(i, len(a_par_data)):
            jac_sum += jac[i, j]

    e = np.sum(err)
    print(e, jac_sum)

    return e, jac_sum


B = np.array([B_x, B_y, 1])
B_0 = B / np.linalg.norm(B)


def sanity_check():
    B = np.array([B_x, B_y, 1])
    B_0 = B / np.linalg.norm(B)

    r = np.array([0, -2.5e-10, 5e-10])
    dr = np.array([3e-13, 1e-13, 1e-13])

    E, dE = compute_a_par(r, B_0)
    E1, _ = compute_a_par(r + dr, B_0)

    print(
        f"""
    Base A parallel: {E}
    Gradient: {dE}
    Energy at r+dr (1st order): {E + np.sum(dE*dr)}
    Energy at r+dr (true): {E1}
    """
    )
    printrel(np.sum(dE * dr), (E1 - E))

    r2 = np.array([0, 0, 0])
    dr2 = np.array([0, 0, 0])

    H = get_hamiltonian2(r, r2, B_0, 1.0)
    H1 = get_hamiltonian2(r + dr, r2 + dr2, B_0, 1.0)
    dH1, dH2 = get_diff_hamiltonian2(r, r2, B_0, 1.0)

    dH = (
        dH1[0] * dr[0]
        + dH1[1] * dr[1]
        + dH1[2] * dr[2]
        + dH2[0] * dr2[0]
        + dH2[1] * dr2[1]
        + dH2[2] * dr2[2]
    )
    print((-H + H1 - (dH)) / (np.abs(H - H1)))

    print(np.abs(H - H1))

    E, dE, dE2, eig0, eigv0 = compute_coupling(r, r2, B_0, 1.0)
    E1, _, _, eig, eigv = compute_coupling(r + dr, r2 + dr2, B_0, 1.0)

    for i in range(8):
        printrel(eigv0[:, i].H * np.asmatrix(dH) * eigv0[:, i], eig[i] - eig0[i])

    print(
        f"""
    Base SEDOR: {E}
    Gradient: {dE, dE2}
    Energy at r+dr (1st order): {E + np.sum(dE*dr) + np.sum(dE2*dr2)}
    Energy at r+dr (true): {E1}
    Relative error = {np.abs(np.sum(dE*dr) + np.sum(dE2*dr2) - (E1-E))/np.abs(E1 - E)}
    """
    )


positions0 = np.array([index_to_position(i, max_distance, site_nb) for i in config])


res = minimize(
    cost,
    positions0.flatten(),
    jac=True,
    options={"maxiter": 1000, "disp": True},
    args=(
        B_0,
        renormalized_data,
        a_par_data,
        nb_par_data,
        a_par_weight,
        nb_par_weight,
        positions0.shape,
    ),
)
print(positions0.flatten())
print(res.x)
