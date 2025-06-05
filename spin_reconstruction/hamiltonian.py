"""
Module for spin Hamiltonian computations in a tungsten spin system.

This module provides utility functions for:
  - Constructing spin operators embedded in composite Hilbert spaces.
  - Computing static Hamiltonians for systems involving erbium and tungsten nuclear spins, including
    Zeeman and dipolar interaction terms.
  - Extracting eigenvalues and effective coupling constants (ZZ and hyperfine) from these Hamiltonians.

Key functionalities:
  - Dipolar and Zeeman Hamiltonian construction: :py:func:`dipolar_hamiltonian`, :py:func:`get_zeeman`, :py:func:`get_hamiltonian`, :py:func:`get_hamiltonian2`
  - Eigenvalue and coupling extraction: :py:func:`nuclei_nuclei_eigval`, :py:func:`tungsten_erbium_eigval`, :py:func:`nuclei_nuclei_coupling`, :py:func:`tungsten_erbium_coupling`

Physical constants are imported from :py:mod:`.constants`.

All Hamiltonians are returned as Hermitian :py:class:`numpy.ndarray` objects.
"""

import numpy as np
from numba import jit
from qutip import jmat

from .constants import (erbium_gamma, erbium_position, gamma_w, h, mu_0,
                        omega_I, omega_S)


def get_spin_matrices(spin, pre_dim, post_dim):
    """
    Constructs all spin-operator matrices (Sx, Sy, Sz) embedded in a larger Hilbert space.

    :param float|int spin: Spin value (e.g., 0.5 for spin-1/2).
    :param int pre_dim: Hilbert space dimension before the spin.
    :param int post_dim: Hilbert space dimension after the spin.
    :returns: List of full spin operator hermitian-matrices [Sx, Sy, Sz], each of size
              ``pre_dim * dim_spin * post_dim``, where
              ``dim_spin = int(2 * spin + 1)``.
    :rtype: :py:class:`numpy.ndarray`


    """
    if pre_dim == 0:
        raise ValueError(
            "Do you really want `pre_dim = 0` ? You probably want put 1 instead"
        )
    if post_dim == 0:
        raise ValueError(
            "Do you really want `post_dim = 0` ? You probably want put 1 instead"
        )
    eye_pre = np.eye(pre_dim)
    eye_post = np.eye(post_dim)
    Jx = jmat(spin, "x").full()
    Jy = jmat(spin, "y").full()
    Jz = jmat(spin, "z").full()
    Sx = np.kron(np.kron(eye_pre, Jx), eye_post)
    Sy = np.kron(np.kron(eye_pre, Jy), eye_post)
    Sz = np.kron(np.kron(eye_pre, Jz), eye_post)
    return np.array([Sx, Sy, Sz])


@jit
def dipolar_hamiltonian(mu_1, mu_2, xyz):
    r"""
    Compute the dipolar interaction Hamiltonian between two spin magnetic
    moments (This doesn't take into account fermi contact term).

    The Hamiltonian is given by the standard dipole-dipole interaction formula:

    .. math::
       H = \frac{\mu_0}{4 \pi r^3} \left[ \vec{\mu}_1 \cdot \vec{\mu}_2 - \frac{3}{r^2} (\vec{\mu}_1 \cdot \vec{r})(\vec{\mu}_2 \cdot \vec{r}) \right]

    :param mu_1: Magnetic moment operators of the first spin (array-like of length 3).
    :param mu_2: Magnetic moment operators of the second spin (array-like of length 3).
    :param xyz: Vector connecting the two spins in 3D space.
    :type mu_1: sequence of magnetic moment operators
    :type mu_2: sequence of magnetic moment operators
    :type xyz: array-like
    :returns: Dipolar Hamiltonian operator.
    :rtype: hermitian matrix as a :py:class:`numpy.ndarray`
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


S = get_spin_matrices(1 / 2, 1, 4)
"""
Erbium spin operators

:meta hide-value:
"""

I1 = get_spin_matrices(1 / 2, 2, 2)
"""
First nuclear spin operators

:meta hide-value:
"""

I2 = get_spin_matrices(1 / 2, 4, 1)
"""
Second nuclear spin operators

:meta hide-value:
"""

mu_S = erbium_gamma[:, None, None] * S
"""
Erbium spin magnetic moment operators

:meta hide-value:
"""

mu_I1 = gamma_w * I1
"""
First tungsten spin magnetic moment operators

:meta hide-value:
"""

mu_I2 = gamma_w * I2
"""
Second tungsten spin magnetic moment operators

:meta hide-value:
"""


@jit
def get_zeeman(B0):
    """
    Compute the Zeeman Hamiltonians for a 3-spin system (1 erbium, 2 tungstens) in a static magnetic field.

    Larmor frequencies are taken from :py:mod:`.constants`.

    :param array-like B0: Static magnetic field vector (in T).
    :returns: Zeeman Hamiltonians for erbium, first tungsten, and second tungsten.
    :rtype: tuple of 3 zeeman hamiltonian
    """
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


@jit
def get_hamiltonian2(r1, r2, B0, gamma_ratio):
    """
    Compute the static Hamiltonian for a 3-spin system (1 erbium, 2 tungstens).

    Includes Zeeman terms and dipolar couplings between each pair of spins, with a
    scaling factor ``gamma_ratio`` for the second tungsten (useful if we want to consider the second tunsten as a niobium).

    :param array-like r1: Position vector of the first tungsten spin.
    :param array-like r2: Position vector of the second tungsten spin.
    :param array-like B0: Static magnetic field vector (in T).
    :param float gamma_ratio: Gyromagnetic ratio scaling for the second tungsten.
    :returns: Hamiltonian operator.
    :rtype: hermitian :py:class:`numpy.ndarray`
    """
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
    """
    Compute the static Hamiltonian for a tungsten–erbium spin system.

    Includes Zeeman terms and dipolar coupling between erbium and one tungsten.

    :param array-like r: Position vector of the tungsten spin.
    :param array-like B0: Static magnetic field vector (in T).
    :returns: Hamiltonian operator.
    :rtype: hermitian :py:class:`numpy.ndarray`
    """
    H_zeeman_erbium, H_zeeman_I1, _ = get_zeeman(B0)
    H_0 = (
        H_zeeman_erbium
        + H_zeeman_I1
        + dipolar_hamiltonian(mu_I1, mu_S, r - erbium_position)
    )
    return H_0


@jit
def nuclei_nuclei_eigval(r1, r2, B, gamma_ratio):
    """
    Compute eigenvalues of the static Hamiltonian for a 2-nucleus, 1-erbium spin system.

    Nuclear spins have a tungsten nuclear gyromagnetic ratio

    Uses :py:func:`get_hamiltonian2` to build the Hamiltonian.

    :param array-like r1: Position vector of the first nuclear spin.
    :param array-like r2: Position vector of the second nuclear spin.
    :param array-like B: Static magnetic field vector (in T).
    :param float gamma_ratio: Gyromagnetic ratio scaling for the second nuclear spin (useful if we want to consider the second nuclear spin as the niobium)
    :returns: Eigenvalues of the system Hamiltonian.
    :rtype: :py:class:`numpy.ndarray`
    """
    H = get_hamiltonian2(r1, r2, B, gamma_ratio)

    eig = np.linalg.eigvalsh(H)
    return eig


@jit
def nuclei_nuclei_coupling(r1, r2, B, gamma_ratio):
    """
    Compute the effective ZZ coupling between two nuclear spins, including
    erbium-mediated renormalization.

    Nuclear spins have a tungsten nuclear gyromagnetic ratio

    Uses :py:func:`nuclei_nuclei_eigval` to extract energy levels and compute the coupling constant.

    :param array-like r1: Position vector of the first nuclear spin.
    :param array-like r2: Position vector of the second nuclear spin.
    :param array-like B: Static magnetic field vector (in T).
    :param float gamma_ratio: Gyromagnetic ratio scaling for the second nucleus.
    :returns: Effective ZZ coupling constant (in Hz).
    :rtype: float
    """
    eig = nuclei_nuclei_eigval(r1, r2, B, gamma_ratio)
    return (eig[0] + eig[3] - eig[1] - eig[2]) / h


@jit
def tungsten_erbium_eigval(r, B):
    """
    Compute eigenvalues of the tungsten–erbium interaction Hamiltonian.

    Uses :py:func:`get_hamiltonian` to build the system Hamiltonian.

    :param array-like r: Position vector of the tungsten spin.
    :param array-like B: Static magnetic field vector (in T).
    :returns: Eigenvalues of the system Hamiltonian.
    :rtype: :py:class:`numpy.ndarray`
    """

    H = get_hamiltonian(r, B)

    eig = np.linalg.eigvalsh(H)
    return eig


@jit
def tungsten_erbium_coupling(r, B):
    """
    Compute the hyperfine coupling constant between a tungsten spin and the erbium.

    Uses :py:func:`tungsten_erbium_eigval` to extract energy levels and compute the coupling.

    :param array-like r: Position vector of the tungsten spin.
    :param array-like B: Static magnetic field vector (in T).
    :returns: Hyperfine coupling constant (in Hz).
    :rtype: float
    """
    eig = tungsten_erbium_eigval(r, B)
    return (
        (eig[0] + eig[1] + eig[7] + eig[6] - eig[2] - eig[4] - eig[3] - eig[5]) / h / 2
    )
