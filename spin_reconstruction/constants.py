"""Some constants in SI units in"""

import numpy as np

#: Vacuum magnetic permeability in :math:`r^3/T^2/J`
mu_0 = 12.5663706127e-7

#: Planck constant in :math:`J/s`
h = 6.62607015e-34

#: Tungsten gyromagnetic factor in :math:`J/T`
gamma_w = 1.1282407e7 * h / 2 / np.pi  # J/T

alpha_w = mu_0 / 4 / np.pi * (gamma_w) ** 2  # J.r^3

#: Ratio between tungsten and niobium gyromagnetic factors
gamma_ratio = 1.1282407 / 6.567400

#: Niobium gyromagnetic factor in :math:`J/T`
gamma_nb = gamma_w / gamma_ratio  # J/T

#: Bohr magneton in :math:`J/T`
mu_B = 9.2740094980e-24

#: Erbium gyromagnetic ratio in :math:`J/T` as a :py:class:`np.ndarray` of diagonal elements
erbium_gamma = -mu_B * np.array((8.38, 8.38, 1.247))  # J/T

#: First CaWO4 lattice constant in meters
#: (`https://doi.org/10.1063/1.1725143 <https://doi.org/10.1063/1.1725143>`_)
a = 5.243e-10

#: Second CaWO4 lattice constant in meters
#: (`https://doi.org/10.1063/1.1725143 <https://doi.org/10.1063/1.1725143>`_)
c = 11.376e-10

#: First CaWO4 lattice vector (in meters)
lattice_x = np.array([a, 0, 0])

#: Third CaWO4 lattice vector (in meters)
lattice_y = np.array([0, a, 0])

#: Third CaWO4 lattice vector (in meters)
lattice_z = np.array([0, 0, c])

#: Tungsten larmor pulsation (mean of measured values **FIXME insert source**)
omega_I = -799.531e3 * 2 * np.pi

#: Erbium larmor pulsation (**FIXME insert source**)
omega_S = 7_741_655_681 * 2 * np.pi

#: CaWO4 lattice sites vectors
lattice_s = np.array([[a, a, c]]) * np.array(
    [
        [0.5, 0.5, 0],
        [0.5, 0, 0.25],
        [0, 0, 0.5],
        [0, 0.5, 0.75],
    ]
)

#: Number of sites per CaWO4 unit cell
site_nb = lattice_s.shape[0]

#: Erbium spin position
erbium_position = lattice_x * 0.5 + lattice_y * 0.5 + lattice_z * 0.5
