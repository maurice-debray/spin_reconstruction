import numpy as np

mu_0 = 12.5663706127e-7  # r^3/T^2/J
h = 6.62607015e-34  # J/s
gamma_w = 1.1282407e7 * h / 2 / np.pi  # J/T
alpha_w = mu_0 / 4 / np.pi * (gamma_w) ** 2  # J.r^3
gamma_ratio = 1.1282407 / 6.567400
gamma_nb = gamma_w / gamma_ratio  # J/T

mu_B = 9.2740094980e-24  # J/T
erbium_gamma = -mu_B * np.array((8.38, 8.38, 1.247))  # J/T

# https://doi.org/10.1063/1.1725143
a = 5.243e-10  # In meters
c = 11.376e-10  # In meters
# Lattice parameters:
lattice_x = np.array([a, 0, 0])
lattice_y = np.array([0, a, 0])
lattice_z = np.array([0, 0, c])


omega_I = -799.531e3 * 2 * np.pi  # Mean of measured values
omega_S = 7_741_655_681 * 2 * np.pi  # Measured value

lattice_s = np.array([[a, a, c]]) * np.array(
    [
        [0.5, 0.5, 0],
        [0.5, 0, 0.25],
        [0, 0, 0.5],
        [0, 0.5, 0.75],
    ]
)

site_nb = lattice_s.shape[0]

erbium_position = lattice_x * 0.5 + lattice_y * 0.5 + lattice_z * 0.5
