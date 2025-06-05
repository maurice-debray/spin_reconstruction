#!/usr/bin/env python
"""
Script to compute full couplings (dipolar and hyperfine) for a grid of magnetic field orientations.

The user must supply the following parameters via the configuration file (`couplings.conf.toml`):
- **range.x_start**, **range.y_start**: Starting indices for the magnetic field sweep grid along x and y.
- **range.x_end**, **range.y_end**: Ending indices for the sweep grid along x and y.
- **angle.x_max**, **angle.y_max**: Maximum field angles (in degrees) along x and y.
- **range.x_size**, **range.y_size**: Number of discrete values along each axis in the sweep grid.
- **lattice.max_distance**: Size (in lattice units) of the cubic region in which spins are placed for coupling calculations.
- **filename**: Name of the HDF5 output file where computed data will be saved.

HDF5 file structure:
- Global attributes: configuration values and Git commit hash.
- One group per magnetic field orientation, named `B_sweep_{x}_{y}`.
  - Each group contains three datasets:
    - `SEDOR_couplings`: dipolar couplings between tungsten spins.
    - `A_par_couplings`: hyperfine couplings between tungsten and erbium.
    - `NB_couplings`: couplings between tungsten and niobium.
  - Each group is annotated with the associated magnetic field vector and sweep indices.

Usage:
------
This script is designed to be run as a standalone program:
    python W_couplings.py

It will abort if a file named `filename` already exists to avoid overwriting existing results.
"""

import os.path

import h5py
import numpy as np
from tqdm import trange

from gitlock import get_commit_hash, get_config
from spin_reconstruction.constants import (erbium_gamma, erbium_position,
                                           gamma_w, lattice_s, lattice_x,
                                           lattice_y, lattice_z, omega_I,
                                           omega_S)
from spin_reconstruction.couplings import all_couplings

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

if __name__ == "__main__":

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
                WW_couplings, a_parallel, nb_par = all_couplings(
                    max_distance=max_distance, B=B_0
                )

                g = f.create_group(grp_name)
                g.create_dataset(name="SEDOR_couplings", data=WW_couplings)
                g.create_dataset(name="A_par_couplings", data=a_parallel)
                g.create_dataset(name="NB_couplings", data=nb_par)

                attrs = {
                    "B": B_0,
                    "B_x": x,
                    "B_y": y,
                    # LEGACY
                    "x": x,
                    "y": y,
                }

                for k, v in attrs.items():
                    g.attrs[k] = v
