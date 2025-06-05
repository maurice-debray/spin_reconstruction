"""
Script to compute dipolar and hyperfine couplings for spins slightly moved around an ansatz configuration.

The user must supply the following parameters via the configuration file (`fine_couplings.conf.toml`):
- **angle.B_x** and **angle.B_y**: angular components (in degrees) of the magnetic field B in the xy plane.
- **displacement.distance**: physical spacing (in lattice units) between neighboring displacement grid points.
- **displacement.size**: number of points along each axis for the displacement grid (e.g., 5 for a 5×5×5 grid).
- **ansatz.config**: list of indices for the lattice positions of tungsten ions to consider.
- **ansatz.to_compute**: list of (x, y) index pairs specifying which tungsten pairs to compute WW couplings for.
- **ansatz.max_distance**: size of the lattice (in number of cells) for unravel the spins indices.
- **filename**: name of the output HDF5 file where all coupling data will be saved.

HDF5 File Structure:
- Global attributes: parameter values and Git commit hash.
- Groups:
  - **SEDOR_couplings/**: dipolar couplings between tungsten pairs.
  - **A_par_couplings/**: hyperfine couplings between tungsten and erbium.
  - **Nb_par_couplings/**: tungsten-niobium couplings for 0 displacement (kept for backward compat).
  - **Nb_par_couplings_full/**: full matrix of tungsten-niobium couplings.

Each dataset is annotated with the corresponding configuration indices.

Usage:
------
This script is designed to be run as a standalone program:
    python W_fine_couplings.py

It will abort if a file named `filename` already exists to prevent overwriting.
"""

import os.path

import h5py
import numpy as np
from tqdm import tqdm, trange

from gitlock import get_commit_hash, get_config
from spin_reconstruction.constants import (erbium_gamma, erbium_position,
                                           gamma_w, lattice_s, lattice_x,
                                           lattice_y, lattice_z, omega_I,
                                           omega_S)
from spin_reconstruction.fine_couplings import (WW_couplings,
                                                erbium_tungsten_couplings,
                                                nb_couplings)
from spin_reconstruction.utils import index_to_position, origin_index

# Import setting
B_x_angle_rad = get_config("fine_couplings", ["angle", "B_x"]) / 180 * np.pi
B_y_angle_rad = get_config("fine_couplings", ["angle", "B_y"]) / 180 * np.pi

displacement_distance = get_config("fine_couplings", ["displacement", "distance"])
displacement_size = get_config("fine_couplings", ["displacement", "size"])

ansatz_config = get_config("fine_couplings", ["ansatz", "config"])
# FIXME: this has nothing to do under ansatz section
sedor_pairs_to_compute = get_config("fine_couplings", ["ansatz", "to_compute"])
ansatz_max_distance = get_config("fine_couplings", ["ansatz", "max_distance"])

filename = get_config("fine_couplings", ["filename"])

if __name__ == "__main__":

    if os.path.isfile(filename):
        raise ValueError(f"A file named {filename} already exists")

    git_commit = get_commit_hash()

    with h5py.File(filename, "w") as f:
        B_cartesian = np.array([B_x_angle_rad, B_y_angle_rad, 1])
        B_unit_vector = B_cartesian / np.linalg.norm(B_cartesian)

        attrs = {
            "git_commit": git_commit,
            "max_distance": ansatz_max_distance,
            "lattice_x": lattice_x,
            "lattice_y": lattice_y,
            "lattice_z": lattice_z,
            "lattice_s": lattice_s,
            "erbium_position": erbium_position,
            "erbium_gamma": erbium_gamma,
            "omega_I": omega_I,
            "omega_S": omega_S,
            "gamma_w": gamma_w,
            "distance": displacement_distance,
            "size": displacement_size,
            "config": ansatz_config,
            "B_x": B_x_angle_rad,
            "B_y": B_y_angle_rad,
            "B": B_unit_vector,
        }
        for k, v in attrs.items():
            f.attrs[k] = v

        sedor_group = f.create_group("SEDOR_couplings")
        a_par_group = f.create_group("A_par_couplings")
        nb_par_group = f.create_group("Nb_par_couplings")
        full_nb_par_group = f.create_group("Nb_par_couplings_full")
        for i in trange(len(ansatz_config)):
            grp_name = f"{i}"

            tungsten_pos1 = index_to_position(
                ansatz_config[i],
                ansatz_max_distance,
            )

            erbium_hyperfine = erbium_tungsten_couplings(
                tungsten_pos1, displacement_size, displacement_distance, B_unit_vector
            )

            nb_coupling_matrix = nb_couplings(
                tungsten_pos1, displacement_size, displacement_distance, B_unit_vector
            )

            center_displacement_index = origin_index(displacement_size)

            hyperfine_dataset = a_par_group.create_dataset(
                name=grp_name, data=erbium_hyperfine
            )

            # LEGACY compatibility. Should be fast to compute
            nb_fixed_dataset = nb_par_group.create_dataset(
                name=grp_name, data=nb_coupling_matrix[center_displacement_index]
            )

            nb_free_dataset = full_nb_par_group.create_dataset(
                name=grp_name, data=nb_coupling_matrix
            )

            for dataset in (hyperfine_dataset, nb_fixed_dataset, nb_free_dataset):
                dataset.attrs["tungsten_position_index"] = i
                # LEGACY compat
                dataset.attrs["x"] = i

        for idx_1, idx_2 in tqdm(sedor_pairs_to_compute):
            grp_name = f"{idx_1}_{idx_2}"

            tungsten_pos1 = index_to_position(
                ansatz_config[idx_1],
                ansatz_max_distance,
            )
            tungsten_pos2 = index_to_position(
                ansatz_config[idx_2],
                ansatz_max_distance,
            )

            WW_couplings_matrix = WW_couplings(
                tungsten_pos1,
                tungsten_pos2,
                displacement_size,
                displacement_distance,
                B_unit_vector,
            )

            sedor_pair_dataset = sedor_group.create_dataset(
                name=grp_name, data=WW_couplings_matrix
            )

            attrs = {
                "x": idx_1,
                "tungsten_position_index_1": idx_1,
                "y": idx_2,
                "tungsten_position_index_2": idx_2,
            }

            for k, v in attrs.items():
                sedor_pair_dataset.attrs[k] = v
