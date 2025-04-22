{ nixpkgs ? (import ./npins).nixpkgs, pkgs ? import nixpkgs { config.allowUnfree = true; } }:
pkgs.mkShell {
  nativeBuildInputs = [
    (pkgs.python3.withPackages (ps: [
      ps.numpy
      ps.scipy
      ps.ipython
      ps.tqdm
      ps.h5py
      ps.ipympl
      ps.numba
      ps.matplotlib
      ps.pandas
      ps.jupyter
      ps.jupyterlab-git

      ps.mkdocs
      ps.mkdocs-material
      ps.mkdocstrings
      ps.mkdocstrings-python
    ]))

    pkgs.vale
  ];
}
