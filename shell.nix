{ nixpkgs ? (import ./npins).nixpkgs, pkgs ? import nixpkgs { config.allowUnfree = true; } }:

pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: [
      (ps.callPackage ./npins/mpl-label-lines.nix {})
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
      ps.qutip


      ps.mkdocs
      ps.mkdocs-material
      ps.mkdocstrings
      ps.mkdocstrings-python
    ]))

    pkgs.vale
  ];
}
