{ nixpkgs ? (import ./npins).nixpkgs, pkgs ? import nixpkgs { config.allowUnfree = true; } }:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: let
      mpl-label-lines = ps.callPackage ./npins/mpl-label-lines.nix {};
      jupyterlab-h5web = ps.callPackage ./npins/jupyterlab-h5web.nix { inherit h5grove; };
      h5grove = ps.callPackage ./npins/h5grove.nix {};
    in [
      mpl-label-lines
      jupyterlab-h5web
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
