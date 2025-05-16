{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  jupyter-packaging,
  jupyterlab,
  breakpointHook,

  jupyter-server,
  h5grove,
  h5py,
  nodejs,
  fetchYarnDeps,
  yarnConfigHook,
  yarn-berry_3,
}:

let
  yarn-berry = yarn-berry_3;
in

buildPythonPackage rec {
  pname = "jupyterlab-h5web";
  version = "12.4.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "silx-kit";
    repo = "jupyterlab-h5web";
    rev = "v${version}";
    hash = "sha256-yWHfV+ZtupFwzT5VlyAB1ob9gEbo8/eWrMAQPe9KAU4=";
  };

  patches = [ ./01-bump-deps.patch ];

  offlineCache = yarn-berry.fetchYarnBerryDeps {
    inherit patches src;
    hash = "sha256-CH+YBvT3BhqVNt9WX5BZUk1cMl8iHSSWDqPwagXJfEk=";
  };

  nativeBuildInputs = [
    yarn-berry.yarnBerryConfigHook
    breakpointHook
    yarn-berry
    nodejs
  ];

  build-system = [
    jupyter-packaging
    jupyterlab
  ];

  dependencies = [
    jupyter-server
    h5grove
    h5py
  ];

  pythonImportsCheck = [
    "jupyterlab_h5web"
  ];

  meta = {
    description = "A JupyterLab extension to explore and visualize HDF5 file contents. Based on https://github.com/silx-kit/h5web";
    homepage = "https://github.com/silx-kit/jupyterlab-h5web";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ ];
  };
}
