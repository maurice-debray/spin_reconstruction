{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  setuptools,
  wheel,
  numpy,
  orjson,
  h5py,
  tifffile,
  typing-extensions,
}:

buildPythonPackage rec {
  pname = "h5grove";
  version = "2.3.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "silx-kit";
    repo = "h5grove";
    rev = "v${version}";
    hash = "sha256-2pfTK4R2pdndNnc5DCoLyUGqULodSu0sHbHb3TL3cqg=";
  };

  build-system = [
    setuptools
    wheel
  ];

  dependencies = [

    numpy
    orjson
    h5py
    tifffile
    typing-extensions
  ];

  pythonImportsCheck = [
    "h5grove"
  ];

  meta = {
    description = "H5grove is a Python package that provides utilities to design backends serving HDF5 file content: attributes, metadata and data";
    homepage = "https://github.com/silx-kit/h5grove";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ ];
  };
}
