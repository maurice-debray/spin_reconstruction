{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  hatchling,
  matplotlib,
  more-itertools,
  numpy,
  pytest,
  pytest-cov,
  pytest-mpl,
}:

buildPythonPackage rec {
  pname = "matplotlib-label-lines";
  version = "0.8.1";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "cphyc";
    repo = "matplotlib-label-lines";
    rev = "v${version}";
    hash = "sha256-qIuOwf5j4wj79MHx7+ZXro/gLSBZU323REETrvmsAso=";
  };

  build-system = [
    hatchling
  ];

  dependencies = [
    matplotlib
    more-itertools
    numpy
  ];

  optional-dependencies = {
    test = [
      matplotlib
      pytest
      pytest-cov
      pytest-mpl
    ];
  };

  meta = {
    description = "Label line using matplotlib";
    homepage = "https://github.com/cphyc/matplotlib-label-lines";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ ];
  };
}
