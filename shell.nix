# shell.nix
with import <nixpkgs> { };
let
  pythonPackages = python39Packages;
in pkgs.mkShell rec {
  name = "impurePythonEnv";
  venvDir = "./.venv";
  buildInputs = [
    # Python virtual environment.
    pythonPackages.python
    pythonPackages.venvShellHook

    # Python libraries
    pythonPackages.spacy
    pythonPackages.jupyter
  ];

  # Now we can execute any commands within the virtual environment.
  # This is optional and can be left out to run pip manually.
  postShellHook = ''
    # Setup TensorFlow environment.
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages_10_1.cudatoolkit}/lib:${pkgs.cudaPackages_10_1.cudatoolkit}/lib:${pkgs.cudaPackages_10_1.cudatoolkit.lib}/lib:$LD_LIBRARY_PATH
    alias pip="PIP_PREFIX='$(pwd)/.build/pip_packages' TMPDIR='$HOME' \pip"
    export PYTHONPATH="$(pwd)/.build/pip_packages/lib/python3.9/site-packages:$PYTHONPATH"
    export PATH="$(pwd)/.build/pip_packages/bin:$PATH"
    unset SOURCE_DATE_EPOCH

    # Setup local virtual environment.
    pip install -r requirements.txt -i https://pypi.douban.com/simple
  '';
}
