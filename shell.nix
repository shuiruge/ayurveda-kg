# shell.nix
with import <nixpkgs> { };
pkgs.mkShell rec {
  name = "impurePythonEnv";
  venvDir = "./.venv";
  buildInputs = [
    # Python virtual environment.
    python3Packages.python
    python3Packages.venvShellHook

    # Python libraries
    python3Packages.spacy
    python3Packages.flask
  ];
}
