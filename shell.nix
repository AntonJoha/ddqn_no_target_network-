{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python3.withPackages (ps: with ps; [
    jupyterlab
    ipykernel
    matplotlib
    numpy
    pandas
    torch
    gymnasium
    seaborn

    array-api-compat
    dill
    flax
    jax
    jaxlib
    matplotlib
    moviepy
    mujoco
    opencv4
    pybox2d
    pygame
    pytestCheckHook
    scipy
    torch

    ruff
  ]);
in
pkgs.mkShell {
  buildInputs = [
    python
    
    # often needed for geospatial stacks (osmnx -> geopandas -> fiona etc.)
    pkgs.geos
    pkgs.gdal
    pkgs.proj
  ];

  shellHook = ''
    echo "Starting Jupyter environment..."
    echo "Run: jupyter lab"
  '';
}
