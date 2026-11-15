# ALAMODE

Python interfaces to [ALAMODE](https://github.com/alamode-code/alamode), a package for lattice
anharmonicity and lattice thermal conductivity from first principles:

- `alm`: harmonic and anharmonic force constants from displacement-force data;
- `anphon`: phonons, self-consistent phonons and thermal conductivity (uses MPI).

The distribution on PyPI is named `alamode-python`; the modules are imported as `alm` and `anphon`.

```
pip install "alamode-python[openmpi]"
```

The `openmpi` extra installs Open MPI from PyPI, which `anphon` needs. In a conda environment, install
`openmpi` from conda-forge and run `pip install alamode-python` without the extra. The wheels are built without
the optional SuiteSparse solvers of `alm`; build from source if you need them.

Documentation: https://alamode-code.github.io/alamode/

The command-line programs `alm` and `anphon` and the tool scripts are not part of the wheels; see the
documentation for building them, or use the conda-forge packages.
