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

The package needs Python 3.11 or newer. Wheels are provided for

- Linux x86_64 and aarch64 with glibc 2.27 or newer;
- macOS 15 or newer on Apple silicon.

There is no source distribution on PyPI, so on other platforms `pip` finds no matching distribution; build the
package from the ALAMODE sources instead (CMake 3.26 or newer; see the documentation).

ALAMODE 2.0.0 and 2.0.1 built from source installed the same modules under the distribution name `alamode`.
Remove it before installing this one: `pip uninstall alamode`.

Documentation: https://alamode-code.github.io/alamode/

The command-line programs `alm` and `anphon` and the tool scripts are not part of the wheels; see the
documentation for building them, or use the conda-forge packages.
