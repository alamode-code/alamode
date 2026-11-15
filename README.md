# ALAMODE
 
[![License][license-image]][license-url]
[![Doc status][docs-image]][docs-url]

### Version 2.0.1
![alt ALAMODE](./docs/img/alamode.png)


- - -

## Introduction 

ALAMODE is a scientific software designed for analyzing lattice anharmonicity
and lattice thermal conductivity of solids. By using an external DFT package
such as VASP and Quantum ESPRESSO, you can extract harmonic and anharmonic
force constants straightforwardly with ALAMODE. Using the calculated anharmonic
force constants, you can also estimate lattice thermal conductivity, phonon
linewidth, and other anharmonic phonon properties from first principles.

## Features

### General
* Extraction of harmonic and anharmonic force constants based on the supercell approach
* Least-squares fitting or sparse regression (elastic net, adaptive LASSO) with cross-validation
* Applicable to arbitrary crystal structures and low-dimensional systems
* Accurate treatment of translational, rotational, and Huang invariance
* Interfaces to the VASP, Quantum ESPRESSO, OpenMX, xTAPP, and LAMMPS codes
* Force-constant conversion to and from phonopy/phono3py, ShengBTE, and FourPhonon
* HDF5 files for force constants, results, and restarts
* Parallelization with MPI+OpenMP
* Python interfaces to *alm* and *anphon*

### Harmonic properties
* Phonon dispersion, phonon DOS, atom-projected phonon DOS
* Vibrational thermodynamic functions (heat capacity, entropy, free energy)
* Mean-square displacement, group velocities, participation ratio
* Two-phonon DOS, three-phonon scattering phase space
* Visualization and animation of phonon modes (requires VMD or XCrySDen)
* Irreducible representations and IR/Raman activity at the Gamma point
* Dielectric function and mode effective charges

### Anharmonic properties
* Mode and generalized Gruneisen parameters
* Lattice thermal conductivity in the RTA or by iterative, variational, or direct solution of the BTE
* Three- and four-phonon scattering
* Coherent (off-diagonal) thermal conductivity, thermal conductivity spectrum
* Isotope and boundary scattering
* Phonon linewidth, frequency shift, and spectral function at arbitrary q points
* Temperature-dependent phonons by the self-consistent phonon (SCPH) method
* Anharmonic free energy (SCPH and bubble correction)
* Crystal structure and thermal expansion at finite temperatures (SCPH or QHA), including cell relaxation
* Thermal conductivity with SCPH or QHA phonons on the relaxed structures

## Prerequisite
* C++ compiler supporting C++17 and OpenMP
* LAPACK and BLAS libraries
* MPI library
* Boost C++ library (version >= 1.66)
* Eigen3 library
* spglib
* HDF5 library
* CMake (version >= 3.17)
* Python (>= 3.9) with NumPy, h5py, PyYAML, and Matplotlib for the scripts and the Python package (optional)

## Download

Download a release from https://github.com/alamode-code/alamode/releases , or clone the repository as

```
$ git clone -b master https://github.com/alamode-code/alamode.git
```

## Install
ALAMODE is built with CMake:

```
$ mkdir _build && cd _build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j
```

The Python package (modules `alm` and `anphon`) is installed from the repository root by `pip install ./python`.
The legacy Makefiles were removed in version 2.0.
Please see the documentation for the build options, and the migration guide if you are coming from 1.x.


## Documentation
For more details about ALAMODE including the tutorial, input parameters, and 
output files, please visit the following webpage.

https://alamode-code.github.io/alamode/


## Versioning
Starting with 2.0.0, ALAMODE follows Semantic Versioning (MAJOR.MINOR.PATCH). Incompatible changes to input tags, file formats, or the Python API increase MAJOR; new features increase MINOR; bug fixes increase PATCH.


## License
Copyright (c) 2014–2026 Terumasa Tadano
This software is released under the MIT license. 
For license rights and limitations, see LICENSE.txt file.

## Author
Terumasa Tadano (National Institute for Materials Science, Japan)

## Contributors

* Ryota Masuki
* Tatsuro Nishimoto
* Yusuke Oba
* Yuto Tanaka
* Atsushi Togo
* Wenhao Zhang


[license-image]: https://img.shields.io/github/license/alamode-code/alamode.svg
[license-url]:  https://github.com/alamode-code/alamode/blob/master/LICENSE.txt

[docs-image]:  https://img.shields.io/badge/docs-online-blue.svg
[docs-url]: https://alamode-code.github.io/alamode/

