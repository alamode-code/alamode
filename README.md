# ALAMODE
 
[![License][license-image]][license-url]
[![Doc status][docs-image]][docs-url]

### Version 1.5.0
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
* Applicable to any crystal structures and low-dimensional systems
* Interface to VASP, Quantum-ESPRESSO, OpenMX, xTAPP, and LAMMPS codes
* Parallelization with MPI+OpenMP

### Harmonic properties

* Phonon dispersion
* Phonon DOS, atom-projected phonon DOS
* Two-phonon DOS
* Vibrational thermodynamic functions (heat capacity, entropy, free energy)
* Mean-square displacement
* Animation and visualization of phonon modes (requires VMD or XCrysDen)
* 3-phonon scattering phase space
* Phonon-isotope scattering rate
* Participation ratio for analyzing the localization of phonon modes


### Anharmonic properties
* Gruneisen parameter via cubic force constants
* Lattice thermal conductivity by BTE-RTA
* Cumulative thermal conductivity
* Coherent component of lattice thermal conductivity
* Phonon linewidth due to 3-phonon interactions
* Phonon frequency shift due to 3- and 4-phonon interactions
* Temperature-dependent effective potential method
* Self-consistent phonon calculation

## Prerequisite
* C++ compiler
* LAPACK library
* MPI library
* Boost C++ library (version >= 1.66)
* FFT library
* Eigen3 library
* spglib

## Download

You can download the latest and previous versions of ALAMODE 
at http://sourceforge.net/projects/alamode .

You can also clone the repository as

```
$ git clone https://github.com/ttadano/alamode.git
```

If you download the GitHub version, please use the 'develop' branch.

## Install
The directories alm/, anphon/, and tools/ contain separate Makefiles.
Please modify the Makefiles appropriately by changing variables such as 
CXX, CXXFLAGS, or MPICXX. Then, issuing the "make" command creates the binary for each program. 

As of version 1.2.0, we also support a build by CMake. This option is recommended for all users.
Please see the documentation page for more details.


## Documentation
For more details about ALAMODE including the tutorial, input parameters, and 
output files, please visit the following webpage.

http://alamode.readthedocs.io


## License
Copyright (c) 2014–2022 Terumasa Tadano
This software is released under the MIT license. 
For license rights and limitations, see LICENSE.txt file.

## Author
Terumasa Tadano (National Institute for Materials Science, Japan)

## Contributors

* Ryota Masuki (Univ. Tokyo)
* Tatsuro Nishimoto (Univ. Tokyo)
* Yusuke Oba (Univ. Tokyo)
* Yuto Tanaka (Kanazawa Univ.)
* Atsushi Togo (Kyoto Univ.)


[license-image]: https://img.shields.io/github/license/ttadano/alamode.svg
[license-url]:  https://github.com/ttadano/alamode/blob/develop/LICENSE.txt

[docs-image]:  https://readthedocs.org/projects/alamode/badge/?version=latest
[docs-url]: https://alamode.readthedocs.io/en/latest/?badge=latest

