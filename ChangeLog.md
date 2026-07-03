# Unreleased (2.0dev)

## New
- ``LENGTH_UNIT`` and ``FORCE_UNIT`` tags (**alm**) for providing the ``&cell`` lattice vectors, DFSET displacements/forces, and ``&cutoff`` radii in units other than the default bohr / Ry/bohr (e.g. angstrom / eV/angstrom). The data are converted to Rydberg atomic units internally. Also available in the Python API as ``ALM(..., length_unit=, force_unit=)``.
- ``FCS_UNIT_OUTPUT`` tag (**alm**) for writing the HDF5 force constant file in eV/angstrom units; the ``unit`` attributes stored in the file are updated accordingly. Also available as ``save_fc(..., fc_unit=)`` in the Python API. The XML and ``.fcs`` outputs always stay in Rydberg atomic units.
- **anphon** now honors the ``unit`` attributes of HDF5 force constant files and converts the data to its internal units automatically; files without the attributes are assumed to be in Rydberg atomic units as before.

# Ver. 1.5.0 (2023-03-31)

## New
- Structural optimization at finite temperatures based on SCPH or QHA.  
> **Note**
> The structural optimization of the current version CANNOT be used with thermal conductivity calculations or bubble corrections.
> The standard SCPH dispersion and thermodynamic properties are correctly calculated for relaxed configurations.

# Ver. 1.4.2 (2022-09-28)

## Fixes

- Fix a bug in group velocity when NONANALYTIC=3


# Ver. 1.4.1 (2022-06-22)

## Fixes

- Fix a bug in xml writer in the previous version


# Ver. 1.4.0 (2022-06-21)

## New
- ``FC_ZERO_THR`` tag (**alm**) for solving #71
- ``STOP_CRITERION`` tag (**alm**) for stopping the cross-validation calculation when appropriate
- ``FC3_SHENGBTE`` tag (**alm**) for saving third-order force constants for ShengBTE format
- ``ZMODE`` tag (**anphon**) for computing mode effective charges
- ``UCORR`` and ``SHIFT_UCORR`` tags (**anphon**) for computing displacement-displacement correlation function
- ``extract.py`` can extract data from vaspout.h5 for VASP6

## Changes

- Update the predefined values of atomic weights and isotope factors
- Add new lines before printing warning/error messages. #61
- Make non-analytic correction of dynamical matrix to be periodic in G
- The file suffix for phonon eigenvectors is .evec, .band.evec, or .mesh.evec depending on the input **KPMODE**.
- ``scph_to_qefc.py`` now accepts ibrav/=0 and .fc files containing the macroscopic field

## Fixes

- Fix minor typos in the documentation pages



# Ver. 1.3.0 (2021-10-01)

## Changes

- For OpenMX, displace.py now generates atomic positions in Cartesian coordinate instead of the fractional coordinate.
- Refactor the codes for future major release (This release does not change the user interface.)

## Fixes

- Fix a few bugs in the SCP class. The stability of the phonon frequency interpolation has been improved.
- Fix other minor bugs and typos in documentation and header parts of output files.


# Ver. 1.2.0 (2021-04-02)

## New features

- CMake installation option
- ``FC_BASIS`` tag (**alm**) for better stability of force constant symmetrization
- ``NONCOLLINEAR`` tag (**alm**) for phonon calculations with noncollinear magnetism
- ``NMAXSAVE`` tag (**alm**) that controls the maximum order of anharmonic terms to be saved in a file
- ``LMODEL = adaptive-lasso`` (**alm**) that performs adaptive LASSO 
- ``KAPPA_COHERENT`` tag (**anphon**) for computing the coherent part of thermal conductivity
- ``ANIME_FRAMES`` tag (**anphon**) that controls the number of frames saved for animation outputs
- ``extract.py`` can extract the dielectric tensor and Born effective charges from vasprun.xml (VASP) and ph.x outputs (Quantum ESPRESSO) by the ``--get born`` option.
- ``fc_virtual.cpp`` which perform a virtual crystal approximation (VCA) like interpolation of force constants

## Changes

- The old tags ``DFILE`` and ``FFILE`` are not obsolete. Use ``DFSET`` instead.
- The filename extension of ``PREFIX``.enet_cv is changed as ``PREFIX``.cvset.
- ``CV_MINALPHA`` and ``CV_MAXALPHA`` are now set automatically (by default)
- The default value of ``CV_NALPHA`` is changed to 50
- The header part in PREFIX.evec has been modified slightly. Please be careful if you are using PREFIX.evec for further analyses.
- ``plotband.py`` now works nicely for discontinuous BZ paths.
- When ``BCONNECT > 0`` and KPMODE = 1, phonon velocities, polarization vectors, and Grüneisen parameters are also reordered before saved in files.
- Support command line usage of ``dfc2``

## Fixes

- The tetrahedron method (``ISMEAR = -1``) now works correctly even when the number of momentum points along each axis is only one. (fix issue #10)
- Fix an issue of MPI communicators when sending large messages (> 2^31-1).
- The parser for LAMMPS now can read *.lammps file that contains charge entries. (fix issue #13)
- Fix a bug in the ``ANIME`` option

# Ver. 1.1.0 (2019-05-01)

## New

- An interface to OpenMX code (contributed by Yuto Tanaka)
- Compressive sensing approach (``LMODEL = enet``) in alm code. Many new variables related to compressive sensing are also added. See the documentation page for details.
- ``SPARSE`` and ``SPARSESOLVER`` tags in alm code
- ``DOS``-tag in anphon code
- A python script scph_to_qe.py that converts the result of an SCPH calculation to Quantum ESPRESSO force-constant format.

## Changes

- The default value of ``ICONST`` is changed to ``ICONST = 11``
- Python scripts now work with python3 as well as python2
- Python interface scripts are moved to tools/interface
- Default values for ``MASS``- and ``ISOFACT``-tags are implemented
- Implement a sparse version of rref, which improves the performance of alm significantly.
- Performance improvements of anphon code.
- ``DFILE`` and ``FFILE`` in alm code are now deprecated. Use ``DFSET`` instead.
- ``&fitting`` field in alm is replaced with ``&optimize`` field.

## Fix

- Fix a minor bug in calc_damping_tetrahedron. The phonon linewidths at high temperatures and the thermal conductivities were not affected by this minor error. In a very low-temperature region (< 10 K), the thermal conductivity may have been underestimated.
- Fix other minor bugs

# Ver. 1.0.1 (2017-11-21)

## Fix
- Fixed a minor issue in the previous version



# Ver. 1.0.2 (2018-1-29)

## New

- Phonon band connection by eigenvector similarity (``BCONNECT`` tag) 
- New option to turn on/off the symmetrization of Born effective charge (``BORNSYM`` tag).

## Changes

- Improve the performance of the "suggest" mode for hexagonal systems
- Use \<unorderd_set\> instead of \<set\> for better performance

## Fix

- Fix a bug in the symmetrization of the Born effective charge


# Ver. 1.0.1 (2017-11-21)

## Fix
- Fixed a minor issue in the previous version


# Ver. 1.0.0 (2017-11-21)

## New

- Self-consistent phonon calculation (``MODE = SCPH``) 
- Ewald summation for the non-analytic term of the dynamical matrix (``NONANALYTIC = 3``).
- Support of the ``CLASSICAL`` option.
- Python auxiliary script for LAMMPS

## Changes

- P+ and P- are printed separately in PREFIX.sps when ``SPS = 1``
- Use C++11 standard. From this version, the C++ compiler must support the C++11 standard.
- The **anphon** code symmetrize the Born effective charges

## Fix

- Loosen the tolerance to detect the multiplicity of force constants. This is an important fix for low symmetry structures.
- Fixed a problem of the restart mode of ``MODE = RTA``


# Ver. 0.9.8 (2016-7-14)

## New

- New tag ``HESSIAN`` in program **alm** for printing entire Hessian matrix
- New tag ``KAPPA_SPEC`` in **anphon** for calculating spectra of thermal conductivity
- New option ``--offset`` in **extract.py** for subtracting residual forces (displacements) in an equilibrium structure from training data sets. We recommended using this option if internal coordinates of atoms (Wyckoff positions) have free parameters.
- New option ``--isotope`` in **analyze_phonons.py**

## Changes

- Improve the performance of thermal conductivity calculations with the tetrahedron method (``ISMEAR=-1``). The new version is <Font color='red'>more than 3 times faster</font> than the previous version.
- Improve the efficiency of the algorithm for generating constraints for the translational invariance
- Avoid 'NaN' in thermal conductivity calculations with imaginary branches
- Loosen the default value of ``TOLERANCE`` for detecting crystal symmetry
- Stop printing the CLASSICAL entry in PREFIX.result files
- Change the unit of the smearing width written in PREFIX.result files (a.u. --> cm^-1)

## Fix

- Fixed an issue regarding the phonon-isotope scattering rate
- Fixed a bug in relaxation.cpp regarding the permutation symmetry of q mesh
- Fixed an invalid memory reference in **analyze_phonons.cpp**
- Added a routine to check the consistency of the crystal structure when the ``FC2XML`` tag is used in **anphon**


# Ver. 0.9.7 (2016-3-10)

## New

- New option ``SPS = 2`` in anphon
- New tag ``MAGMOM`` for considering collinear spin (alm). The format is the same as VASP.
- ALAMODE logo 

## Changes

- From this version, the variable 'NNP' for the translational part of 
  space group operations will no longer be used. Because of this, 
  file formats of SYMM_INFO and SYMM_INFO_PRIM also changed.

- Comment lines in the &cell, &position, and &kpoints fields are supported.

- Any words starting from 'N' or 'n' can be used instead of 'None' for the &cutoff field.

## Fix

- Fixed a bug in fitting.cpp (alm) regarding out-of-range of an array.
- Fixed a bug in the calculation of the maximum distance for multi-body interaction clusters.
- Fixed a bug in the calculation of constraints for translational invariance. 
  This bug appeared in previous versions with some specific structures/cutoff radii.
- anphon code incorrectly printed RMSD instead of MSD in previous versions. Fixed.
- qe2alm.cpp now works with ibrav=0.
- Fixed several issues in displace.py and extract.py for processing QE files.
- ``PRINTPR`` now works properly when KPMODE = 0.


# Ver. 0.9.6 (2015-09-14)

## New

- New option ``ICONST=11`` in **alm** which considers translational invariance algebraically

- ``FC2XML``-tag in **anphon**. Using this tag, it is now possible to employ different size of 
   supercells for harmonic and anharmonic terms.

- qe2alm.cpp in the tools/ directory which converts Quantum-ESPRESSO fc files to ALAMODE xml files.

## Changes

- Changed the meaning of ``--calc=cumulative`` in analyze_phonons.py. 
   The ``--calc=cumulative`` in the previous version is now equivalent to ``--calc=cumulative2``.

- Updated tutorial

## Fix

- Fix bugs related to memory allocation

- Fix an issue in extract.py for QE


# Ver. 0.9.5 (2015-06-28)

## New

- PERIODIC tag for low dimensional systems

## Changes

- Print distances for anharmonic terms in PREFIX.fcs

- Improved the way to define the multiplicity of force constants

- OpenMP parallelization for generating constraints between force constants

## Fix

- Fixed a minor bug in interaction.cpp

- Fixed a bug in displace.py for QE

# Ver. 0.9.4 (2015-02-16)

## New

- New tag ``PRINTPR`` (anphon) for calculating (atomic) participation ratio

- Implement ``ISOTOPE = 2`` to print the selfenergy due to phonon-isotope scatterings (anphon)

## Changes

- Stop printing "nzero" in alm because it may be confusing.

## Fix

- Fixed a bug related to PDOS (anphon)

- Fixed issue in displace.py for QE


# Ver. 0.9.3 (2014-12-12)

## New

- New tag ``FC3XML`` (alm)

- New tag ``SPS`` (anphon) for calculating the scattering phase space  

- New mode ``--calc=kappa_boundary`` in analyze_phonons.py

## Changes

- Changed the filename of TDOS from \*.dos2 to \*.tdos

## Fix

- Fixed a bug in phonon DOS and two-phonon DOS

- Fixed a bug in anphon regarding the NSYM variable

- Fixed a bug related to the isotope scattering rate

- Fixed xml_writer to be compatible with boost 1.56.0

# Ver. 0.9.2 (2014-10-29)

## New

- Python auxiliary tool for Quantum-ESPRESSO

- Tutorial added

- Added displace.py and extract.py in tools/ directory

## Changes

- Different algorithm for anharmonic scattering rates (a little performance improvement is expected)

- ``PRINTVEL = 1`` now print the xyz component

- Renamed phonon_thermodynamics.{cpp,h} as thermodynamics.{cpp,h}

## Fix

- Fixed a bug in analyze_phonons.cpp

- Fixed an MPI-related bug in relaxationc.pp

- Avoid unnecessary memory allocation in relaxation.cpp

- Fixed a bug in the rank-revealing algorithm of alm

# Ver. 0.9.1 (2014-08-24)

## New

- Python auxiliary tool for xTAPP

- alm now checks if ``DFILE`` and ``FFILE`` have enough lines

## Changes

- Change the format of eigenvector *.evec file

- Tmin and Tmax are now inclusive


## Fix

- Fixed many minor bugs

# Ver. 0.9.0 (2014-08-15)

-  The first release of ALAMODE
