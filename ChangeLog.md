# Ver. 2.0.1 (2026-11-15)

## Bug fixes

- **anphon** Reading `MagneticMoments` from an XML force-constant file with a supercell larger than the primitive cell read past the end of the array (segmentation fault, or undefined moments in the symmetry search). Each supercell atom now takes the moment of its primitive atom. HDF5 force-constant files and version 1.5.0 were not affected (GitHub #340, thanks to masato1122).
- **alm** Inter-order rotational constraints (`ICONST = 2-4, 12-14`, `NORDER >= 2`) had the wrong sign of the higher-order IFC terms relative to the lower-order ones, so the fitted IFCs violated the rotational invariance (Si NORDER = 3: fitting error 7.3 % -> 4.0 %). IFCs fitted with these settings should be refitted.

# Ver. 2.0.0 (2026-10-30)

ALAMODE follows Semantic Versioning from this release on: incompatible changes to input tags, file formats, or the Python API increase the major version.

## Breaking changes and changed defaults

- **anphon** `FCSXML`/`FC2XML` renamed `FCSFILE`/`FC2FILE`; the old names are rejected.
- **anphon** `NKD` and `NSYM` removed (inputs with them stop); `KD` optional, taken from FCSFILE.
- **anphon** New `&kappa` field: `ISOTOPE`, `ISOFACT`, `KAPPA_SPEC`, `KAPPA_COHERENT` moved there from `&analysis` (rejected in `&analysis`); `RESTART`/`RESTART_4PH` moved there (still accepted in `&general` with a warning).
- **anphon** `MODE = RTA` renamed `MODE = kappa` with the `SOLVER` tag; `MODE = RTA` accepted with a warning.
- **anphon** `FSTATE_K` removed; `kplane` option removed.
- **anphon** `ISOTOPE = 1` by default (the kappa of natural Si at 300 K is about 7 % lower than with `ISOTOPE = 0`).
- **anphon** `TREVSYM` (default 1): time-reversal symmetry in the k-point reduction; restart files of the other setting are rejected.
- **anphon** SCPH mixing default `IMIX = 1` (DIIS); `IMIX = 0` = 1.5 simple mixing.
- **anphon** HDF5 restart/result files by default: PREFIX.kappa.h5, PREFIX.scph.h5, PREFIX.qha.h5; `FILE_FORMAT = text` keeps the legacy .result/.scph_dymat/.renorm_harm_dymat (deprecated); legacy files are imported once.
- **anphon** `PRINTEVEC` writes `.evec.h5` (was `.evec.hdf5` + text `.evec`); text only with `FILE_FORMAT = text`; NEWFCS writes `PREFIX_+.h5`/`PREFIX_-.h5`.
- **anphon** Unconverged SCPH/QHA temperatures recorded; later runs refuse them unless `ALLOW_UNCONVERGED = 1`; their thermo/DOS/MSD rows are NaN unless `ALLOW_UNCONVERGED = 1`.
- **anphon** `RENORM_2TO1ST`/`RENORM_34TO1ST`/`RENORM_3TO2ND` replaced by `STRAIN_COUPLING` (default 7); old tags accepted with a warning, they will be removed in a later release.
- **anphon** `COOLING_U0_INDEX`/`COOLING_U0_THR` deprecated and ignored; `SET_INIT_STR = 3` detects the high-symmetry phase with spglib.
- **anphon** `KS_INPUT` deprecated in favour of `MODE = selfenergy`.
- **anphon** Group velocities for kappa from the velocity matrix of degenerate blocks (basis invariant), little-group symmetrization, zero Gamma translational velocities; restarting from an older kappa.h5 recomputes kappa.
- **anphon** `BCONNECT` matches bands by eigenvectors and frequencies (frequency only across path breaks) and applies to SCPH/QHA bands (GitHub #309).
- **anphon** HDF5 FCS `unit` attributes honoured.
- **anphon** `&cell` optional when the HDF5 FCSFILE contains the primitive cell; stricter cell checks (cell must tile the FCS supercell, handedness, skewed supercell rejected with NONANALYTIC = 2, exact BORNINFO count).
- **anphon** `.scph_dfc2` corrections are relative to the full harmonic dynamical matrix (header `DELTA_BASELINE = 1`); 1.x `.scph_dymat` converted for NONANALYTIC = 2/3, refused for 1.
- **alm** `PREFIX.h5` is the only default output; `.xml` and `.fcs` need `FCS_ALAMODE = 1`.
- **alm** `FCSYM_BASIS = Cartesian` by default (was Lattice); XML FC2FIX/FC3FIX files keep their recorded basis; warning for imprecise input structures.
- **alm** `FC2XML`/`FC3XML` renamed `FC2FIX`/`FC3FIX` (old names warn).
- **alm** `NAT`, `NKD` optional (inferred from &position / KD / STRUCTURE_FILE).
- **alm** Displacement-pattern files in YAML by default; `FORMAT_PATTERN = old` keeps the 1.5 format.
- **alm** ShengBTE outputs named `PREFIX.FORCE_CONSTANTS_3RD`/`_4TH` (was `FORCE_CONSTANT_3RD`); warn when NORDER is too small.
- **alm** Numerical `ICONST = 2-4` drop nearly dependent constraint rows by `TOL_CONST`; algebraic `ICONST = 12-14` with `ALGO_REDUCTION = 3` use scale-free rank + threshold pivoting and now match 2-4.
- **alm** Python API default `TOL_CONST = 1e-6`, as in the CLI.
- **alm** Order of rows in `/ForceConstants` independent of round-off; FC orders without nonzero IFCs written as empty datasets.
- **alm/anphon/tools** CODATA 2018 physical constants throughout.
- **alm/anphon** Stricter input validation: malformed inputs that were accepted or crashed are errors (DFSET 6 finite numbers per line, ICONST whitelist, range checks, numeric parsing, `&kpoint` numbers, EMIN/EMAX honoured by DOS, KMESH_COARSE checks, extra &cutoff columns warn, only tokens starting with N mean None in &cutoff).
- **alm/anphon** WARNING/ERROR messages word-wrapped at 90 columns.
- **tools** `analyze_phonons.{py,cpp}` replaced by `analyzer.py` (reads `--h5 PREFIX.kappa.h5`, `--3ph PREFIX.result`).
- **tools** `makedisp_qe.py` removed (use `displace.py --QE`).
- **tools** `displace.py --random_normalcoord`: sigma capping by nearest-neighbour distance, 4 sigma truncation, 10 cm^-1 frequency floor, fixed default seed.
- **docs** Hosted on GitHub Pages with the Breeze theme.

## New features

### anphon
- Four-phonon scattering in kappa (`INCLUDE_4PH`, separate `KMESH_COARSE`, `ISMEAR_4PH`, `EPSILON_4PH`, `INTERPOLATOR`, `WRITE_INTERPOL`, `RESTART_4PH`, `.kl3`/`.kl4`), factorized 4ph kernel.
- Quartic IFCs can be given separately with `FC4FILE` (ALAMODE XML/HDF5; convert FourPhonon files with `fcconvert.py`).
- Beyond-RTA solvers: `SOLVER = IBTE | VBTE | DBTE` with `MAX_CYCLE`, `MIN_CYCLE`, `ITER_THRESHOLD`, `IBTE_MIXING`, isotope in-scattering (`ISOTOPE_INSCATTERING`), sparse collision operator, irreducible-wedge iteration, warm restart and per-temperature storage in kappa.h5.
- Adaptive smearing `ISMEAR = 2` with `ADAPTIVE_FACTOR`.
- Boundary scattering `LEN_BOUNDARY`.
- `MODE = selfenergy` with `&selfenergy` field: self-energies at arbitrary q, band paths, `PREFIX.selfenergy.h5`.
- `IRREPS`: Gamma-point irreps and IR/Raman activity.
- `GRUNEISEN = 2, 3`: generalized Grüneisen parameters; `PREFIX.gru_kpoints`.
- Elastic constants (second and third order) from IFCs, `SUBLATTICE_RELAX`.
- `PRINTEVAL`; `DISPMODE = 2` for initial displacements along normal modes.
- `FC2_TEMPERATURE` and `DFC2FILE`: SCPH/QHA force constants used directly by later runs; temperature-resolved kappa.h5.
- `RELAXED_STRUCTURE = 1`: run on the structure relaxed by SCPH/QHA (FC3 deformed as Phi3 + Phi4:d).
- `BUBBLE = 1-3` allowed with `RELAX_STR > 0`; `KMESH_BUBBLE`; robust QP root search for `BUBBLE = 3`.
- `BUBBLE = 4`: curvature of the SCP free energy (strain block, finite Q, unstable directions, `PREFIX.scph_hessian*`, `PREFIX.scph_fe.h5`), `BUBBLE_HESS`, `V4_REAL_SPACE`.
- `FE_BUBBLE`: bubble correction to the vibrational free energy with `MODE = phonons` and `MODE = SCPH` (SCPH phonons of each temperature); available without the `_FE_BUBBLE` build flag of 1.5 and documented.
- `CV_ANHARM` anharmonic Cv correction.
- Structural optimization: `RELAX_STR = 4` (fixed-strain coordinate relaxation), Farkas-III optimizer, GDIIS safeguards on by default (`GDIIS_PLAIN = 1` disables them), `GRADIENT_CONV_TOL`, `CELL_GRADIENT_CONV_TOL`, bounded optimization steps.
- `STRAINFILE`: one HDF5 container for the cell-relaxation inputs; GPa units in text inputs; `&reference_cell`.
- `STRAIN_FC5`, `STRAIN_FC5_CHANNELS`: strain derivative of FC4 from FC5.
- `INTERPOLATOR = modified-log-linear`.
- `TREVSYM` (see above); `TRISYM` honoured in the BTE collision operator.
- Input tags and git commit echoed to stdout and to the h5 files.
- `VERBOSITY`-controlled stage timers.

### alm
- HDF5 force-constant files `PREFIX.h5` with `COMPRESSION`; magnetic moments saved.
- `SUPERCELL`, `PRIMCELL` (incl. `PRIMCELL = Auto`), `STRUCTURE_FILE`.
- `SYMMETRIZE_STRUCTURE`.
- `LENGTH_UNIT`, `FORCE_UNIT`, `FCS_UNIT_OUTPUT`.
- Huang invariance `ICONST = 4, 14`; algebraic rotational invariance `ICONST = 12, 13`, usable with enet/adaptive LASSO.
- `STRESS` tag for stress-aware Huang conditions.
- `ALGO_REDUCTION` (0-4) for the constraint reduction; SuiteSparse SPQR backend.
- L1 solvers `L1_SOLVER = cd | fista | admm`; `USE_CHOLESKY`, `CHUNKSIZE`; MINRES and SuiteSparse KKT solvers.
- `CV_MINALPHA_RATIO`; `adalasso` alias; `DEBIAS_OLS = 1` with adaptive LASSO; parallel CV.
- `FC4_SHENGBTE` output.

### Python
- One distribution `alamode` (`pip install ./python`) with packages `alm` (nanobind wrapper, sensing-matrix access for external regressors, FC2FIX, units, ROTAXIS, stress) and `anphon` (experimental: `anphon.run`, `Result`, `Phonon.frequencies`).
- `alm` compatible with phonopy's ALM interface.
- Errors raised as Python exceptions instead of exiting.

### tools
- `analyzer.py` (replaces analyze_phonons), reads kappa.h5, modified-log-linear interpolation, |v| and mfp output.
- `fcconvert.py`: FC2/FC3 between ALAMODE, phonopy/phono3py and ShengBTE, FC4 between ALAMODE and FourPhonon; `fcsinfo.py`.
- Strain tools `strainifc.py`, `elastic.py`, `strainfile.py`.
- `dfc2.py` (HDF5 dfc2), also reading `.scph.h5`/`.qha.h5`; `scph_to_qefc.py` reads state files and writes QE XML.
- `calcpes.py` (energies from IFCs, comparison with DFSET), `taylor.py` with optional numba kernel.
- `displace.py`: random seed, robust random normal-coordinate sampling, `--softmode-only`, flexible `--pes`, builds the alm supercell for QE/OpenMX/LAMMPS.
- `extract.py --vca`; FORCE_CONSTANTS writer for phonopy; `plotband.py`/`plotdos.py -o`.
- `qe2alm` supports the newer QE force-constant format.

## Performance

- **alm** Constraint reduction overhaul (coord factorization, SPQR, forward elimination + back-substitution, OpenMP), much faster for large models.
- **alm** Faster sensing-matrix construction, CD Gram via GEMM, standardized adaptive LASSO, normal-equation OLS weights; efficient cluster generation.
- **anphon** SCPH: DIIS, V4 kernel memory halved, row-distributed V4 over MPI, GEMM-based V3/V4 builders, zheevd for large cells, O(nk ns^3) forces/stress and free-energy postprocess (setup 325 -> 70 s, postprocess 373 -> 3 s per T on a clathrate).
- **anphon** Faster 3ph/4ph scattering rates, OpenMP over triplets/quartets, batched tetrahedron weights.
- **anphon** 4ph kernel skips the (c d) atom pairs with no fourth-order IFC (large cells: Re4Si7 455 -> 348 s, 1.5x per mode for a 54-atom clathrate; unchanged for 2-atom cells).
- **anphon** Isotope scattering with the tetrahedron method much faster (Re4Si7 131 -> 7 s).
- **anphon** k-parallel harmonic diagonalization and interpolation; diagonalize once across ranks.
- **anphon** Improved Ewald summation; FFTW replaced by direct DFT.

## Bug fixes

- **anphon** Cell volume and distance list, isotope, min/max frequency, DOS writer, evec.hdf5 writer bugs.
- **anphon** Group velocities wrong at branch crossings/degeneracies and in supercells.
- **anphon** Boundary scattering and adaptive smearing in the IBTE solver; 4ph scattering rate bug; inconsistent kl4 between anphon and analyzer.py.
- **anphon** IFC lattice vectors taken from the run's primitive cell (FC2 from SCPH state + FC3 from alm gave wrong kappa, Ba8Ga16Ge30 0.058 vs 0.82 W/mK).
- **anphon** Complete minimum-image search for skewed cells; consistent gauge and harmonic baseline for NONANALYTIC != 0.
- **anphon** Symmetric, ASR-consistent strain derivatives of IFCs; Green-Lagrange strain in the strain-energy expansion.
- **anphon** SCP forces/stress exact derivatives of the interpolated model; symmetric SCPH/QHA FC2 corrections.
- **anphon** ASR check and minimum image detection; acoustic-mode identification at Gamma from eigenvectors.
- **anphon** Many MPI broadcast fixes (TOLERANCE, RELAX_STR, ADAPTIVE_FACTOR, ISMEAR_4PH, LEN_BOUNDARY, TRISYM, element names, QHA IALGO) and rank-dependent IFC order with NEWFCS = 1.
- **anphon** FC3FILE/FC4FILE handling; empty FC3/FC4 arrays; restart problems on some platforms; TMAX dropped from the temperature grid by round-off; temperature index rounding.
- **anphon** Silent wrong results, MPI hangs and input checks found in review; heap overflow broadcasting an empty symmetry list.
- **alm** Bugs in `set_fc_zero_threshold`, `fix_constraint`, equivalent-site detection, supercell symmetry update; NDATA_CV guess and manual CV alphas; ALGO_REDUCTION not passed.
- **alm** Constraint rank reduction; RREF sub-tolerance leftovers crash; Huang constraint looked up FC2 with the wrong atom index; elastic-net CD update; NaN CV scores for pattern DFSETs and cluster-search overrun; stop when every KKT solver fails.
- **alm** Uninitialized `symmetry_tolerance` and the lattice convention of the Python API.
- **tools** POSCAR element order; cell mapping in fcsio/taylor; QE `!` comments; temperature-grid counting in Python readers.

## Build / packaging

- C++17; CMake (>= 3.17, CMake 4 supported) is the only build system, the legacy Makefiles are removed.
- HDF5 and spglib required (anphon uses spglib); FFTW3 no longer needed; Boost >= 1.66, fewer Boost components.
- Optional backends `USE_MKL_BACKEND` (LP64 only), `USE_ACCEL_BACKEND`, `USE_SUITESPARSE_BACKEND`, `USE_EIGEN_BLAS`.
- `make install` / `cmake --install` for binaries and Python scripts; anphon built as a static library.
- Python package built with scikit-build-core + nanobind; header-only Boost fallback.
- conda environment file `etc/alamode-environment.yml` (includes SuiteSparse); PyYAML added to requirements.
- Unit tests off by default.
- CI: GitHub Actions (Si, ZnO, BaTiO3, tools, Python), sanitizer workflow, pre-commit (clang-format, ruff), self-hosted runners, cheap tests per push / expensive nightly.

## Docs

- Migration to GitHub Pages with the Breeze theme, dark mode, logos.
- Japanese translation (HTML and PDF).
- New pages: Features, HDF5 file formats with layout diagrams, input-file builder, using other codes (ASE, WIEN2k, CRYSTAL), force-constant conversion, Python API (alm, anphon), strain tools, "Running ALAMODE" flowcharts.
- Expanded anphon input documentation: adaptive smearing, BTE solvers, velocity-matrix formulation, four-phonon scattering, finite-temperature structural optimization.
- Tutorials: interactive plots, 3D structures and mode animations, Jupyter notebooks on Binder, BAs 4ph, Silicon with Python.
- FAQ rewritten; oneAPI 2025 OpenMP startup delay.
- "Migrating from 1.x to 2.0" page.

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
