ALM: Output files 
-----------------

* ``PREFIX``.pattern_HARMONIC, ``PREFIX``.pattern_ANHARM?

 These files contain displacement patterns in Cartesian coordinate. 
 The length of displacement is normalized to unity for each atom.
 Created when ``MODE = suggest``.
 Patterns for anharmonic force constants are printed only when ``NORDER > 1``.

* ``PREFIX``.fcs

 Harmonic and anharmonic force constants in Rydberg atomic units.
 In the first section, only symmetry-reduced force constants are printed.
 All symmetry-related force constants are shown in the following section
 with the symmetry prefactor (:math:`\pm 1`).
 Created when ``MODE = optimize``.

* ``PREFIX``.xml

 An XML file containing the necessary information for performing
 phonon calculations.
 The file can be read by *anphon* using the ``FCSFILE``-tag.
 Created by default when ``MODE = optimize`` (controlled by ``FCS_ALAMODE``).
 When ``LMODEL = enet | adaptive-lasso``, the file is created only when the cross-validation mode is off (``CV = 0``).

* ``PREFIX``.h5

 An HDF5 file containing the same information as ``PREFIX``.xml in the HDF5 format.
 It can also be read by *anphon* using the ``FCSFILE``-tag.
 Created by default when ``MODE = optimize``.
 When ``LMODEL = enet | adaptive-lasso``, the file is created only when the cross-validation mode is off (``CV = 0``).

* ``PREFIX``.FORCE_CONSTANT_3RD

 Third-order force constants in the FORCE_CONSTANT_3RD format of the ShengBTE code.
 Created when ``MODE = optimize`` and ``FC3_SHENGBTE = 1``.
 When ``LMODEL = enet | adaptive-lasso``, the file is created only when the cross-validation mode is off (``CV = 0``).

* ``PREFIX``.FORCE_CONSTANT_4TH

 Fourth-order force constants in the FORCE_CONSTANT_4TH format of the ShengBTE (FourPhonon) code.
 Created when ``MODE = optimize``, ``FC4_SHENGBTE = 1``, and ``NORDER > 2``.

* ``PREFIX``.fc

 Second-order force constants in the Quantum ESPRESSO ``.fc`` format.
 Created when ``MODE = optimize`` and ``FC2_QEFC = 1``.

* ``PREFIX``.hessian

 The entire Hessian matrix of the supercell.
 Created when ``MODE = optimize`` and ``HESSIAN = 1``.

* ``PREFIX``.cvset 
 
 This file contains training and validation errors of cross-validation performed with the *manually* given ``DFSET`` (training dataset) and ``DFSET_CV`` (validation dataset). Created when the manual cross-validation mode is selected by setting ``CV = -1``.

* ``PREFIX``.cvset[1, ..., ``CV``]

 These files contain training and validation errors of cross-validation performed for ``CV`` subsets. Created when the automatic cross-validation mode is selected by setting ``CV > 1``.

* ``PREFIX``.cvscore

 The mean value and standard deviation of the training and validation errors are reported. Created when the automatic cross-validation (``CV > 1``) is finished.

