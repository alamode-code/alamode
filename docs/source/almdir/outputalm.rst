ALM: Output files
-----------------

Displacement patterns (``MODE = suggest``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``PREFIX``.pattern_HARMONIC, ``PREFIX``.pattern_ANHARM3, ``PREFIX``.pattern_ANHARM4, ...

 Displacement patterns in Cartesian coordinates, one file per order, in the YAML format.
 The length of the displacement is normalized to unity for each atom.
 The anharmonic patterns are written only when ``NORDER > 1``.
 These files are read by ``displace.py --pattern_file``.

Force constants (``MODE = optimize``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``PREFIX``.h5

 The force constants and the crystal structure in the HDF5 format.
 This is the standard output: it is always written and is read by *anphon* through ``FCSFILE``.
 The layout of the file is described in :ref:`label_hdf5_fcs`.

* ``PREFIX``.xml, ``PREFIX``.fcs

 Legacy outputs, written only when ``FCS_ALAMODE = 1``.
 ``PREFIX``.xml holds the same information as ``PREFIX``.h5 in the XML format, and *anphon* can still read it.
 ``PREFIX``.fcs is a human-readable list of the force constants in Rydberg atomic units:
 the symmetry-reduced force constants first, then all symmetry-related ones with the symmetry prefactor (:math:`\pm 1`).

* ``PREFIX``.FORCE_CONSTANT_3RD, ``PREFIX``.FORCE_CONSTANT_4TH

 Third- and fourth-order force constants in the format of the ShengBTE and FourPhonon codes.
 Written when ``FC3_SHENGBTE = 1`` and ``FC4_SHENGBTE = 1`` (the latter needs ``NORDER > 2``).

* ``PREFIX``.hessian

 The entire Hessian matrix of the supercell. Written when ``HESSIAN = 1``.

.. note::
   With ``LMODEL = enet`` or ``adaptive-lasso``, the force-constant files are written only when
   cross-validation is off (``CV = 0``). A cross-validation run only selects the regularization parameter.

Cross-validation (``MODE = optimize``, ``LMODEL = enet`` or ``adaptive-lasso``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``PREFIX``.cvset

 Training and validation errors of the cross-validation with the given ``DFSET`` (training) and ``DFSET_CV`` (validation) data.
 Written when ``CV = -1``.

* ``PREFIX``.cvset1, ..., ``PREFIX``.cvset\ *N*

 Training and validation errors for each of the *N* = ``CV`` subsets. Written when ``CV > 1``.

* ``PREFIX``.cvscore

 Mean and standard deviation of the training and validation errors over the subsets. Written when ``CV > 1``.
