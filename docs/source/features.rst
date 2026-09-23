Features
========

.. |umulaut_u|    unicode:: U+00FC

General
-------

* Extraction of harmonic and anharmonic force constants based on the supercell approach
* Applicable to arbitrary crystal structures and low-dimensional systems
* Accurate treatment of translational and rotational invariance
* Interfaces to the VASP, Quantum ESPRESSO, OpenMX, xTAPP, and LAMMPS codes
* Mainly written in C++, parallelized with MPI+OpenMP

.. _label_features_fcs:

Properties and the force constants they need
--------------------------------------------

The tables below list what *anphon* can calculate and which force constants must be available.
FC2, FC3, and FC4 are the harmonic, cubic, and quartic force constants.
In *alm*, they correspond to ``NORDER = 1``, ``2``, and ``3``, respectively.
✓: required; (✓): required only for the option given in the note.

Harmonic properties
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 44 36 7 7 6

   * - Property
     - Setting
     - FC2
     - FC3
     - FC4
   * - Phonon dispersion
     - ``MODE = phonons``, ``KPMODE = 1``
     - ✓
     -
     -
   * - Phonon DOS, atom-projected DOS
     - ``MODE = phonons``, ``KPMODE = 2``; ``PDOS``
     - ✓
     -
     -
   * - Heat capacity, entropy, and free energy
     - ``MODE = phonons``, ``KPMODE = 2``
     - ✓
     -
     -
   * - Mean-square displacements, displacement correlation
     - ``PRINTMSD``, ``UCORR``
     - ✓
     -
     -
   * - Group velocities
     - ``PRINTVEL``
     - ✓
     -
     -
   * - Two-phonon DOS, three-phonon scattering phase space
     - ``TDOS``, ``SPS``
     - ✓
     -
     -
   * - Participation ratio (localization of phonon modes)
     - ``PRINTPR``
     - ✓
     -
     -
   * - Visualization and animation of phonon modes
     - ``PRINTXSF``, ``ANIME``
     - ✓
     -
     -
   * - Irreducible representations and IR/Raman activity at :math:`\Gamma`
     - ``IRREPS``
     - ✓
     -
     -
   * - Dielectric function, mode effective charges
     - ``DIELEC``, ``ZMODE`` (with ``BORNINFO``)
     - ✓
     -
     -

Cubic anharmonic properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 44 36 7 7 6

   * - Property
     - Setting
     - FC2
     - FC3
     - FC4
   * - Gr\ |umulaut_u|\ neisen parameters
     - ``MODE = phonons``, ``GRUNEISEN``
     - ✓
     - ✓
     -
   * - Lattice thermal conductivity (RTA or full solution of the BTE)
     - ``MODE = kappa``, ``SOLVER``
     - ✓
     - ✓
     -
   * - Coherent (off-diagonal) thermal conductivity
     - ``KAPPA_COHERENT``
     - ✓
     - ✓
     -
   * - Thermal conductivity spectrum, cumulative thermal conductivity
     - ``KAPPA_SPEC``, ``analyzer.py``
     - ✓
     - ✓
     -
   * - Isotope and boundary scattering
     - ``ISOTOPE``, ``LEN_BOUNDARY``
     - ✓
     - ✓
     -
   * - Phonon linewidth, frequency shift, spectral function
     - ``MODE = selfenergy``
     - ✓
     - ✓
     - (✓)
   * - Force constants of strained cells
     - ``NEWFCS``
     - ✓
     - ✓
     - (✓)

.. note::
   FC4 is needed for the frequency shift due to the quartic term (``QUARTIC = 1``),
   and for the strain dependence of the cubic force constants (``NEWFCS`` with ``QUARTIC = 1``).

Quartic anharmonic properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 44 36 7 7 6

   * - Property
     - Setting
     - FC2
     - FC3
     - FC4
   * - Four-phonon scattering in the thermal conductivity
     - ``MODE = kappa``, ``INCLUDE_4PH = 1``
     - ✓
     - ✓
     - ✓
   * - Temperature-dependent phonons by the self-consistent phonon (SCPH) method
     - ``MODE = SCPH``
     - ✓
     - ✓
     - ✓
   * - Anharmonic free energy
     - ``MODE = SCPH``
     - ✓
     - ✓
     - ✓
   * - Crystal structure and thermal expansion at finite temperatures by SCPH
     - ``MODE = SCPH``, ``RELAX_STR``
     - ✓
     - ✓
     - ✓
   * - Crystal structure and thermal expansion within the quasi-harmonic approximation
     - ``MODE = QHA``
     - ✓
     - ✓
     - ✓
   * - Thermal conductivity with SCPH or QHA phonons and structures
     - ``FC2_TEMPERATURE`` (and ``RELAXED_STRUCTURE``)
     - ✓
     - ✓
     - ✓

.. note::
   Relaxing the cell (``RELAX_STR = 2``) also needs DFT calculations of strained cells (``STRAINFILE``),
   unless symmetry allows them to be skipped (``STRAIN_COUPLING = 0``); see :doc:`quickstart`.

The input variables are described in :ref:`label_inputvar_anphon`, and step-by-step examples are given in the :doc:`tutorial`.
