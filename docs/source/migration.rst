.. _label_migration:

Migrating from 1.x to 2.0
=========================

This page lists what a user of version 1.5 has to change, and what will look different, when moving to ALAMODE 2.0.
New features are described in the input-variable pages; see also the
`ChangeLog <https://github.com/alamode-code/alamode/blob/master/ChangeLog.md>`_.

In short:

* **Inputs.** *anphon* rejects unknown tags field by field, so a 1.5 input with ``FCSXML``, ``NKD``, or ``ISOTOPE`` in
  ``&analysis`` stops at once. Rename and move the tags as shown in :ref:`label_migration_anphon`.
* **Files.** *alm* writes ``PREFIX``.h5 instead of ``PREFIX``.xml, and *anphon* keeps its results and restart data in
  ``PREFIX``.kappa.h5, ``PREFIX``.scph.h5, and ``PREFIX``.qha.h5.
* **Numbers.** Isotope scattering is on by default, and several fixes change the thermal conductivity. See
  :ref:`label_migration_results`.
* **Build.** HDF5 and spglib are required, FFTW is no longer used, and CMake >= 3.17 and a C++17 compiler are needed.


.. _label_migration_build:

Building
--------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Item
     - 1.5
     - 2.0
   * - CMake
     - >= 3.1
     - >= 3.17
   * - C++ standard
     - C++14
     - C++17
   * - HDF5
     - optional (``WITH_HDF5_SUPPORT``)
     - **required** by *alm* and *anphon*
   * - spglib
     - *alm* only
     - **required** by *alm* and *anphon*
   * - FFTW3
     - required by *anphon* (unless MKL)
     - not used
   * - Boost
     - any version
     - >= 1.66 (headers only)
   * - OpenMP
     - optional
     - required by *alm*
   * - Intel MKL
     - --
     - supported through ``USE_MKL_BACKEND``, LP64 interface only (``-DMKL_INTERFACE=lp64``)
   * - Optional backends
     - --
     - ``USE_MKL_BACKEND``, ``USE_ACCEL_BACKEND``, ``USE_SUITESPARSE_BACKEND``, ``USE_EIGEN_BLAS``
       (see :ref:`install_backends`)
   * - Python interface
     - separate project
     - ``pip install .`` in ``python/`` (Python >= 3.9, nanobind, scikit-build-core, and MPI unless the anphon module is
       switched off), see :ref:`label_migration_python`

``cmake --install`` (or ``make install``) now installs the binaries and the common scripts; see :doc:`install`.

The legacy ``Makefile.*`` files of 1.x (in ``alm/``, ``anphon/`` and ``tools/``) are removed; CMake is the only
supported build.


.. _label_migration_alm:

Changes in *alm*
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Tag
     - 1.5
     - 2.0
   * - ``FCS_ALAMODE``
     - default 1: ``PREFIX``.xml and ``PREFIX``.fcs
     - default 0: only ``PREFIX``.h5. Set ``FCS_ALAMODE = 1`` for the XML and ``.fcs`` files
       (:ref:`FCS_ALAMODE <alm_fcs_alamode>`)
   * - ``FCSYM_BASIS``
     - default ``Lattice``
     - default ``Cartesian`` (:ref:`FCSYM_BASIS <alm_fcsym_basis>`)
   * - ``NAT``, ``NKD``
     - required
     - optional; ``NKD``, when given, must match ``KD``
   * - ``FC2XML``, ``FC3XML``
     - fix FC2 / FC3 to an XML file
     - renamed ``FC2FIX``, ``FC3FIX`` (accept ``.xml`` and ``.h5``); the old names still work with a warning
       (:ref:`FC2FIX <alm_fc2fix>`)
   * - ``ICONST``
     - 0, 1, 2, 3, 11
     - 0--4, 10--14: Huang invariance (4, 14), algebraic rotational invariance (12--14), see :ref:`ICONST <alm_iconst>`
   * - ``FC3_SHENGBTE`` output
     - ``PREFIX``.FORCE_CONSTANT_3RD
     - ``PREFIX``.FORCE_CONSTANTS_3RD (the name ShengBTE reads); ``FC4_SHENGBTE`` writes ``PREFIX``.FORCE_CONSTANTS_4TH
   * - ``MODE = suggest`` output
     - plain-text ``PREFIX``.pattern_*
     - YAML ``PREFIX``.pattern_* (``displace.py`` reads both; it needs PyYAML).
       ``FORMAT_PATTERN = old`` (:ref:`FORMAT_PATTERN <alm_format_pattern>`) writes the 1.5 format

New tags you may want when updating old inputs: ``STRUCTURE_FILE`` (read a POSCAR), ``SUPERCELL`` and ``PRIMCELL``
(build the supercell and the primitive cell inside *alm*, see :ref:`SUPERCELL <alm_supercell>` and
:ref:`PRIMCELL <alm_primcell>`), ``SYMMETRIZE_STRUCTURE``, and the unit tags
``LENGTH_UNIT``, ``FORCE_UNIT``, and ``FCS_UNIT_OUTPUT`` (:ref:`LENGTH_UNIT <alm_length_unit>`).

Notes on the changed defaults:

* **FCSYM_BASIS.** For ``LMODEL = ols`` the force constants do not depend on the basis.
  With ``enet`` or ``adaptive-lasso`` the optimal ``L1_ALPHA`` can change, so redo the cross-validation.
  An XML file of 1.5 given as ``FC2FIX`` / ``FC3FIX`` keeps working: without an explicit ``FCSYM_BASIS``,
  *alm* uses the basis recorded in that file.
  *alm* now warns when the input structure deviates from its space group by more than :math:`10^{-6}` bohr.
  In that case, give the coordinates with full precision or set :ref:`SYMMETRIZE_STRUCTURE = 1 <alm_symmetrize_structure>`.
* **Constraints.** With rotational or Huang constraints (``ICONST = 2, 3, 4``), constraint rows that are nearly
  linearly dependent are now dropped, using a relative test with :ref:`TOL_CONST <alm_tol_const>`
  (default :math:`10^{-6}`, as in 1.5); ``ICONST = 1`` skips this test.
  Such rows occur, for example, for internally relaxed structures combined with a short cutoff, where imposing them
  exactly inflates the fitting error. When no row is dropped, the results are the same as before.
  The algebraic constraints (``ICONST >= 11``) are reduced by a new algorithm (:ref:`ALGO_REDUCTION <alm_algo_reduction>`,
  default 3), which is much faster for large models.
* **Stressed reference structures.** The new ``STRESS`` tag in ``&optimize`` generalizes the Huang conditions of ``ICONST = 4, 14``
  to a reference structure under stress (:ref:`STRESS <alm_stress>`).

Input checks are stricter. The following inputs are now errors:

* a ``DFSET`` line that does not hold exactly six finite numbers;
* ``ICONST`` outside 0--4 and 10--14, an invalid ``ROTAXIS``, ``SKIP = 0``, and ``TOLERANCE <= 0``;
* trailing characters after a number (``1.0abc``), and ``nan`` or ``inf`` in numeric fields;
* a ``&cutoff`` entry that is neither a number nor a token starting with ``N`` (``None``), and unknown element names in ``&cutoff``.
  Extra columns in a ``&cutoff`` line are ignored with a warning;
* negative ``NDATA``, ``NSTART``, ``NEND``, data ranges beyond the ``DFSET`` size, duplicate ``KD`` entries,
  and ``FC2FIX`` / ``FC3FIX`` equal to the output file.


.. _label_migration_anphon:

Changes in *anphon* input
-------------------------

*anphon* stops with ``Could not recognize the variable ...`` when a field contains a tag it does not know.
The table lists how 1.5 inputs are treated.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - 1.5 input
     - 2.0
     - What to do
   * - ``FCSXML``, ``FC2XML`` (``&general``)
     - error
     - Use ``FCSFILE``, ``FC2FILE`` (:ref:`FCSFILE <anphon_fcsfile>`). XML files of 1.5 are still read.
   * - ``NKD`` (``&general``)
     - error
     - Delete it. ``KD`` is optional and defaults to the elements in ``FCSFILE`` (:ref:`KD <anphon_kd>`).
   * - ``NSYM`` (``&general``)
     - error
     - Delete it. In 1.5, ``NSYM = 1`` restricted the symmetry to the identity and ``NSYM > 1`` read the operations
       from ``SYMM_INFO_PRIM``; 2.0 always uses the space group found by spglib, and has no replacement for these options.
   * - ``ISOTOPE``, ``ISOFACT``, ``KAPPA_SPEC``, ``KAPPA_COHERENT`` (``&analysis``)
     - error
     - Move them to the new ``&kappa`` field (``MODE = kappa`` only).
   * - ``FSTATE_K`` (``&analysis``)
     - error
     - Removed.
   * - ``MODE = RTA``
     - warning
     - Use ``MODE = kappa``; the solver is chosen with :ref:`SOLVER <anphon_solver>` (default ``RTA``).
   * - ``RESTART`` (``&general``)
     - warning
     - Move it to ``&kappa`` (:ref:`RESTART <anphon_restart>`).
   * - ``KS_INPUT`` (``&analysis``)
     - accepted, deprecated
     - Prefer ``MODE = selfenergy`` with the ``&selfenergy`` field, which also accepts :math:`q` points off the mesh
       (:ref:`KS_INPUT <anphon_ks_input>`).
   * - ``RENORM_2TO1ST``, ``RENORM_34TO1ST``, ``RENORM_3TO2ND`` (``&relax``)
     - warning
     - Use ``STRAIN_COUPLING`` (see below). The old tags will be removed in a later release.
   * - ``COOLING_U0_INDEX``, ``COOLING_U0_THR`` (``&relax``)
     - ignored, warning
     - ``SET_INIT_STR = 3`` now detects the high-symmetry phase with spglib (:ref:`SET_INIT_STR <anphon_set_init_str>`).
   * - ``&cell`` field
     - required
     - Optional when ``FCSFILE`` is an HDF5 file of *alm* that defines the primitive cell (``PRIMCELL``); see :ref:`anphon_cell_field`.

Default changes:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Tag
     - 1.5
     - 2.0
     - Effect
   * - ``ISOTOPE`` (``&kappa``)
     - 0
     - 1
     - Isotope scattering with natural abundances is included; set ``ISOTOPE = 0`` for the 1.5 behaviour
       (:ref:`ISOTOPE <anphon_isotope>`)
   * - ``TREVSYM`` (``&general``, new)
     - --
     - 1
     - Time-reversal symmetry also reduces the :math:`k` meshes; fewer irreducible points for non-centrosymmetric crystals.
       ``TREVSYM = 0`` gives the 1.5 reduction (:ref:`TREVSYM <anphon_trevsym>`)
   * - ``IMIX`` (``&scph``, new)
     - --
     - 1
     - SCPH iterations use DIIS mixing; ``IMIX = 0`` gives the simple mixing of 1.5 (:ref:`IMIX <anphon_imix>`)
   * - ``FILE_FORMAT`` (``&general``, new)
     - --
     - h5
     - Restart and eigenvector files are HDF5; ``FILE_FORMAT = text`` writes the 1.5 text files (deprecated;
       :ref:`FILE_FORMAT <anphon_file_format>`)

Thermal conductivity, 1.5::

  &general
    PREFIX = si222
    MODE = RTA
    NKD = 1; KD = Si
    FCSXML = si222.xml
  /
  &cell
    ...
  /
  &kpoint
    2
    10 10 10
  /
  &analysis
    ISOTOPE = 2
    KAPPA_SPEC = 1
  /

and 2.0::

  &general
    PREFIX = si222
    MODE = kappa
    FCSFILE = si222.h5     # or si222.xml from 1.5
  /
  &kpoint
    2
    10 10 10
  /
  &kappa
    ISOTOPE = 2
    KAPPA_SPEC = 1
  /

Here ``&cell`` was dropped because ``si222.h5`` holds the primitive cell. Keep it when *alm* was run without
``PRIMCELL`` or when ``FCSFILE`` is an XML file.

*anphon* also checks the input more strictly. A 1.5 input may stop for one of these reasons:

* The ``&cell`` must tile the supercell of ``FCSFILE`` exactly. A lattice constant rounded to a few digits makes the
  transformation matrix non-integer and stops the run: give the lattice constant with full precision, or omit ``&cell``
  with an HDF5 file. A left-handed ``&cell``, a ``&cell`` larger than the supercell, and, with ``NONANALYTIC = 2``, an FC2
  supercell so skewed that some minimum-distance images lie beyond the 26 neighbouring supercells are errors
  (use ``NONANALYTIC = 1`` or ``3``, or a near-orthogonal supercell, in that case).
* ``BORNINFO`` must hold exactly :math:`\varepsilon_\infty` and one :math:`Z^*` tensor per atom.
* Range checks: ``DT > 0``, ``TMIN <= TMAX``, ``TMIN >= 0``, ``DELTA_E > 0``, ``EPSILON > 0`` for ``ISMEAR = 0, 1``,
  ``0 < TOLERANCE < 0.5``, ``MASS > 0`` with one value per species, ``NBANDS`` not larger than the number of modes.
* Malformed numbers, such as ``ISOFACT = abc`` or ``4.5`` for a mesh size in ``&kpoint``, are now rejected (earlier versions read them as 0 and 4).
* ``EMIN`` and ``EMAX`` now set the DOS energy range (1.5 reset ``EMIN`` to 0 and followed the eigenvalue range).


.. _label_migration_finite_t:

SCPH, QHA, and structural optimization
--------------------------------------

**Temperature-dependent harmonic force constants.** In 1.5, the SCPH (or QHA) force constants were used in a later run
by converting ``PREFIX``.scph_dfc2 with the ``dfc2`` program::

  $ dfc2 si222.xml si222_300K.xml si222.scph_dfc2 300

  # in the kappa input
  FCSXML = si222.xml
  FC2XML = si222_300K.xml

In 2.0, *anphon* reads them from the SCPH result file:

.. code-block:: none

  FCSFILE = si222.h5
  DFC2FILE = si222.scph.h5
  FC2_TEMPERATURE = 300

``DFC2FILE`` adds the correction of ``FC2_TEMPERATURE`` to the harmonic force constants of ``FCSFILE`` (or ``FC2FILE``),
which may come from a larger supercell than the SCPH run.
Use the same ``NONANALYTIC`` and ``BORNINFO`` as the SCPH run.
For ``MODE = kappa``, run one temperature per calculation with ``TMIN = TMAX = FC2_TEMPERATURE``.
The results then accumulate in one ``PREFIX``.kappa.h5.
See :ref:`DFC2FILE <anphon_dfc2file>` and :ref:`FC2_TEMPERATURE <anphon_fc2_temperature>`.
``tools/dfc2.py`` (HDF5 version of ``dfc2``) and the ``dfc2`` program are still shipped. Both read the
``PREFIX``.scph_dfc2 files of 1.x and 2.0.

**Relaxed structures.** The 1.5 note that the structural optimization cannot be combined with thermal conductivity
or bubble calculations no longer applies:

* ``BUBBLE = 1, 2, 3`` can be used with ``RELAX_STR > 0``. ``FE_BUBBLE`` still cannot.
* ``RELAXED_STRUCTURE = 1`` with ``FC2_TEMPERATURE`` runs any mode (``kappa``, ``phonons``, ...) on the structure
  relaxed at that temperature. The state file of the relaxation run must be given as ``DFC2FILE`` (or ``FC2FILE``,
  ``FCSFILE``); its prefix is that of the earlier run, not the ``PREFIX`` of the new one
  (:ref:`RELAXED_STRUCTURE <anphon_relaxed_structure>`). For example:

  .. code-block:: none

     PREFIX = bto_kappa_300K
     MODE = kappa
     FCSFILE = bto.h5
     DFC2FILE = bto_relax.scph.h5      # written by the SCPH run with PREFIX = bto_relax and RELAX_STR > 0
     FC2_TEMPERATURE = 300
     RELAXED_STRUCTURE = 1
     TMIN = 300; TMAX = 300

**Cell relaxation inputs.** ``STRAIN_COUPLING`` (``&relax``) replaces the ``RENORM_*`` tags.
Its units digit adds up the inputs read from the strain-coupling files: 1 = elastic constants,
2 = strain--force coupling, 4 = strain--harmonic-IFC coupling (:ref:`STRAIN_COUPLING <anphon_strain_coupling>`).
The default, 7, reads all three, which is what the 1.5 defaults did.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 1.5
     - 2.0
   * - (defaults: ``RENORM_2TO1ST = 2``, ``RENORM_3TO2ND = 2``, ``RENORM_34TO1ST = 0``)
     - ``STRAIN_COUPLING = 7`` (default)
   * - ``RENORM_2TO1ST = 0``
     - clear bit 2, e.g. ``STRAIN_COUPLING = 5``
   * - ``RENORM_2TO1ST = 1``
     - tens digit 1, e.g. ``STRAIN_COUPLING = 15``
   * - ``RENORM_34TO1ST = 1`` (with ``RENORM_2TO1ST = 1``)
     - tens digit 2, e.g. ``STRAIN_COUPLING = 25``
   * - ``RENORM_3TO2ND = 1``
     - clear bit 4, e.g. ``STRAIN_COUPLING = 3``
   * - ``RENORM_3TO2ND = 2`` or ``3``
     - bit 4; a subset of strain patterns is detected from the file

The old tags still work; the warning they print names the equivalent ``STRAIN_COUPLING`` value.
The text files in ``STRAIN_IFC_DIR`` still work, and the HDF5 container ``STRAINFILE`` (made by the strain tools,
:ref:`label_strain_tools`) is the recommended replacement.
Elastic constants in the text files may now be given in GPa.

**Convergence.** Temperatures whose SCPH iteration or structural optimization did not converge are flagged in
``PREFIX``.scph.h5 / ``PREFIX``.qha.h5. Their free energy, heat capacity, and entropy in ``.scph_thermo`` / ``.qha_thermo`` (and
the DOS, MSD, and ucorr rows) are written as NaN unless ``ALLOW_UNCONVERGED = 1``. Later runs refuse to read those
temperatures unless ``ALLOW_UNCONVERGED = 1``.


.. _label_migration_outputs:

Output files
------------

The formats of the HDF5 files are described in :doc:`hdf5_format`.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - 1.5
     - 2.0
     - Note
   * - ``PREFIX``.xml, ``PREFIX``.fcs (*alm*)
     - ``PREFIX``.h5
     - old files with ``FCS_ALAMODE = 1``
   * - ``PREFIX``.result (``MODE = RTA``)
     - ``PREFIX``.kappa.h5
     - results and restart data (:ref:`label_hdf5_kappa`); ``.result`` with ``FILE_FORMAT = text``
   * - ``PREFIX``.scph_dymat
     - ``PREFIX``.scph.h5
     - results and restart data (:ref:`label_hdf5_scph`); ``.scph_dymat`` with ``FILE_FORMAT = text``
   * - ``PREFIX``.renorm_harm_dymat (QHA)
     - ``PREFIX``.qha.h5
     - ``.renorm_harm_dymat`` with ``FILE_FORMAT = text``
   * - ``PREFIX``\ [.band | .mesh].evec and .evec.hdf5 (``PRINTEVEC = 1``)
     - ``PREFIX``\ [.band | .mesh].evec.h5
     - same datasets; text ``.evec`` only with ``FILE_FORMAT = text``
   * - --
     - ``PREFIX``\ [.band | .mesh].eval.h5
     - new tag ``PRINTEVAL``
   * - ``PREFIX``\_+.xml, ``PREFIX``\_-.xml (``NEWFCS = 1``)
     - ``PREFIX``\_+.h5, ``PREFIX``\_-.h5
     - ``.xml`` with ``FILE_FORMAT = text``
   * - --
     - ``PREFIX``.selfenergy.h5
     - ``MODE = selfenergy``

The text outputs (``.bands``, ``.dos``, ``.thermo``, ``.kl``, ``.kl_spec``, ``.self_isotope``, ``.scph_bands``,
``.scph_thermo``, ``.scph_dfc2``, ``.V0``, ...) are still written in both formats.
``.scph_thermo`` and ``.qha_thermo`` now hold the free energies with full precision.
With ``BCONNECT = 1, 2``, the bands are now matched by eigenvectors and frequencies, also across breaks of the path.
The matching also applies to ``.scph_bands`` and ``.qha_bands`` (:ref:`BCONNECT <anphon_bconnect>`).
The log and the HDF5 files record the input variables and the git commit of the binary.

**Restarting from 1.5 files.** A ``PREFIX``.result file is imported into ``PREFIX``.kappa.h5 once and left untouched,
provided that the restart is enabled (``RESTART``, on by default when the file exists), ``FC2_TEMPERATURE`` is not
given, and ``PREFIX``.kappa.h5 holds no linewidths of that channel yet.
Since ``TREVSYM = 1`` changes the irreducible :math:`k` points of non-centrosymmetric crystals,
continue such a run with ``TREVSYM = 0`` (or start with a new ``PREFIX``).
``PREFIX``.scph_dymat files of 1.x are read as they are for ``NONANALYTIC = 0``, converted for ``NONANALYTIC = 2, 3``,
and refused for ``NONANALYTIC = 1`` (see :ref:`label_hdf5_scph_nonanalytic`).
Migrated text restart data are not marked as converged.

A restart starts automatically when ``PREFIX``.kappa.h5, ``PREFIX``.scph.h5, ... (or the 1.5 text files) exist.
Remove them or change ``PREFIX`` to start from scratch.


.. _label_migration_results:

Results that change
-------------------

Rerunning a 1.5 input with 2.0 can give different numbers for these reasons:

* **Isotope scattering is on** (``ISOTOPE = 1``), which lowers :math:`\kappa`, most strongly around its low-temperature peak.
* **Group velocities.** The thermal conductivity uses the velocity matrix of degenerate blocks, which does not depend on
  the choice of eigenvectors. 1.5 took finite differences of sorted eigenvalues, which mixes branches at crossings and
  degeneracies. The change is large in supercells and for degenerate branches: a supercell and the primitive cell sampled at the
  same :math:`q` points now give the same :math:`\kappa`. The velocities are also symmetrized over the little group of :math:`k`, and the translational
  modes at :math:`\Gamma` have zero velocity.
* **Tetrahedron weights** (``ISMEAR = -1``) are averaged over the symmetry-equivalent triplets and the degenerate modes
  for the three-phonon and isotope linewidths, ``SELF_W``, and ``FSTATE_W``.
* **Time-reversal symmetry** (``TREVSYM = 1``) can shift the results of non-centrosymmetric crystals slightly with
  ``ISMEAR = -1``, within the :math:`k`-mesh convergence error.
* **Physical constants** are the CODATA 2018 values instead of older hard-coded constants, which changes the last digits.
* **Non-analytic correction** with ``NONANALYTIC = 1`` (Parlinski): the phase factor of the non-analytic term now uses
  the atomic positions of the primitive cell, the gauge of the force constants, instead of those of the mapped
  supercell atoms. Results of new runs can therefore differ from 1.5 when the two positions differ by a lattice vector.
* **alm** results change when the default ``FCSYM_BASIS`` matters (``enet``, ``adaptive-lasso``) and when nearly dependent
  constraints are dropped (see :ref:`label_migration_alm`).

**Example: Si.** The table compares :math:`\kappa_{xx}` (W/mK, RTA) of the Si tutorial run by the 1.5.0 and 2.0 binaries
with the same force constants: one **alm** 2.0 fit of ``example/Si/reference/DFSET_harmonic`` and ``DFSET_cubic``
(the settings of ``test/test_si.py``), written as ``.xml`` and read by both versions
(2.0 gives the same numbers from the ``.h5`` file of the same fit).
The *anphon* input is the 1.5 tutorial input: ``KD = Si``, ``MASS = 28.0855``, the fcc primitive cell with
:math:`a = 10.203` Bohr, a :math:`10\times 10\times 10` :math:`k` mesh, the tetrahedron method (``ISMEAR = -1``, the default
of both versions), and ``ISOTOPE`` set explicitly (in ``&analysis`` for 1.5, in ``&kappa`` for 2.0).

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14 14 14

   * - Version and ``ISOTOPE``
     - 20 K
     - 50 K
     - 100 K
     - 300 K
     - 1000 K
   * - 1.5.0, ``ISOTOPE = 0`` (1.5 default)
     - 41339
     - 4942.7
     - 817.88
     - 112.25
     - 29.832
   * - 2.0, ``ISOTOPE = 0``
     - 39431
     - 5063.8
     - 821.58
     - 112.35
     - 29.934
   * - 1.5.0, ``ISOTOPE = 1``
     - 2305.6
     - 1504.0
     - 553.52
     - 104.35
     - 29.198
   * - 2.0, ``ISOTOPE = 1`` (2.0 default)
     - 2366.0
     - 1485.4
     - 551.46
     - 104.37
     - 29.285

At 300 K, the default isotope scattering lowers :math:`\kappa` by 7.1 % (112.35 to 104.37 W/mK), and the other changes
(velocities, tetrahedron weights, constants) change it by less than 0.1 % (112.25 to 112.35 W/mK without isotopes).
The remaining differences grow at low temperature, where :math:`\kappa` on this coarse mesh is dominated by few modes
near :math:`\Gamma` and is not converged anyway.


.. _label_migration_tools:

Scripts in ``tools``
--------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 1.5
     - 2.0
   * - ``analyze_phonons.py --calc tau --temp 300 PREFIX.result``
     - ``analyzer.py --calc tau --temp 300 --h5 PREFIX.kappa.h5``; a 1.5 file is read with ``--3ph PREFIX.result``.
       ``analyze_phonons.py`` and its C++ helper are removed
   * - ``makedisp_qe.py``
     - removed; use ``displace.py --QE``, which can now also build the supercell
   * - ``displace.py --random_normalcoord``
     - the standard deviation of each mode is capped so that its typical atomic displacement does not exceed 0.20 of the
       nearest-neighbour distance (individual samples can exceed it), samples are truncated at :math:`4\sigma`, and positive
       frequencies below 10 cm\ :sup:`-1` (except the acoustic modes at :math:`\Gamma`) are raised to 10 cm\ :sup:`-1`.
       The default random seed is fixed (12345). See ``--no_cap_sigma_by_nn``, ``--nn_frac``, ``--truncate_k``,
       ``--omega_min``, ``--random_seed``
   * - ``plotband.py``, ``plotdos.py``
     - unchanged options; ``-o`` saves the figure to a file
   * - --
     - new: ``fcconvert.py`` (convert force constants to and from phonopy, phono3py, and ShengBTE; :ref:`label_fc_conversion`),
       ``fcsinfo.py``, ``calcpes.py``, ``dfc2.py``, and the strain tools ``elastic.py``, ``strainifc.py``, ``strainfile.py``

The Python scripts use the CODATA 2018 constants (``tools/alamode_constants.py``), and ``displace.py`` needs PyYAML
to read YAML pattern files.


.. _label_migration_python:

Python
------

* The Python interface is part of the source tree. The distribution ``alamode-python`` contains two packages, ``alm`` and ``anphon``.
  Build it with ``pip install .`` in the ``python`` directory (:ref:`label_python_api`). It needs Python >= 3.9 and,
  for the ``anphon`` package, an MPI library; to build ``alm`` only, use
  ``pip install . -C cmake.define.ALAMODE_PYTHON_ANPHON=OFF``.
* ``alm`` replaces the standalone ALM Python module of 1.x and keeps its high-level API (``define``, ``set_training_data``,
  ``optimize``, ``get_fc``, ``set_fc``, ``save_fc``, ``optimizer_control``). phonopy's ALM interface works with it unchanged.
  Errors are raised as Python exceptions instead of terminating the process. See :ref:`alm_python`.
* The default ``TOL_CONST`` of the Python API is now :math:`10^{-6}`, as in the command-line program.
* ``anphon`` is new: it runs *anphon* from Python and reads the HDF5 results (:ref:`anphon_python`).

* The 1.x Python module ``alm`` (for example the conda-forge package ``alm``) provides the same top-level module name.
  Uninstall it before installing 2.0 (``conda remove alm`` or ``pip uninstall alm``), so that ``import alm`` picks up
  the 2.0 package.

.. TODO(owner): update this once the conda-forge package for 2.0 exists and has been checked (package name, whether
   it replaces the old "alm" package).
