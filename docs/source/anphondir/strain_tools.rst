.. _label_strain_tools:

Tools for the cell-relaxation inputs (strainifc.py, elastic.py, strainfile.py)
================================================================================

The structural optimization with cell relaxation (``RELAX_STR = 2, 3`` in the
SCPH/QHA modes) needs a few quantities that anphon cannot compute from the
force constants alone: the reference stress and the second- and third-order
elastic constants, the strain–force coupling, and the strain–harmonic-IFC
coupling. They are prepared from DFT calculations of strained cells by the
Python scripts described below, and handed to anphon in one of two ways:

* **the strain-coupling container** (recommended): one HDF5 file given as
  ``STRAINFILE`` in the ``&relax`` field, see :ref:`label_strain_container`;
* **the legacy text files** in the directory given by ``STRAIN_IFC_DIR``
  (plus ``C1_array.in`` in the working directory of anphon), see
  :ref:`label_strain_legacy_files`. anphon prints a note when this route is
  used; it will be removed in a later release.

.. _label_strain_container:

One file for all inputs: the strain-coupling container (``STRAINFILE``)
-----------------------------------------------------------------------

The container is an HDF5 file with the schema ``alamode:strain_coupling``.
Every ingredient lives in its own group, the units are stored as attributes,
and the reference structure is stored once, so that anphon can verify that
the pieces belong together and to the cell of the run:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Group
     - Needed by
     - Content
   * - ``/ReferenceCell``
     - always
     - The reference structure (lattice, fractional coordinates, elements) in
       anphon's primitive-cell atom order. anphon checks that its own
       primitive cell (the ``&cell`` field) describes the same crystal; a nested
       super- or sub-cell is accepted.
   * - ``/Elastic``
     - ``STRAIN_COUPLING`` bit 1 (``soec``, ``toec``); the reference stress is
       used with bit 1 clear as well
     - ``stress`` (3×3), ``soec`` (9×9) and ``toec`` (9×9×9) in GPa, the layout
       of the text files (:math:`i = 3\mu + \nu`). The stress may be absent
       (zero is used, with a note); ``soec`` and ``toec`` come together.
   * - ``/StrainForce``
     - ``STRAIN_COUPLING`` bit 2
     - The strain-mode table (``modes``, ``smag``, ``weight``), the forces
       ``[n_modes, n_atoms, 3]`` in eV/Å, and ``Cell``, the cell the rows
       belong to (the role of the ``&reference_cell`` header of the text file).
   * - ``/StrainHarmonic``
     - ``STRAIN_COUPLING`` bit 4
     - The strain-mode table and one sub-group ``entry_NNN`` per strained
       supercell holding its harmonic force constants in the layout of the
       alm ``.h5`` files (``SuperCell``, ``ForceConstants/Order2``). anphon
       checks every entry against the supercell of ``FC2FILE``/``FCSFILE``
       deformed by the entry's own strain, atom by atom.

The producing commands write into the container directly; the same file can
be updated by each, replacing only its own groups and refusing a reference
structure that is not the crystal already stored::

    elastic.py fit ... --fcs FC2FILE --anphon-cell anphon.in --strain-file ZnO.strain.h5
    strainifc.py collect --coupling harmonic ... --fcs FC2FILE --anphon-cell anphon.in --fcs-format h5 --strain-file ZnO.strain.h5

``elastic.py fit`` writes ``/Elastic`` **and** ``/StrainForce``: its single-mode runs at
:math:`\pm s` are exactly the strained primitive cells of the strain–force coupling, so no
separate ``strainifc.py --coupling force`` calculations are needed. That route remains for
runs that compute the elastic constants from the IFCs (``STRAIN_COUPLING = 6``, no ``elastic.py``)::

    strainifc.py collect --coupling force    ... --fcs FC2FILE --anphon-cell anphon.in --strain-file ZnO.strain.h5

and the anphon input needs a single line::

    &relax
      ...
      STRAINFILE = ZnO.strain.h5
    /

``strainfile.py`` inspects, checks, and converts containers::

    strainfile.py show  ZnO.strain.h5
    strainfile.py check ZnO.strain.h5 --anphon-cell anphon.in --fcs FC2FILE
    strainfile.py pack  --strain-ifc-dir strain_IFC [--c1 C1_array.in] --fcs FC2FILE --anphon-cell anphon.in
                        [--legacy-cell anphon.in] -o ZnO.strain.h5

``show`` prints the contents (cells, units, the elastic constants in GPa, the
strain modes and their weight sums) and which anphon settings the file
supports; ``check`` repeats anphon's consistency checks against a planned run
before it is submitted; ``pack`` converts an existing ``STRAIN_IFC_DIR`` (any
subset of the text files; legacy ``Ry`` files are converted to GPa with the
volume of ``--legacy-cell``, the ``&cell`` of the run they were made for, and
that assumption is recorded in the file). Every write records its command
line in the ``provenance`` attribute. The container is read on every MPI
rank and must not be modified while anphon runs. Non-magnetic reference
structures only.

.. _label_strain_legacy_files:

Legacy text layout (``STRAIN_IFC_DIR``)
---------------------------------------

The text files are read from the directory given by ``STRAIN_IFC_DIR``
(except ``C1_array.in``, which is read from the working directory of anphon):

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - File
     - Tag
     - Content
   * - ``elastic_constants.in``
     - ``STRAIN_COUPLING`` bit 1
     - Second- and third-order elastic constants in GPa: a label with the unit token
       (``SOEC GPa``), 81 values :math:`C_{\mu_1\nu_1,\mu_2\nu_2}` in the full-index layout
       (:math:`i = 3\mu + \nu`, row-major), a label (``TOEC GPa``) and 729 values. anphon
       multiplies them by the volume of its primitive cell, so the file does not depend on the
       cell size (any nested ``&cell`` of the same reference crystal, in the same Cartesian
       frame and strain convention). Files without the unit token (the legacy layout) or with
       an explicit ``Ry`` token hold :math:`V C^{(2)}`, :math:`V C^{(3)}` in Ry for one
       specific cell; anphon cannot check that cell and warns when such a file is used
       together with a user-defined ``&cell``.
   * - ``C1_array.in`` (working directory)
     - always
     - Reference stress :math:`\sigma_{\mu\nu}` in GPa: ``C1 GPa`` followed by 9 values
       (row-major); files without the unit token (or with ``Ry``) hold :math:`V\sigma` in Ry
       for one specific cell. Zero when the file is absent.
   * - ``strain_force.in``
     - ``STRAIN_COUPLING`` bit 2
     - Forces (eV/Å) in strained cells, one block per strain mode: a header ``mode smag weight``
       followed by one line ``fx fy fz`` per atom. An optional ``&reference_cell ... /`` header
       (see below) records the cell the rows belong to; without it the rows must follow the
       atom order of the anphon primitive cell.
   * - ``strain_harmonic.in`` + force-constant files
     - ``STRAIN_COUPLING`` bit 4
     - One line ``mode smag weight filename`` per strained supercell; ``filename`` (relative to
       ``STRAIN_IFC_DIR``, ``.xml`` or ``.h5``) holds the harmonic force constants of the strained
       supercell in Ry/bohr\ :sup:`2`.

The elastic constants are the Brugger constants, i.e. derivatives of the static energy with
respect to the Green–Lagrange strain :math:`\eta = \mathrm{sym}(u) + \frac{1}{2} u u^{T}` of the
deformation gradient :math:`F = I + u` (:math:`u` symmetric), and they must be the **clamped-ion**
constants because anphon relaxes the internal coordinates explicitly.
Strain modes are named ``xx, yy, zz, yz, zx, xy``; a mode ``yz`` with magnitude ``smag``
means :math:`u_{yz} = u_{zy} = \mathrm{smag}/2`. Weights of a mode must sum to 1
(one-sided differences: one line with weight 1; central differences: ``+smag`` and ``-smag`` with
weight 0.5 each).

Two Python scripts in the ``tools/`` directory prepare these files from DFT calculations
(they need ``numpy``, ``ase`` and ``spglib``; ``strainifc.py --coupling harmonic`` additionally
needs the ``alm`` Python package built from the ``python/`` directory):

* ``elastic.py`` — finite-strain workflow for :math:`\sigma`, :math:`C^{(2)}` and :math:`C^{(3)}`
  (``elastic_constants.in``, ``C1_array.in``); the same runs also give the strain–force
  coupling (``strain_force.in``).
* ``strainifc.py`` — strain–harmonic-IFC coupling (``strain_harmonic.in``) and, for runs
  without ``elastic.py``, the strain–force coupling (``strain_force.in``). This is a port of the
  `strainIFCcoupling <https://github.com/r-masuki/strainIFCcoupling>`_ scripts by Ryota Masuki
  [Masuki2022]_ [Masuki2023]_ onto the in-tree ``alm`` package.

Both follow the same pattern: ``generate`` writes the strained input structures (VASP ``POSCAR``
or Quantum-ESPRESSO ``pw.in``, taken from a template directory whose other files are copied),
the user runs the DFT code in every directory (single-point calculations: fixed cell **and**
fixed ions), and ``fit`` / ``collect`` read the outputs (``vasprun.xml`` / ``pw.out``) and write
the anphon files into ``results/``. A JSON manifest written by ``generate`` carries all
parameters, so nothing has to be re-typed. Every DFT output is checked against the generated
structure (cell, species and fractional coordinates at the same index); relaxed geometries are
rejected.

Atom ordering
-------------

The generated inputs and the force-constant files keep the order of the template structure.
For ``--coupling harmonic`` the template must be the supercell of the force-constant file
given to anphon: ``collect`` checks it index-wise against ``--fcs`` and checks every generated
force-constant file for identical translation tables. The rows of ``strain_force.in``, on the
other hand, refer to the atoms of anphon's primitive cell, i.e. the order obtained by folding
the supercell of the force-constant file into the ``&cell`` lattice, keeping the first
occurrence of every site (for ``.h5`` files without ``&cell``, the stored primitive cell).
Give the reference force-constant file (``--fcs``, the ``FC2FILE``/``FCSFILE`` of the anphon
run) and the anphon input (``--anphon-cell``) to ``collect``: the atoms of the DFT cell are
matched by position to that cell, and the rows are written in anphon's order and preceded by
an ``&reference_cell`` header that records that cell (a permuted template is reported as an
error unless ``--reorder`` is given; for a conventional anphon cell the rows of the
translation-equivalent atoms are duplicated, which requires the DFT setup to have the full
translational symmetry — e.g. no magnetic order enlarging the cell; for a DFT supercell of the
anphon cell one translation image per atom is used). ``strainifc.py check`` prints the full
picture before any DFT calculation is run.

The ``&reference_cell`` header of strain_force.in
-------------------------------------------------

``strain_force.in`` holds one force row per atom, so by itself it is tied to the atom list of
the cell the strained calculations were done for. ``strainifc.py collect --fcs ...`` therefore
records that cell at the top of the file::

    &reference_cell
      1.889726124565062
       3.235859326375770   0.000000000000000   0.000000000000000
      -1.617929663187880   2.802336379714220   0.000000000000000
       0.000000000000000   0.000000000000000   5.224712025937350
      4
      Zn    0.333333333333333   0.666666666666667   0.000000000000000
      Zn    0.666666666666667   0.333333333333333   0.500000000000000
      O     0.333333333333333   0.666666666666667   0.381500000000000
      O     0.666666666666667   0.333333333333333   0.881500000000000
    /
    xx 0.005 1.0
    ...

i.e. a scale factor and three lattice vectors (one per line) as in the ``&cell`` field (here
Å converted to bohr), the number of atoms, one line ``symbol x y z`` per atom in fractional
coordinates, and a closing ``/``. No comments are allowed anywhere in the file: anphon reads it
as a plain stream of tokens.

With the header, anphon compares this cell with its own primitive cell (the ``&cell`` field)
and matches the atoms by position, so the row order in the file no longer matters. If the
anphon cell is an integer supercell of the reference cell — for instance an enlarged ``&cell``
chosen to condense a zone-boundary instability — every row is copied onto the translation
images of its atom; if it is a sub-cell, the images are averaged (a warning reports their
spread when it exceeds 1e-8 eV/Å). The two cells must describe the same crystal in the same
Cartesian frame, and one must be an integer supercell of the other: rotated settings and
commensurate but non-nested cells, inconsistent atom counts and atoms without a counterpart
are rejected with an explicit message. anphon logs the atom counts and the volume ratio of
the mapping it applied. Without the header the rows must follow the atom order of the anphon
primitive cell, as before.

``elastic_constants.in`` and ``C1_array.in`` need no such header: written in GPa they do not
depend on the cell size (the reference structure and the Cartesian frame must of course be
the same), and anphon multiplies them by the volume of its own primitive cell.

elastic.py
----------

::

    elastic.py generate --code {VASP,QE} --template DIR [--smag 0.01] [--nmag 2]
                        [--dirset {minimal,full}] [--outdir DIR] [--job-template job.sh]
                        [--dft-command DFT_command.sh] [--force]
    elastic.py fit      [--outdir DIR] [--fit {stress,energy,both}] [--fcs REF] [--anphon-cell FILE]
                        [--no-symmetrize] [--symprec 1e-5] [--compare anphon.log] [--exclude strain_NNN,...]
                        [--strain-file FILE.h5 [--force]]
    elastic.py show     elastic_constants.in [--structure FILE | --volume V_A3] [--c1 C1_array.in]

``generate`` creates the unstrained reference ``strain_000`` and strained cells
:math:`u = k\,s\,d` for :math:`k = \pm 1, \ldots, \pm n_\mathrm{mag}` along a set of directions
:math:`d` in the six-dimensional strain space: ``minimal`` (6 single + 15 pair directions, 85
calculations with the defaults) determines all constants from the stresses; ``full`` (56
directions, 225 calculations) is required when only energies are fitted. ``fit`` builds the
second Piola–Kirchhoff stress :math:`S = \det(F) F^{-1}\sigma F^{-T}` from the DFT (Cauchy)
stress and solves the linear least-squares problem
:math:`S(\eta) = \sigma_0 + C^{(2)}\eta + \frac{1}{2}C^{(3)}\eta\eta` (and/or the energy
expansion) for the 83 independent Voigt components, symmetrizes the tensors over the point
group of the reference structure, prints the constants in GPa and writes the files in GPa
(unit token ``GPa`` after each label), which makes them independent of the size of the cell
anphon uses (they still refer to the DFT reference structure and its Cartesian frame).
``--fcs``/``--anphon-cell`` are optional and only report how the anphon cell relates to the DFT
cell; when given, the two cells must be commensurate in the same Cartesian frame: one must be
an integer combination of the lattice vectors of the other (a conventional anphon cell and a
primitive DFT cell, or the reverse); rotated settings are rejected. ``--compare`` prints the
difference to the clamped-ion constants that anphon prints when it computes them from the IFCs
(``STRAIN_COUPLING`` bit 1 clear).
The forces of the reference and of the single-mode runs at :math:`k = \pm 1` are the
central-difference strain–force coupling (weights 1/2), written as ``strain_force.in`` next to
the elastic files (rows in anphon's order with the ``&reference_cell`` header when ``--fcs`` is
given, as ``strainifc.py collect --coupling force`` does). ``--strain-file`` additionally writes
the reference stress and the constants (``/Elastic``) and the strain–force coupling
(``/StrainForce``) into the strain-coupling container; with ``--fcs``/``--anphon-cell`` the
container is labeled with the anphon primitive cell after the DFT reference structure has been
verified to be that crystal. ``show`` prints any ``elastic_constants.in`` in GPa; ``--structure``/``--volume``
are needed only for legacy files holding :math:`V C` in Ry.

strainifc.py
------------

::

    strainifc.py generate --coupling {harmonic,force} --code {VASP,QE} --template DIR
                          [--smag 0.005] [--dmag 0.01] [--central] [--no-offset] [--with-reference]
                          [--modes strain_modes.json | --modes xx,yy,...] [--outdir DIR]
                          [--nbody 2] [--cutoff R] [--job-template job.sh] [--dft-command FILE]
    strainifc.py collect  [--outdir DIR] [--fcs REF [--anphon-cell FILE]] [--fcs-format {xml,h5}]
                          [--prefix strain] [--reorder] [--write-dfset] [--unchecked]
                          [--strain-file FILE.h5 [--force]]
    strainifc.py check    [--outdir DIR] --fcs REF [--anphon-cell FILE]

``--fcs`` is mandatory for ``--coupling harmonic`` (unless ``--unchecked``); for
``--coupling force`` it is optional but recommended: without it the rows are
written in the order of the template, which must then be anphon's order.
``--anphon-cell`` is required with an XML ``--fcs`` (anphon needs ``&cell`` for
XML force-constant files).

* ``--coupling force`` (only needed when ``elastic.py fit`` is not run, e.g. with
  ``STRAIN_COUPLING = 6``): the template is the primitive cell. ``strain_000/primitive`` (reference)
  and ``strain_NNN/primitive`` (strained cells) are generated; ``collect`` subtracts the reference
  forces and writes ``strain_force.in`` in anphon's atom order, with the ``&reference_cell``
  header when ``--fcs`` is given (and ``/StrainForce`` of the container with ``--strain-file``). All six strain modes are required
  (anphon demands that the weights of every component sum to 1); ``--modes`` subsets are only
  meaningful for ``--coupling harmonic`` (anphon completes the missing components by symmetry).
  Use ``--central``: the one-sided difference can bias the thermal expansion noticeably
  (by about 10 percent for the c axis of ZnO at 1000 K).
* ``--coupling harmonic``: the template is the **same supercell** as the one used to fit the
  harmonic force constants given to anphon. For every strained supercell the ALM displacement
  patterns are generated (``strain_NNN/disp_MM``), plus the undisplaced strained cell
  (``strain_NNN/nodisp``) whose residual forces are subtracted unless ``--no-offset`` is given.
  ``collect`` fits the harmonic force constants of every strained supercell with the ``alm``
  package (translational invariance imposed), writes them as ``results/strain_NNN.xml`` (or
  ``.h5``) and ``results/strain_harmonic.in``; with ``--strain-file`` (which requires
  ``--fcs-format h5``) the fitted force constants are embedded in ``/StrainHarmonic`` of the container. With ``--with-reference`` the undeformed supercell
  is generated as well (``strain_000``); ``collect`` fits it to ``results/strain_000.*`` (not
  listed in ``strain_harmonic.in``) and prints its difference to ``--fcs`` — a direct check that
  the DFT setup reproduces the harmonic force constants given to anphon.

A job-script template containing the line ``RUN_DFT_CALCULATION`` (``--job-template``) and a
file with the shell lines that run the DFT code in one directory (``--dft-command``) produce
``job.sh`` files in the same way as the original strainIFCcoupling scripts; without them a plain
``run_all.sh`` loop is written. Template inputs for wurtzite ZnO are provided in
``example/ZnO/strain_IFC_workflow``.

.. [Masuki2022] R. Masuki, T. Nomoto, R. Arita, and T. Tadano, Phys. Rev. B **106**, 224104 (2022).
.. [Masuki2023] R. Masuki, T. Nomoto, R. Arita, and T. Tadano, Phys. Rev. B **107**, 134119 (2023).
