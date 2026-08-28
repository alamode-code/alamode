.. _label_strain_tools:

Tools for the cell-relaxation inputs (strainifc.py, elastic.py)
=================================================================

The structural optimization with cell relaxation (``RELAX_STR = 2, 3`` in the
SCPH/QHA modes) needs a few quantities that anphon cannot compute from the
force constants alone. They are read from the directory given by
``STRAIN_IFC_DIR`` (except ``C1_array.in``, which is read from the working
directory of anphon):

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - File
     - Tag
     - Content
   * - ``elastic_constants.in``
     - ``ELASTIC_CONST = 2``
     - Second- and third-order elastic constants :math:`V C^{(2)}`, :math:`V C^{(3)}` (Ry): a label
       (``SOEC``), 81 values :math:`C_{\mu_1\nu_1,\mu_2\nu_2}` in the full-index layout
       (:math:`i = 3\mu + \nu`, row-major), a label (``TOEC``) and 729 values. :math:`V` is the
       volume of the anphon primitive cell (the ``&cell`` field).
   * - ``C1_array.in`` (working directory)
     - ``ELASTIC_CONST = 1, 2``
     - Reference stress :math:`V\sigma_{\mu\nu}` (Ry): a label followed by 9 values (row-major).
       Zero when the file is absent.
   * - ``strain_force.in``
     - ``RENORM_2TO1ST = 2``
     - Forces (eV/Å) on the atoms of the anphon primitive cell in strained cells, one block per
       strain mode: a header ``mode smag weight`` followed by ``natmin`` lines ``fx fy fz``.
   * - ``strain_harmonic.in`` + force-constant files
     - ``RENORM_3TO2ND = 2, 3``
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
  (``elastic_constants.in``, ``C1_array.in``).
* ``strainifc.py`` — strain–force and strain–harmonic-IFC couplings
  (``strain_force.in``, ``strain_harmonic.in``). This is a port of the
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

The tools never reorder atoms: the order of the template structure is used everywhere
(generated inputs, force-constant files, ``strain_force.in`` rows). anphon's primitive-cell
order is the order obtained by folding the supercell of the force-constant file into the
``&cell`` lattice, keeping the first occurrence of every site (for ``.h5`` files without
``&cell``, the stored primitive cell). Give the reference force-constant file (``--fcs``, the
``FC2FILE``/``FCSFILE`` of the anphon run) and the anphon input (``--anphon-cell``) to
``collect``: the supercell template is checked index-wise against the file, every generated
force-constant file is checked for identical translation tables, and the rows of
``strain_force.in`` are written in anphon's order (a permuted template is reported as an error
unless ``--reorder`` is given; for a conventional anphon cell the rows of the translation-equivalent
atoms are duplicated, which requires the DFT setup to have the full translational symmetry — e.g.
no magnetic order enlarging the cell). ``strainifc.py check`` prints the full picture before any
DFT calculation is run.

elastic.py
----------

::

    elastic.py generate --code {VASP,QE} --template DIR [--smag 0.01] [--nmag 2]
                        [--dirset {minimal,full}] [--outdir DIR] [--job-template job.sh]
                        [--dft-command DFT_command.sh] [--force]
    elastic.py fit      [--outdir DIR] [--fit {stress,energy,both}] [--fcs REF] [--anphon-cell FILE]
                        [--no-symmetrize] [--symprec 1e-5] [--compare anphon.log] [--exclude strain_NNN,...]
    elastic.py show     elastic_constants.in (--structure FILE | --volume V_A3) [--c1 C1_array.in]

``generate`` creates the unstrained reference ``strain_000`` and strained cells
:math:`u = k\,s\,d` for :math:`k = \pm 1, \ldots, \pm n_\mathrm{mag}` along a set of directions
:math:`d` in the six-dimensional strain space: ``minimal`` (6 single + 15 pair directions, 85
calculations with the defaults) determines all constants from the stresses; ``full`` (56
directions, 225 calculations) is required when only energies are fitted. ``fit`` builds the
second Piola–Kirchhoff stress :math:`S = \det(F) F^{-1}\sigma F^{-T}` from the DFT (Cauchy)
stress and solves the linear least-squares problem
:math:`S(\eta) = \sigma_0 + C^{(2)}\eta + \frac{1}{2}C^{(3)}\eta\eta` (and/or the energy
expansion) for the 83 independent Voigt components, symmetrizes the tensors over the point
group of the reference structure, prints the constants in GPa and writes the files with
:math:`V` of the anphon cell (``--fcs``/``--anphon-cell``; without them the volume of the DFT
cell is used and a note is printed). The anphon cell and the DFT cell must be commensurate in
the same Cartesian frame: one must be an integer combination of the lattice vectors of the
other (a conventional anphon cell and a primitive DFT cell, or the reverse); rotated settings are
rejected. ``--compare`` prints the difference to the clamped-ion constants that anphon prints
with ``ELASTIC_CONST = 1``.

strainifc.py
------------

::

    strainifc.py generate --coupling {harmonic,force} --code {VASP,QE} --template DIR
                          [--smag 0.005] [--dmag 0.01] [--central] [--no-offset] [--with-reference]
                          [--modes strain_modes.json | --modes xx,yy,...] [--outdir DIR]
                          [--nbody 2] [--cutoff R] [--job-template job.sh] [--dft-command FILE]
    strainifc.py collect  [--outdir DIR] [--fcs REF [--anphon-cell FILE]] [--fcs-format {xml,h5}]
                          [--prefix strain] [--reorder] [--write-dfset] [--unchecked]
    strainifc.py check    [--outdir DIR] --fcs REF [--anphon-cell FILE]

``--fcs`` is mandatory for ``--coupling harmonic`` (unless ``--unchecked``); for
``--coupling force`` it is optional but recommended: without it the rows are
written in the order of the template, which must then be anphon's order.
``--anphon-cell`` is required with an XML ``--fcs`` (anphon needs ``&cell`` for
XML force-constant files).

* ``--coupling force``: the template is the primitive cell. ``strain_000/primitive`` (reference)
  and ``strain_NNN/primitive`` (strained cells) are generated; ``collect`` subtracts the reference
  forces and writes ``strain_force.in`` in anphon's atom order. All six strain modes are required
  (anphon demands that the weights of every component sum to 1); ``--modes`` subsets are only
  meaningful for ``--coupling harmonic`` with ``RENORM_3TO2ND = 3``.
* ``--coupling harmonic``: the template is the **same supercell** as the one used to fit the
  harmonic force constants given to anphon. For every strained supercell the ALM displacement
  patterns are generated (``strain_NNN/disp_MM``), plus the undisplaced strained cell
  (``strain_NNN/nodisp``) whose residual forces are subtracted unless ``--no-offset`` is given.
  ``collect`` fits the harmonic force constants of every strained supercell with the ``alm``
  package (translational invariance imposed), writes them as ``results/strain_NNN.xml`` (or
  ``.h5``) and ``results/strain_harmonic.in``. With ``--with-reference`` the undeformed supercell
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
