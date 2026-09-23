Running ALAMODE
===============

.. |Angstrom|   unicode:: U+00C5

ALAMODE consists of two programs.
**alm** extracts the interatomic force constants (IFCs) from forces calculated by DFT,
and **anphon** uses these force constants to calculate phonon properties.
The chart below shows how the two programs, the DFT code, and the helper scripts in the ``tools/`` directory work together.

.. figure:: ../img/flowcharts/overview.*
   :align: center
   :alt: Overall workflow: DFT, alm, anphon and the analysis tools

   Overall workflow. Blue: *alm*; green: *anphon*; grey: DFT calculations; yellow: helper scripts in ``tools/``.


Step 1: Force constants with alm
--------------------------------

.. figure:: ../img/flowcharts/alm_workflow.*
   :align: center
   :alt: Flowchart for calculating force constants with alm

   How to obtain the force constants. The harmonic force constants are calculated first; they are needed to generate the random displacements for the anharmonic force constants.

1. **Prepare the DFT reference.**
   Converge the cutoff energy and the :math:`k`-point density, then relax the primitive cell until the forces are negligible.
   Phonons are sensitive to the lattice constant, especially in polar materials such as perovskites.

2. **Choose a supercell.**
   A conventional cell is usually a reasonable start.
   If the primitive cell is already large (:math:`a \sim 10` |Angstrom|), it can be used as it is.

3. **Harmonic force constants (FC2).**
   Run *alm* with ``MODE = suggest`` and ``NORDER = 1`` to obtain the displacement patterns (``PREFIX``.pattern_HARMONIC),
   and create the displaced structures with ``displace.py``::

     $ alm alm.in > alm.log
     $ displace.py --QE supercell.pw.in --mag 0.01 --prefix disp --pattern_file PREFIX.pattern_HARMONIC

   Calculate the forces of all structures with DFT, and collect the displacements and forces into one file (``DFSET``)::

     $ extract.py --QE supercell.pw.in disp*.pw.out > DFSET

   Then set ``MODE = optimize``, give the ``DFSET`` file in the ``&optimize`` field, and run *alm* again with ``LMODEL = ols``.
   The harmonic force constants are written to ``PREFIX``.h5.

4. **Anharmonic force constants (FC3, FC4, ...)** — needed for ``kappa``, ``SCPH``, ``QHA``, and more.
   The recommended way is to fit them to random displacements:

   * Calculate the harmonic phonons of the supercell with *anphon* (``MODE = phonons``, ``KPMODE = 0``) using the harmonic force constants.
     This writes the phonon eigenvectors to ``PREFIX``.evec.
   * ``displace.py --random_normalcoord --evec PREFIX.evec --temp T`` generates structures sampled at the temperature ``T``.
     Alternatively, ``--random`` adds random displacements to snapshots of a molecular-dynamics run.
   * Calculate the forces with DFT and make a ``DFSET`` file as in step 3.
   * Run *alm* with ``MODE = optimize``, ``NORDER = 2`` or larger, and ``LMODEL = enet`` or ``adaptive-lasso``.
     Give the harmonic force constants with ``FC2FIX`` so that only the anharmonic terms are fitted.

   The result, ``PREFIX``.h5, contains everything *anphon* needs.

``displace.py`` and ``extract.py`` support VASP, Quantum ESPRESSO, OpenMX, xTAPP, and LAMMPS.
The ``DFSET`` format is described :ref:`here <label_format_DFSET>`, and a complete example is given in :doc:`tutorial_pages/silicon_ifc`.

The input variables of *alm* are listed :ref:`here <label_inputvar_alm>`.


Step 2: Phonon properties with anphon
-------------------------------------

.. figure:: ../img/flowcharts/anphon_modes.*
   :align: center
   :alt: Flowchart for choosing the MODE of anphon

   Choosing ``MODE``. Each box lists the force constants (FC2: harmonic, FC3: cubic, FC4: quartic) that ``PREFIX``.h5 must contain.

Set ``MODE`` and ``FCSFILE = PREFIX.h5`` in the ``&general`` field of the *anphon* input, and run::

  $ anphon anphon.in > anphon.log
  $ mpirun -np 8 anphon anphon.in > anphon.log     # with MPI

OpenMP threads are set by ``OMP_NUM_THREADS`` and can be combined with MPI.
The main output files are:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``MODE``
     - Main output files
   * - phonons
     - ``PREFIX``.bands (dispersion), .dos, .thermo (thermodynamic functions), .msd (mean-square displacements)
   * - kappa
     - ``PREFIX``.kappa.h5 (all results), .kl (thermal conductivity)
   * - selfenergy
     - ``PREFIX``.selfenergy.h5
   * - SCPH
     - ``PREFIX``.scph.h5, .scph_bands, .scph_dos, .scph_thermo
   * - QHA
     - ``PREFIX``.qha.h5, .qha_thermo

The input variables of *anphon* are listed :ref:`here <label_inputvar_anphon>`, and all output files :ref:`here <reference_output>`.


Finite-temperature calculations (SCPH and QHA)
----------------------------------------------

.. figure:: ../img/flowcharts/finite_temperature.*
   :align: center
   :alt: Flowchart for the structural optimization at finite temperatures

   Options of the finite-temperature calculations, and how to use their results in later calculations.

* ``RELAX_STR`` selects what is relaxed at each temperature: nothing, the atomic positions, or the atomic positions and the cell.
* Relaxing the cell also needs the elastic constants and the coupling between strain and forces.
  They come from DFT calculations of strained cells, prepared with the :ref:`strain tools <label_strain_tools>` and given as ``STRAINFILE``.
  For high-symmetry crystals, ``STRAIN_COUPLING = 0`` estimates them from the force constants instead.
* The results at every temperature are stored in ``PREFIX``.scph.h5 or ``PREFIX``.qha.h5.
  To use them in a later ``MODE = kappa`` or ``phonons`` calculation, set ``FC2_TEMPERATURE``,
  and also ``RELAXED_STRUCTURE = 1`` if the structure was relaxed.


Step 3: Analyze the results
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Script
     - Reads
     - Shows
   * - ``plotband.py``
     - ``PREFIX``.bands
     - Phonon dispersion
   * - ``plotdos.py``
     - ``PREFIX``.dos
     - Phonon density of states
   * - ``analyzer.py``
     - ``PREFIX``.kappa.h5
     - Phonon lifetimes, mean free paths, cumulative thermal conductivity

For example::

  $ plotband.py PREFIX.bands
  $ analyzer.py --calc tau --temp 300 --h5 PREFIX.kappa.h5
  $ analyzer.py --calc cumulative --temp 300 --h5 PREFIX.kappa.h5

Each script prints its options with ``-h``.
Complete examples are given in the :doc:`tutorial`.
