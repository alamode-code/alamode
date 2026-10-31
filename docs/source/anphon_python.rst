.. _anphon_python:

anphon (phonon properties)
==========================

.. warning::

   The ``anphon`` Python API is **experimental** in version 2.0. It is tested
   against the command-line program, but its interface may still change in
   later releases; see :ref:`anphon_python_stability`.

Overview
--------

The Python package ``anphon`` runs anphon inside the Python process. A run goes
through the same code path as the command-line program (``anphon file.in``):
the input is ordinary anphon input text, parsed by anphon itself, and the
results are the usual output files. The package adds three conveniences on top:

* ``anphon.run`` writes the input file from Python dictionaries (or takes an
  existing input file) and runs it;
* ``anphon.Result`` locates the output files and reads the HDF5 results
  (``PREFIX.kappa.h5``, ``PREFIX.scph.h5``, ``PREFIX.qha.h5``, ...) into nested
  dictionaries;
* ``anphon.Phonon`` returns harmonic phonon frequencies and eigenvectors at
  arbitrary :math:`q` points as NumPy arrays.

Everything is file based: the inputs are files (force constants, ``BORNINFO``,
...), and the results are read back from the files anphon writes.

One Python distribution, ``alamode``, provides both packages,
:ref:`alm <alm_python>` and ``anphon`` (see :ref:`label_python_api`).


.. _anphon_python_install:

Installation
------------

The ``anphon`` package is built by the same command as ``alm``
(see :ref:`alm_python_install`)::

   % cd python
   % pip install .

On top of the ``alm`` requirements, the anphon module needs an **MPI** library
at build and run time, and ``h5py`` at run time (installed by ``pip`` as a dependency).
The conda environment file ``etc/alamode-environment.yml`` and the
:ref:`pixi route <install_pixi>` already provide OpenMPI, HDF5, and h5py.

* If CMake cannot find MPI, the configure step fails with a message. To build
  only ``alm`` on a machine without MPI, pass::

     % pip install . -C cmake.define.ALAMODE_PYTHON_ANPHON=OFF

* The build compiles the anphon sources. Set ``CMAKE_BUILD_PARALLEL_LEVEL``
  to bound the number of parallel compile jobs, e.g.
  ``CMAKE_BUILD_PARALLEL_LEVEL=4 pip install .``.

Verify the installation with:

.. code-block:: python

   import alm, anphon
   print(anphon.__version__)   # same as alm.__version__ and `pip show alamode`

After an ALM-only build (``ALAMODE_PYTHON_ANPHON=OFF``), only ``import alm``
works.


.. _anphon_python_run:

Running anphon: ``anphon.run``
------------------------------

``anphon.run`` has two forms. With an existing input file, it runs the file as
the command ``anphon si.in`` would:

.. code-block:: python

   import anphon
   r = anphon.run("si.in")                  # outputs in the current directory
   r = anphon.run("si.in", workdir="run1")  # outputs in run1/

With keyword arguments, one per input section, it writes the input file
``<workdir>/<PREFIX>.in`` and runs it:

.. code-block:: python

   r = anphon.run(
       general=dict(PREFIX="si", MODE="phonons", FCSFILE="si222.xml"),
       cell=dict(a=10.263, lattice=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]),
       kpoint=dict(KPMODE=2, mesh=[20, 20, 20]),
       workdir="dos",
   )

The input tags are those of the command-line program (:doc:`anphon_root`).
The section values are turned into input text as follows:

* A ``dict`` (``general``, ``scph``, ``kappa``, ``analysis``, ...) becomes one
  ``KEY = value`` line per entry. Keys are written in upper case, lists and
  arrays are joined by spaces (``KD=["Ba", "Ti", "O"]`` gives ``KD = Ba Ti O``),
  ``True``/``False`` become ``1``/``0``, and entries whose value is ``None`` are
  dropped. A value that contains a newline, ``;``, ``#``, or ``=``, or a
  non-finite number, raises ``ValueError``.
* ``cell`` is ``dict(a=<factor>, lattice=<3x3>)``, as in the ``&cell``
  field: the rows of ``lattice`` (dimensionless) multiplied by ``a`` (in bohr,
  default 1.0) are the lattice vectors
  :math:`\boldsymbol{a}_1, \boldsymbol{a}_2, \boldsymbol{a}_3` in bohr.
  Alternatively, give ``lattice`` in bohr and leave ``a = 1``. A bare 3x3
  array is accepted as the lattice with ``a = 1``.
* ``kpoint`` is a dict with ``KPMODE`` and

  * ``KPMODE=0``: ``qpoints=[[qx, qy, qz], ...]``;
  * ``KPMODE=1``: ``path=[("G", [0, 0, 0], "X", [0.5, 0.5, 0], 51), ...]``,
    one ``(label1, q1, label2, q2, npoints)`` tuple per line segment;
  * ``KPMODE=2``: ``mesh=[n1, n2, n3]``;
  * any mode: ``lines=[...]``, the lines after the ``KPMODE`` line, verbatim.

  ``KPMODE``, the mesh, and the numbers of points must be integers
  (``4.0`` is accepted, ``4.5`` or ``True`` raise ``ValueError``).
* Any other non-dict value (e.g. a 3x3 matrix for ``strain``) is written one
  line per row.
* A ``str`` is written verbatim as the body of the section. This is the escape
  hatch for anything the rules above do not cover.

Only the form is checked in Python; the content is validated by anphon's input
parser, whose errors raise ``RuntimeError`` (see :ref:`anphon_python_errors`).
Passing both an input file and sections raises ``TypeError``, and a missing
``general`` section or ``PREFIX`` raises ``ValueError``.
``anphon.format_input(**sections)`` returns the generated text without running
anything, which is useful to check it:

.. code-block:: python

   print(anphon.format_input(general=dict(PREFIX="si", MODE="phonons", FCSFILE="si.xml"),
                             kpoint=dict(KPMODE=2, mesh=[8, 8, 8])))

.. code-block:: none

   &general
     PREFIX = si
     MODE = phonons
     FCSFILE = si.xml
   /

   &kpoint
     2
     8 8 8
   /

``workdir`` (default: the current directory) is created if it does not exist.
anphon runs inside it, so **relative file names in the input** (``FCSFILE``,
``BORNINFO``, ...) **are relative to** ``workdir``, as are all outputs. The
current directory of the Python process is restored when the run ends, also
after an error. Use absolute paths when in doubt.

anphon prints its usual log to standard output. Set ``VERBOSITY = 0`` in
``general`` to make it shorter; a run from Python then also omits the banner
and the start and finish lines. The log is written by C++, so when the output
is redirected to a file, it can interleave out of order with text printed by
Python.


.. _anphon_python_result:

Results: ``anphon.Result``
--------------------------

``run`` returns a ``Result`` with the attributes ``workdir`` (absolute path)
and ``prefix``:

* ``r.path(".bands")`` returns ``workdir / (prefix + ".bands")``;
* ``r.files()`` lists the files ``<prefix>.*`` in ``workdir`` (including the
  input file ``<prefix>.in`` written by ``run``);
* ``r.h5(kind)`` reads ``PREFIX.<kind>.h5`` into a nested dictionary, e.g.
  ``r.h5("kappa")``, ``r.h5("scph")``, or ``r.h5("qha")``
  (``FileNotFoundError`` if the file does not exist). It returns the raw file
  contents.

The text outputs (``.bands``, ``.dos``, ...) are read with the usual tools,
e.g. ``numpy.loadtxt(r.path(".bands"))``.

``r.h5`` uses ``anphon.read_h5(path)``, which works for any HDF5 file:

* groups become dicts, datasets NumPy arrays (scalars as NumPy scalars);
* strings are decoded to ``str`` (string arrays become lists of ``str``);
* an attribute of a group is stored in that group's dict under ``"@name"``,
  and an attribute of a dataset under ``"dataset@name"`` next to the dataset.

For example, for a ``MODE = kappa`` run:

.. code-block:: python

   k = r.h5("kappa")
   k["@schema"]                         # 'alamode:kappa_result'
   T = k["metadata"]["temperatures"]    # K
   kap = k["kappa"]["kappa_peierls"]    # shape (nT, 3, 3)
   k["kappa"]["kappa_peierls@unit"]     # 'W/mK'

The layouts of the files are described in :doc:`hdf5_format`
(:ref:`label_hdf5_kappa`, :ref:`label_hdf5_scph`).


.. _anphon_python_phonon:

Phonons at arbitrary q: ``anphon.Phonon``
-----------------------------------------

``Phonon`` returns harmonic frequencies and eigenvectors as arrays:

.. code-block:: python

   ph = anphon.Phonon("si222_harmonic.h5")
   q = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5]]
   w = ph.frequencies(q)
   w, evec = ph.frequencies(q, eigenvectors=True)

* ``Phonon(fcsfile, *, cell=None, kd=None, borninfo=None, nonanalytic=0, general=None)``
  takes the settings. They are validated, and the files are loaded, at
  construction or at the first calculation. ``cell`` has the form described
  for ``run``.
  An HDF5 force-constant file written by ALM 2.0 carries the primitive cell, so
  ``cell`` and ``kd`` are not needed; an XML file does not, so ``cell=`` must
  be given. Other ``&general`` tags (``MASS``, ``TOLERANCE``, ``VERBOSITY``, ...)
  go in the dict ``general``. ``VERBOSITY`` defaults to 0 here, so ``Phonon``
  prints only warnings; set ``VERBOSITY=1`` to see the anphon log. The tags ``PREFIX``, ``MODE``, ``FCSFILE``,
  ``BORNINFO``, ``NONANALYTIC``, ``FILE_FORMAT``, and ``KD`` are set by
  ``Phonon`` itself (or through its arguments) and raise ``ValueError`` in
  ``general``, in any letter case. ``Phonon`` keeps its own copy of the
  settings, so changing the dicts passed to it afterwards has no effect.
* ``frequencies(q, eigenvectors=False)`` takes :math:`q` points in fractional
  coordinates of the reciprocal primitive lattice (shape ``(nq, 3)`` or
  ``(3,)``; other shapes raise ``ValueError``) and returns the frequencies in
  cm\ :sup:`-1` with shape ``(nq, nmodes)``, ``nmodes = 3 * natoms`` of the
  primitive cell. Imaginary modes come out as negative numbers. With ``eigenvectors=True`` it also
  returns the complex eigenvectors of the dynamical matrix with shape
  ``(nq, nmodes, nmodes)``, laid out as ``evec[iq, imode, 3 * iatom + ixyz]``
  (axis 1 is the mode).
* Each call runs anphon (``MODE = phonons``, ``KPMODE = 0``) in a temporary
  directory, which is removed afterwards. Ask for all :math:`q` points in one
  call rather than one call per point.
* ``Phonon`` is serial only: it raises ``RuntimeError`` when MPI runs with more
  than one process.

.. note::

   **Non-analytic correction at** :math:`\Gamma`. With ``nonanalytic > 0``,
   the direction of :math:`q` used by the non-analytic term is obtained by
   shifting each fractional component of :math:`q` by the nearest integer
   (:math:`q_i - \mathrm{nint}(q_i)`) and converting the result to Cartesian
   coordinates. At :math:`\Gamma` and at every
   reciprocal-lattice point (``[0, 0, 0]``, ``[1, 0, 0]``, ...), this direction
   is zero and the non-analytic term vanishes, so there is **no LO-TO splitting
   exactly at these points**. To get the LO frequencies for a given direction,
   use a small :math:`q` along it (e.g. ``[1e-4, 0, 0]``), or run a band
   structure (``KPMODE = 1``) with ``run``, which takes the direction of the
   path.


.. _anphon_python_errors:

Errors
------

* In a serial run, errors while ``run`` prepares the input in Python raise
  ordinary Python exceptions: ``TypeError`` for an input file plus sections,
  ``ValueError`` for a missing ``general`` section or ``PREFIX`` or a bad
  value, and ``OSError`` from the filesystem.
* In a serial run, errors that anphon reports (invalid input, a missing or
  broken file, ...) raise ``RuntimeError`` with anphon's error message. The
  message is also printed to the log first. The process stays usable: the
  current directory is restored, and the next run starts from a clean state.
* Errors raised inside an OpenMP parallel region, and out-of-memory errors,
  still terminate the process, as in the command-line program.
* Under MPI with more than one process, errors in ``run`` abort all ranks
  (``MPI_Abort``): both an error while ``run`` prepares the input in Python,
  even on one rank only, and a failure inside the collective anphon run. This
  avoids a deadlock in which the other ranks wait forever. Errors outside a
  run still raise locally: ``format_input`` and ``read_h5`` errors, the errors
  of ``Phonon`` (``TypeError`` or ``ValueError`` for its arguments, ``RuntimeError`` because it
  is serial only), and the ``RuntimeError`` for a
  call from a thread other than the main thread, which is raised before the
  run starts.


.. _anphon_python_mpi:

MPI
---

anphon is parallelized with MPI and OpenMP, and the Python interface runs
under MPI as the command-line program does:

.. code-block:: none

   % mpirun -np 4 python script.py

* Importing ``anphon`` initializes MPI (``MPI_THREAD_FUNNELED``) unless it is
  already initialized, e.g. by ``mpi4py``. Initialization is collective, so in
  that case **all ranks** must import ``anphon`` (an import on rank 0 only can
  hang). Both orders of ``import mpi4py`` / ``import anphon`` work. MPI that
  is already initialized must provide at least ``MPI_THREAD_FUNNELED``;
  otherwise the import raises ``ImportError``. MPI is finalized at exit only
  if ``anphon`` initialized it.
* ``mpi4py`` and the anphon module must be built against the **same MPI
  library**.
* Every ``run`` call is **collective** over ``MPI_COMM_WORLD``: all ranks
  must call it with the same arguments. Rank 0
  writes the outputs, and every rank returns only after they are complete, so
  all ranks can read them afterwards.
* Import ``anphon`` and call ``run`` from the main Python thread only (other
  threads get ``ImportError`` / ``RuntimeError``). Python threads keep running
  during a run, but cannot start another one.

.. note::

   **Reproducibility.** OpenMP reductions can make the results of two runs
   differ in the last bit, as for the command-line program. With
   ``OMP_NUM_THREADS=1``, the results are numerically identical to those of
   the command-line program with the same build and settings (all datasets
   equal); only the metadata attributes ``created_date``,
   ``alamode_git_commit``, and ``alamode_version`` differ.


.. _anphon_python_example:

Example: silicon
----------------

The following script uses the silicon force constants shipped in
``example/Si``. Copy them to a working directory first::

   % cp example/Si/anharm_IFCs/1_harmonic/reference/si222_harmonic.xml .
   % cp example/Si/anharm_IFCs/2_generate_config/si222_harmonic.h5 .
   % cp example/Si/reference/si222_cubic.h5 .

It computes the phonon dispersion, the lattice thermal conductivity in the
relaxation-time approximation (on a coarse :math:`4\times4\times4` mesh, which
takes a few seconds), and the frequencies at :math:`\Gamma`, X, and L.

.. code-block:: python

   import os
   import numpy as np
   import anphon

   here = os.getcwd()
   fcc = [[0.0, 0.5, 0.5],
          [0.5, 0.0, 0.5],
          [0.5, 0.5, 0.0]]

   # 1. phonon dispersion along G-X-G-L
   bands = anphon.run(
       general=dict(PREFIX="si", MODE="phonons", VERBOSITY=0,
                    FCSFILE=os.path.join(here, "si222_harmonic.xml")),
       cell=dict(a=10.2631026, lattice=fcc),
       kpoint=dict(KPMODE=1, path=[
           ("G", [0, 0, 0], "X", [0.5, 0.5, 0], 51),
           ("X", [0.5, 0.5, 1], "G", [0, 0, 0], 51),
           ("G", [0, 0, 0], "L", [0.5, 0.5, 0.5], 51),
       ]),
       workdir="bands",
   )
   print(bands.files())                         # ['si.bands', 'si.in']
   b = np.loadtxt(bands.path(".bands"))         # (153, 7): path coordinate, 6 branches

   # 2. thermal conductivity (RTA)
   res = anphon.run(
       general=dict(PREFIX="si", MODE="kappa", VERBOSITY=0,
                    FCSFILE=os.path.join(here, "si222_cubic.h5"),
                    TMIN=100, TMAX=500, DT=100),
       kpoint=dict(KPMODE=2, mesh=[4, 4, 4]),
       kappa=dict(SOLVER="RTA"),
       workdir="kappa",
   )
   k = res.h5("kappa")
   for T, kap in zip(k["metadata"]["temperatures"], k["kappa"]["kappa_peierls"]):
       print(f"{T:5.0f} K  {kap[0, 0]:8.2f} W/mK")

   # 3. frequencies at Gamma, X, and L (the .h5 file carries the primitive cell)
   ph = anphon.Phonon("si222_harmonic.h5")
   w, evec = ph.frequencies([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5]],
                            eigenvectors=True)
   print(np.round(w, 2))                        # cm^-1, shape (3, 6)
   print(evec.shape)                            # (3, 6, 6)

   # the same with the XML file, which needs the cell
   ph_xml = anphon.Phonon("si222_harmonic.xml", cell=dict(a=10.2631026, lattice=fcc))

The frequencies printed in step 3 are:

.. code-block:: none

   [[  0.     0.     0.   509.62 509.62 509.62]
    [134.17 134.17 405.55 405.55 458.65 458.65]
    [103.5  103.5  368.81 410.16 486.47 486.47]]

The coarse mesh of step 2 gives a thermal conductivity far from convergence
(34 W/mK at 300 K, against 104 W/mK on the :math:`10\times10\times10` mesh of
``example/Si/reference/si222_kappa.in``); use a denser mesh for real
calculations.


.. _anphon_python_limitations:

Limitations
-----------

The following are not available in this version:

* in-memory input of force constants and structures: force constants from
  ``alm`` go through a file (``ALM.save_fc``), which is then passed as
  ``FCSFILE``;
* a persistent ``Phonon`` object: each ``frequencies`` call runs anphon anew;
* sub-communicators: runs always use ``MPI_COMM_WORLD``.


.. _anphon_python_stability:

API stability
-------------

* **Intended to stay stable in 2.x:** ``run`` (the file runner) and its documented arguments;
  ``Result.workdir``, ``Result.prefix``, ``Result.path()``, and
  ``Result.files()``; ``format_input``; ``__version__``; the unit and shape
  conventions of ``Phonon.frequencies`` (cm\ :sup:`-1`, fractional
  :math:`q`, shape ``(nq, nmodes)``, ``evec[iq, imode, 3 * iatom + ixyz]``).
* **Provisional** (may change in a 2.x minor release, with release notes):
  ``Phonon`` (its implementation and any further arguments); ``Result.h5``
  and ``read_h5``, whose content is the raw file schema, versioned separately
  in :doc:`hdf5_format`.
* **Deprecation rule:** a stable name is not removed or changed incompatibly
  in 2.x. A deprecation warns for at least one minor release before the
  removal in a major release.


.. _anphon_python_api:

API reference
-------------

``anphon.run(input=None, *, workdir=None, **sections)``
   Run anphon on the input file ``input``, or on the input generated from
   ``sections``, inside ``workdir``. Returns a ``Result``. Collective under MPI.
   The keyword-only argument names (``workdir``, and future ones such as
   ``comm``) are reserved; all other keywords are input sections.

``anphon.format_input(**sections)``
   Return the input text that ``run`` would write for ``sections``.

``anphon.Result(workdir, prefix)``
   Outputs of a run. Attributes ``workdir``, ``prefix``; methods
   ``path(suffix)``, ``files()``, ``h5(kind)``.

``anphon.read_h5(path)``
   Read an HDF5 file into a nested dict (``"@name"`` and ``"dataset@name"``
   keys for attributes).

``anphon.Phonon(fcsfile, *, cell=None, kd=None, borninfo=None, nonanalytic=0, general=None)``
   Harmonic phonons from a force-constant file. Method
   ``frequencies(qpoints_frac, eigenvectors=False)``. Serial only.

``anphon.__version__``
   Version of the ``alamode`` distribution.

The low-level binding underneath ``run`` is private.


See also
--------

* :doc:`anphon_root` — the anphon command-line program and its input tags.
* :doc:`alm_python` — the ALM Python interface, which writes the force-constant
  files used here.
* :doc:`hdf5_format` — layouts of the HDF5 files read by ``Result``.
* :ref:`label_tutorial_si_python` — a tutorial that uses ``run``, ``Phonon``, and
  ``Result.h5`` on silicon.
