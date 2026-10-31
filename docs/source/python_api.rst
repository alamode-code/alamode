.. _label_python_api:

Python API
==========

One Python distribution, ``alamode``, provides two packages:

* ``alm`` fits force constants from displacement--force data inside Python, with
  the built-in solvers or with any external regressor, and writes them to the
  usual force-constant files;
* ``anphon`` runs anphon inside the Python process and reads its results back as
  NumPy arrays and dictionaries.

Both packages are built by a single ``pip install .`` in the ``python``
directory of the source tree; see :ref:`alm_python_install` and
:ref:`anphon_python_install`. The tutorial :ref:`label_tutorial_si_python` uses both
packages on silicon.

Wheels for Python 3.11 and later, on Linux (x86_64 and aarch64) and macOS (arm64,
macOS 15 or later), are also published on PyPI, so the packages can be installed without
a compiler::

   % pip install "alamode[openmpi]"

The ``openmpi`` extra installs Open MPI from PyPI, which ``anphon`` needs. In a conda
environment, install ``openmpi`` from conda-forge instead and run ``pip install alamode``
without the extra, since the Open MPI from PyPI would replace conda's ``mpirun`` and
``libmpi``. On Cray systems, remove Cray's libfabric from ``LD_LIBRARY_PATH`` in the shell
that runs ``anphon``, because the loader finds it before the libfabric that comes with
the Open MPI from PyPI::

   % export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v libfabric | paste -sd: -)

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`gear;1.2em` alm (force constants)
      :link: alm_python
      :link-type: ref

      Build a force-constant model, get ALM's sensing matrix, fit it with
      ALM's solvers or an external regressor, and save the result.

   .. grid-item-card:: :octicon:`pulse;1.2em` anphon (phonon properties)
      :link: anphon_python
      :link-type: ref

      Run anphon from Python dictionaries, read the HDF5 results, and get
      phonon frequencies and eigenvectors at arbitrary :math:`q` points.

The two packages share the force-constant file: ``alm`` writes it with
``save_fc(..., format="alamode_h5")`` and ``anphon`` reads it. The script below
fits the harmonic force constants of silicon to the displacement--force data
of ``example/Si/anharm_IFCs/1_harmonic`` (the 64-atom supercell of
``si_alm_opt.in``) and computes the frequencies at :math:`\Gamma` and X:

.. code-block:: python

   import numpy as np
   import alm, anphon

   # 64-atom cubic supercell of si_alm_opt.in (Bohr, fractional coordinates)
   lavec = 20.5262052 * np.eye(3)
   block = open("si_alm_opt.in").read().split("&position")[1].split("/")[0]
   xcoord = np.array([[float(x) for x in line.split()[1:4]]
                      for line in block.splitlines() if line.strip()])

   # DFSET: one line "ux uy uz fx fy fz" per atom (Bohr, Ry/Bohr)
   data = np.loadtxt("DFSET_harmonic").reshape(-1, 64, 6)

   a = alm.ALM(lavec, xcoord, [14] * 64, verbosity=0)
   a.set_supercell(np.eye(3), autoset_primcell=True)   # find the primitive cell
   a.define(maxorder=1)                                 # harmonic, no cutoff
   a.set_training_data(data[:, :, :3], data[:, :, 3:])
   a.optimize()
   a.save_fc("si_harmonic.h5", format="alamode_h5")

   # the .h5 file carries the primitive cell, so no cell is needed here
   ph = anphon.Phonon("si_harmonic.h5")
   print(ph.frequencies([[0, 0, 0], [0.5, 0.5, 0]]))   # cm^-1

.. toctree::
   :maxdepth: 2

   alm_python
   anphon_python
