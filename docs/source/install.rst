Installation
============

Requirement
-----------

Mandatory requirements
~~~~~~~~~~~~~~~~~~~~~~~~

* C++ compiler (supporting the C++17 standard)
* LAPACK and BLAS libraries
* MPI library (OpenMPI, MPICH2, IntelMPI, etc.)
* `Boost C++ library <http://www.boost.org>`_ (version >= 1.66)
* `Eigen3 library <http://eigen.tuxfamily.org/>`_
* `spglib <https://atztogo.github.io/spglib/>`_
* `HDF5 library <https://www.hdfgroup.org/solutions/hdf5/>`_
* `CMake <https://cmake.org>`_ (version >= 3.17 and < 4.0)

All of these libraries can be installed easily with conda or pixi.

In addition to the above requirements, users need to install a first-principles package
(such as VASP_, QUANTUM-ESPRESSO_, OpenMX_, or xTAPP_) or a force-field package (such as
LAMMPS_) by themselves to compute harmonic and anharmonic force constants.

.. _VASP: http://www.vasp.at
.. _OpenMX: http://www.openmx-square.org
.. _QUANTUM-ESPRESSO: http://www.quantum-espresso.org
.. _xTAPP: http://frodo.wpi-aimr.tohoku.ac.jp/xtapp/index.html
.. _LAMMPS: http://lammps.sandia.gov


Optional requirements
~~~~~~~~~~~~~~~~~~~~~~~

* Python (>= 3.x), Numpy, and Matplotlib
* XcrySDen_ or VMD_

Small Python scripts for visualizing phonon dispersion relations, phonon DOS, etc. are provided;
they require the Python packages listed above.
Additionally, XcrySDen is necessary to visualize the normal-mode directions and animate the normal modes.
VMD may be more convenient for making animations, but any other visualization software that supports the XYZ format can be used instead.

.. _XcrySDen: http://www.xcrysden.org
.. _VMD: http://www.ks.uiuc.edu/Research/vmd/


Install using conda (recommended for non-experts)
-------------------------------------------------

This option is recommended for users who want a reproducible build with minimal
manual setup. It follows the same conda environment used by the GitHub Actions
tests. If you need the best performance for large production runs, especially
large sparse least-squares fits, see the :ref:`native installation <install_native>`
section and the :ref:`linear-algebra backend notes <install_backends>`.


Step 1. Prepare build tools with conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, prepare a conda environment named ``alamode``.
::

   % conda create --name alamode -c conda-forge python=3.10
   % conda activate alamode

If ``git`` is not available on your system, install it before downloading the
source, for example with ``conda install -c conda-forge git`` after activating
the environment. After the source tree is downloaded in Step 2, update this
environment from ``etc/alamode-environment.yml``. That environment file installs
the compiler tools, OpenMPI, Boost, Eigen, CMake, spglib, HDF5, FFTW, NumPy,
SciPy, and h5py from conda-forge, matching the GitHub Actions workflow.


Step 2. Download source 
~~~~~~~~~~~~~~~~~~~~~~~

Download the source files from the GitHub repository::

  % git clone https://github.com/alamode-code/alamode.git
  % cd alamode
  % git checkout 2.0dev
  % conda env update -n alamode -f etc/alamode-environment.yml
  % conda activate alamode

Use the branch that corresponds to the documentation you are reading. For the
development documentation and the current GitHub workflow, this is ``2.0dev``.
The directory structure assumed in this document is shown below::

   $HOME
    ├── alamode
    │   ├── CMakeLists.txt
    │   ├── alm
    │   │   └── CMakeLists.txt
    │   ├── anphon
    │   │   └── CMakeLists.txt
    │   ├── docs
    │   ├── example
    │   ├── external
    │   ├── include
    │   └── tools
    │       └── CMakeLists.txt
    │
    ├── $CONDA_PREFIX/include
    ├── $CONDA_PREFIX/include/eigen3
    ├── $CONDA_PREFIX/lib
    ├── ...

The meaning of each subdirectory is as follows:

  * alm/      : Source files of alm (force constant calculator)
  * anphon/   : Source files of anphon (anharmonic phonon calculator)
  * docs/     : Source files for making documents
  * example/  : Example files
  * external/ : Third-party include files
  * include/  : Commonly-used include files
  * tools/    : Small auxiliary programs and scripts


Step 3. Build by CMake 
~~~~~~~~~~~~~~~~~~~~~~~

If you want to build all binaries (**alm**, **anphon**, and the others), please use ``CMakeLists.txt`` in the ``$HOME/alamode`` directory.
::

  % pwd
  * $HOME/alamode
  % mkdir _build; cd _build
  % cmake ..

Please make sure that cmake detected the C++ compiler correctly. If the automatic detection fails, you can specify the compilers
by using the ``-DCMAKE_C_COMPILER`` and ``-DCMAKE_CXX_COMPILER`` options. If ``${CC}`` and ``${CXX}`` variables are not set properly,
you may need to ``conda deactivate`` once and ``conda activate alamode`` again.

After the cmake configuration finishes, build the binaries by
::

  % make -j

It will create all binaries in alm/, anphon/, and tools/ subdirectories under the current directory (_build). 
You can specify the binary to build, for example, as
::

  % make alm -j

.. note::

    If the build of **alm** fails due to an error related to spglib, e.g., ``cannot find -lsymspg``, 
    please add the ``-DSPGLIB_ROOT`` option as
    ::
      
      % cmake -DSPGLIB_ROOT=$CONDA_PREFIX ..

    Also, when using the binaries, it may be necessary to set ``$LD_LIBRARY_PATH`` as
    ::

      % export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH


Installing the binaries and scripts (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The binaries can be used directly from the build directory, so this step is optional.
To install ALAMODE system-wide (or into any prefix of your choice), run
::

  % cmake --install . --prefix /desired/prefix

or equivalently ``make install`` after configuring with ``-DCMAKE_INSTALL_PREFIX=/desired/prefix``
(the default prefix is ``/usr/local``, which usually requires ``sudo``).

This installs

* the executables (``alm``, ``anphon``, ``dfc2``, ``qe2alm``, ``fc_virtual``, ``parse_fcsxml``) into ``<prefix>/bin``,
* the python scripts of the tools directory into ``<prefix>/share/alamode/tools``, and
* symbolic links in ``<prefix>/bin`` for the scripts used in the tutorials
  (``analyzer.py``, ``dfc2.py``, ``displace.py``, ``extract.py``, ``plotband.py``, ``plotdos.py``, ``scph_to_qefc.py``),
  so they can be invoked by name once ``<prefix>/bin`` is on your ``$PATH``.

.. note::

    The python scripts require NumPy and, depending on the script, matplotlib, h5py, or ASE.
    These must be available in the python environment (``python3``) from which you run the
    scripts; ``make install`` does not install any python dependencies.


.. _install_pixi:

Install using pixi (project-local, reproducible)
------------------------------------------------

`pixi <https://pixi.sh>`_ is a fast package manager for the same conda-forge
ecosystem used in the conda instructions above. The main difference from conda
is that pixi manages a **project-local** environment: all packages are
installed under ``.pixi/`` inside the source tree, together with a lockfile
(``pixi.lock``) that records the exact version of every dependency. Nothing is
installed globally, no ``conda activate`` is needed, different checkouts or
branches can have independent toolchains, and removing the environment is as
simple as deleting the ``.pixi`` directory.


Step 1. Install pixi
~~~~~~~~~~~~~~~~~~~~

Follow the `official instructions <https://pixi.sh/latest/installation/>`_,
for example::

  % curl -fsSL https://pixi.sh/install.sh | sh

or, with Homebrew on macOS, ``brew install pixi``. Restart the terminal after
the installation so that the ``pixi`` command is found.


Step 2. Download source and create the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  % git clone https://github.com/alamode-code/alamode.git
  % cd alamode
  % git checkout 2.0dev
  % pixi init --import etc/alamode-environment.yml
  % pixi add "python=3.12" pip make

``pixi init --import`` seeds the project manifest (``pixi.toml``) from the
same conda environment file used by the conda instructions and the GitHub
Actions workflow, so the toolchain (compilers, OpenMPI, Boost, Eigen, CMake,
spglib, HDF5, FFTW, NumPy, SciPy, and h5py) is identical. The ``pixi add``
line pins Python and adds ``pip`` and ``make``, which are used by the build
and by the :ref:`Python wrapper <alm_python_install>`. Both commands solve the
dependencies, install them into ``.pixi/envs/default``, and write
``pixi.lock``.

The manifest is created for the platform the command is run on; it is a local
file (ignored by git), so users on other platforms simply generate their own.


Step 3. Build by CMake
~~~~~~~~~~~~~~~~~~~~~~

Run the usual CMake commands inside the pixi environment, either from an
activated shell::

  % pixi shell
  % cmake -S . -B _build -DCMAKE_BUILD_TYPE=Release
  % cmake --build _build -j
  % exit

or by prefixing each command with ``pixi run``::

  % pixi run 'cmake -S . -B _build -DCMAKE_BUILD_TYPE=Release'
  % pixi run 'cmake --build _build -j'

ALAMODE's CMake setup automatically searches the active environment prefix
(pixi sets ``$CONDA_PREFIX``, exactly like conda), so no ``-DSPGLIB_ROOT`` or
similar options are necessary. The binaries are created in ``_build/alm/``,
``_build/anphon/``, and ``_build/tools/`` with an rpath pointing at the
environment's library directory, so setting ``$LD_LIBRARY_PATH`` is not
required when they are launched via ``pixi shell`` or ``pixi run``.


Step 4. Build and install the Python wrapper (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :ref:`ALM Python interface <alm_python>` builds in the same environment
with a single command::

  % pixi run 'cd python && pip install .'

The wrapper's CMake setup likewise picks up spglib, Boost, Eigen, and HDF5
from the environment prefix automatically. Verify the installation with::

  % pixi run python -c 'import alm; print(alm.__version__)'


Optional: register pixi tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Frequently used commands can be stored in the manifest as tasks::

  % pixi task add configure 'cmake -S . -B _build -DCMAKE_BUILD_TYPE=Release'
  % pixi task add build 'cmake --build _build -j'
  % pixi task add install-python 'cd python && pip install .'

after which the whole build becomes::

  % pixi run configure
  % pixi run build
  % pixi run install-python

.. note::

    ``pixi.toml``, ``pixi.lock``, and the ``.pixi/`` environment directory are
    local to your checkout and ignored by git. If you want to share a
    bit-reproducible environment with collaborators, commit ``pixi.toml`` and
    ``pixi.lock`` to your own fork.

.. note::

    On macOS, the conda-forge compilers use the system SDK, so the Xcode
    Command Line Tools must be installed (``xcode-select --install``). This
    applies to the conda route as well.

.. note::

    **Recent macOS SDKs (macOS 26 "Tahoe" and newer).** If your Xcode Command
    Line Tools provide a very new macOS SDK, the conda-forge linker
    (``cctools``/``ld64``) bundled in the environment may not yet support it,
    and ``cmake ..`` fails during MPI detection even though ``mpicc`` works::

        -- Could NOT find MPI_C (missing: MPI_C_WORKS)
        -- Could NOT find MPI_CXX (missing: MPI_CXX_WORKS)

    This is not actually an MPI problem: FindMPI's test link fails on base
    ``libSystem`` symbols (``_puts``, ``___stack_chk_fail``), so *any*
    non-trivial link through the conda compiler would fail -- MPI is simply the
    first library checked. Until conda-forge ships a ``cctools``/``ld64`` that
    supports the new SDK, configure with Apple's system Clang while still
    linking the conda libraries::

        % env -u CC -u CXX -u FC cmake .. \
              -DCMAKE_C_COMPILER=/usr/bin/clang \
              -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
              -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
              -DHDF5_ROOT=$CONDA_PREFIX

    OpenMP still works this way (Apple Clang uses ``-Xclang -fopenmp`` with the
    conda ``llvm-openmp`` runtime). For portable binaries, also set
    ``-DCMAKE_OSX_DEPLOYMENT_TARGET=<your minimum>`` and verify the result with
    ``otool -L`` so that only one copy of ``libc++`` / ``libomp`` is loaded.


.. _install_native:

Install using native environment (recommended for performance-critical runs)
----------------------------------------------------------------------------

Use a native installation when performance matters more than convenience:
large supercells, large sparse sensing matrices, numerically constrained sparse
OLS fits (``SPARSE = 1`` with ``ICONST = 1, 2, 3, 4``), or repeated production
runs on HPC systems. The conda build is convenient and reproducible, but native
compilers and optimized sparse backends such as MKL/PARDISO or SuiteSparse can
be substantially faster and more memory-efficient for these large-scale sparse
problems.

The example below uses Intel oneAPI compilers and the MKL/PARDISO backend
(``-DUSE_MKL_BACKEND=yes``), which is the recommended high-performance
configuration when available.


.. _install_backends:

Optional linear-algebra backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, ALAMODE links against the LAPACK/BLAS libraries detected by CMake.
This is sufficient for small and medium calculations. For large sparse
least-squares problems, especially ``LMODEL = ols``, ``SPARSE = 1`` with
numerical constraints ``ICONST = 1, 2, 3, 4``, a native build with optimized
sparse direct solvers can be much faster and more memory-scalable.

For the **KKT solver** (the symmetric-indefinite system of the numerically
constrained sparse fit) the recommended priority is:

1. **Intel MKL / PARDISO** on Linux or Intel oneAPI systems.
2. **SuiteSparse / SPQR + CHOLMOD** when SuiteSparse is available.
3. **Apple Accelerate** on macOS when MKL is unavailable.

These three are mutually exclusive *as the KKT solver*. **SuiteSparse, however,
is an orthogonal add-on, not just a KKT-solver choice:** ``-DUSE_SUITESPARSE_BACKEND=yes``
can be combined with ``-DUSE_MKL_BACKEND=yes`` (or ``-DUSE_ACCEL_BACKEND=yes``).
In that combined build the LDLT backend (PARDISO / Accelerate) still solves the
KKT system, while the multithreaded **SuiteSparseQR** replaces the dense LAPACK
``dgeqp3`` for the rank-revealing **constraint-matrix reduction** -- which can
otherwise dominate the runtime on large numerically-constrained fits (e.g. a
254k-parameter ``ICONST = 2`` reduction dropped from ~9 min to ~10 s). The
recommended high-performance configuration on Intel systems is therefore
``-DUSE_MKL_BACKEND=yes -DUSE_SUITESPARSE_BACKEND=yes`` together. (Build
SuiteSparse against the same BLAS as the solver backend -- e.g. MKL -- to avoid
linking two BLAS implementations.)

.. raw:: html

   <details>
   <summary><strong>Intel MKL / PARDISO</strong></summary>

``-DUSE_MKL_BACKEND=yes`` enables Intel MKL and uses **PARDISO**
(``Eigen::PardisoLDLT``) for the sparse KKT system used by the numerically
constrained sparse OLS path (``LMODEL = ols``, ``SPARSE = 1``,
``ICONST = 1, 2, 3, 4``). This is the preferred backend for large constrained
sparse fits. ALAMODE currently supports only the LP64 MKL interface.

Example with Intel oneAPI compilers:

.. code-block:: console

   % source /path/to/oneapi/setvars.sh
   % mkdir _build-mkl; cd _build-mkl
   % cmake -DCMAKE_BUILD_TYPE=Release \
           -DUSE_MKL_BACKEND=yes -DMKL_INTERFACE=lp64 \
           -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
           -DCMAKE_CXX_FLAGS="-O2 -xHOST" ..
   % make -j

If CMake cannot find MKL automatically, make sure the oneAPI environment has
been loaded, or add MKL's CMake package location to ``CMAKE_PREFIX_PATH``.
The MKL backend also routes Eigen dense products through MKL
(``EIGEN_USE_MKL_ALL``). ``-DUSE_EIGEN_BLAS=yes`` is ignored when the MKL
backend is enabled.

.. raw:: html

   </details>

   <details>
   <summary><strong>SuiteSparse / SPQR and CHOLMOD</strong></summary>

``-DUSE_SUITESPARSE_BACKEND=yes`` enables SuiteSparse support. This is an
add-on, not an exclusive BLAS backend: it can be combined with the default BLAS,
MKL, or Accelerate. It makes two additional ``SPARSESOLVER`` values available
for the unconstrained or algebraically constrained sparse OLS path:
``SuiteSparseQR`` and ``CHOLMOD``. It also makes SuiteSparseQR the preferred QR
fallback in the numerically constrained sparse KKT path after any MKL or
Accelerate LDLT backend.

The implementation calls the SuiteSparseQR C interface directly for SPQR,
rather than Eigen's ``SPQR`` wrapper, and uses Eigen's CHOLMOD wrapper for
CHOLMOD. Point CMake to the SuiteSparse installation prefix, i.e. the directory
that contains ``lib/cmake/SPQR`` and ``lib/cmake/CHOLMOD``.

Example:

.. code-block:: console

   % mkdir _build-suitesparse; cd _build-suitesparse
   % cmake -DCMAKE_BUILD_TYPE=Release \
           -DUSE_SUITESPARSE_BACKEND=yes \
           -DSUITESPARSE_ROOT=/path/to/suitesparse/prefix ..
   % make -j

When SuiteSparse is installed by a package manager in a standard prefix, the
``-DSUITESPARSE_ROOT`` option may not be necessary. For large constrained
sparse fits, SuiteSparseQR is usually more robust than Eigen's built-in serial
``SparseLU``/``SparseQR`` fallback, but MKL/PARDISO remains the first choice
when available.

.. raw:: html

   </details>

   <details>
   <summary><strong>Apple Accelerate</strong></summary>

``-DUSE_ACCEL_BACKEND=yes`` uses Apple's Accelerate framework on macOS. It
enables ``Eigen::AccelerateLDLT`` for the same numerically constrained sparse
KKT path where MKL/PARDISO is used, and routes Eigen dense products through
Accelerate. This is the recommended native backend on Apple systems where MKL
is not available.

Example:

.. code-block:: console

   % mkdir _build-accel; cd _build-accel
   % cmake -DCMAKE_BUILD_TYPE=Release -DUSE_ACCEL_BACKEND=yes ..
   % make -j

This backend requires an Eigen version that provides ``Eigen/AccelerateSupport``
(Eigen 3.4.90 or later).

.. raw:: html

   </details>

Generic BLAS for Eigen dense products
+++++++++++++++++++++++++++++++++++++

``-DUSE_EIGEN_BLAS=yes`` routes Eigen's dense matrix products through the BLAS
detected by CMake (``EIGEN_USE_BLAS``). For example, with OpenBLAS::

  % cmake -DCMAKE_BUILD_TYPE=Release \
          -DUSE_EIGEN_BLAS=yes -DBLA_VENDOR=OpenBLAS ..

This option is ignored when ``-DUSE_MKL_BACKEND=yes`` or
``-DUSE_ACCEL_BACKEND=yes`` is enabled, because those backends already provide
the dense BLAS route.

When none of ``-DUSE_MKL_BACKEND``, ``-DUSE_ACCEL_BACKEND``, nor
``-DUSE_SUITESPARSE_BACKEND`` is given, the numerically constrained sparse KKT
system falls back to Eigen's built-in sparse solvers (``SparseLU``, then
``SparseQR``, then ``BiCGSTAB``). This portable fallback is useful for testing
but is not recommended for very large sparse constrained fits.


Step 1. Install all required libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Boost C++ and Eigen3 libraries (header files only)
++++++++++++++++++++++++++++++++++++++++++++++++++

(If boost and Eigen3 are already installed in your system, please skip this.)

Some header files of Boost C++ and Eigen3 libraries are necessary to build ALAMODE binaries.
Here, we install header files of these libraries in ``$(HOME)/include``.
You can skip this part if these libraries are already installed on your system.
   
To install the Boost C++ library, please download a source file from the `webpage <http://www.boost.org>`_ and
unpack the file. Then, copy the 'boost' subdirectory to ``$(HOME)/include``.
This can be done as follows::
    
  % cd
  % mkdir etc; cd etc
  (Download a source file and mv it to ~/etc)
  % tar xvf boost_x_yy_z.tar.bz2
  % cd ../
  % mkdir include; cd include
  % ln -s ../etc/boost_x_yy_z/boost .

In this example, we place the boost files in ``$(HOME)/etc`` and create a symbolic link to the ``$(HOME)/boost_x_yy_z/boost`` in ``$(HOME)/include``.
Instead of installing from source, you can install the Boost library with `Homebrew <http://brew.sh>`_ on macOS, or with ``apt-get`` or ``yum`` on Linux.

In the same way, please install the Eigen3 include files as follows::

  % cd
  % mkdir etc; cd etc
  (Download a source file and mv it to ~/etc)
  % tar xvf eigen-eigen-*.tar.bz2 (* is an array of letters and digits)
  % cd ../
  % cd include
  % ln -s ../etc/eigen-eigen-*/Eigen .  


If you have followed the instructions, you will see the following results::

  % pwd
  * /home/tadano/include
  % ls -l
  * total 0
  * lrwxrwxrwx 1 tadano sim00 25 May 17  2017 boost -> ../etc/boost_1_64_0/boost
  * lrwxrwxrwx 1 tadano sim00 38 May 17  2017 Eigen -> ../etc/eigen-eigen-67e894c6cd8f/Eigen/


spglib
++++++

Please install spglib by following the instruction on the `spglib webpage <https://atztogo.github.io/spglib/install.html>`_.
Here, we assume spglib is installed in ``$SPGLIB_ROOT``.

HDF5
++++

ALAMODE requires the HDF5 library. Please install it by following the instruction on the
`HDF5 webpage <https://www.hdfgroup.org/solutions/hdf5/>`_, or via your system package
manager. If HDF5 is installed in a non-standard location, pass its path to CMake via the
``-DHDF5_ROOT`` option.


Step 2. Download source
~~~~~~~~~~~~~~~~~~~~~~~

From the GitHub repository::

  % git clone https://github.com/alamode-code/alamode.git
  % cd alamode
  % git checkout 2.0dev

Use the branch that corresponds to the documentation you are reading. For the
development documentation and the current GitHub workflow, this is ``2.0dev``.

The directory structure assumed in this section is shown below::

   $HOME
    ├── alamode
    │   ├── CMakeLists.txt
    │   ├── alm
    │   │   └── CMakeLists.txt
    │   ├── anphon
    │   │   └── CMakeLists.txt
    │   ├── docs
    │   ├── example
    │   ├── external
    │   ├── include
    │   └── tools
    │       └── CMakeLists.txt
    │
    ├── include
    │   ├── boost
    │   └── Eigen

   $SPGLIB_ROOT
    ├── include
    └── lib

   $HDF5_ROOT
    ├── include
    └── lib

Step 3-1. Build by CMake
~~~~~~~~~~~~~~~~~~~~~~~~

CMake is the recommended (and supported) way to build ALAMODE. To use this approach,
you need to install cmake version 3.17 or later (and below 4.0).

To generate Makefiles with CMake, issue the following commands::

  % cd alamode
  % mkdir _build; cd _build
  % source /path/to/oneapi/setvars.sh
  % cmake -DUSE_MKL_BACKEND=yes -DMKL_INTERFACE=lp64 \
    -DSPGLIB_ROOT=${SPGLIB_ROOT} -DHDF5_ROOT=${HDF5_ROOT} \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_FLAGS="-O2 -xHOST" ..

.. You can use ``-DCMAKE_C_COMPILER`` and ``-DCMAKE_CXX_COMPILER`` options to specify the compilers to build ALAMODE binaries. 
.. If these options are not given, cmake will detect the compilers automatically by referencing the environmental variables 
.. ``${CC}`` and ``${CXX}``. 

.. The cmake options for popular compilers are shown below:

.. **Intel compiler**::

..   $ cmake -DSPGLIB_PATH=/path/to/spglib/installdir -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS="-O2 -xHOST" ..

.. **gcc**::

..   $ cmake -DSPGLIB_PATH=/path/to/spglib/installdir -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS="-O2 -march=native" ..

.. **gcc (Installed via Homebrew on macos)**::

..   $ cmake -DSPGLIB_PATH=/path/to/spglib/installdir -DCMAKE_C_COMPILER=gcc-10 -DCMAKE_CXX_COMPILER=g++-10 -DCMAKE_CXX_FLAGS="-O2 -march=native" ..

.. You may need to change ``gcc-10`` (``g++-10``) to ``gcc-9`` (``g++-9``) or older versions installed on your system.

.. note::

    If cmake cannot find Boost, Eigen3, or HDF5 automatically, you need to tell where these libraries
    are installed by using the ``-DBOOST_INCLUDE``, ``-DEIGEN3_INCLUDE``, and ``-DHDF5_ROOT`` options.
    For example, if the directory structure of Step 2 is used, the cmake option will be::

        % cmake -DUSE_MKL_BACKEND=yes -DMKL_INTERFACE=lp64 -DSPGLIB_ROOT=${SPGLIB_ROOT} \
          -DHDF5_ROOT=${HDF5_ROOT} -DBOOST_INCLUDE=${HOME}/include -DEIGEN3_INCLUDE=${HOME}/include \
          -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_FLAGS="-O2 -xHOST" ..

After the configuration finishes successfully, please issue
::

  % make -j

to build all binaries in alm/, anphon/, and tools/ subdirectories under the current directory (_build). 
You can specify the binary to build, for example, as
::

  % make alm -j


.. note::

    When using the binaries, it may be necessary to set ``$LD_LIBRARY_PATH`` as
    ::

      % export SPGLIB_ROOT=/path/to/spglib/installdir
      % export HDF5_ROOT=/path/to/hdf5/installdir
      % export LD_LIBRARY_PATH=$SPGLIB_ROOT/lib:$HDF5_ROOT/lib:$LD_LIBRARY_PATH


Step 3-2. Build by Makefile (deprecated)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

    The legacy ``Makefile.{linux,osx,...}`` files under ``alm/``, ``anphon/``, and ``tools/``
    are no longer maintained and are not kept in sync with the current build system. For
    example, they still target the C++11 standard and do not link the now-mandatory HDF5
    library, so they will not produce working binaries without manual editing. Please build
    ALAMODE with CMake as described above.
