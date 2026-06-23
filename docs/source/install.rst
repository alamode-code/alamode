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

No worries! All of these libraries can be installed easily by using conda.

In addition to the above requirements, users have to get and install a first-principles package 
(such as VASP_, QUANTUM-ESPRESSO_, OpenMX_, or xTAPP_) or another force field package (such as
LAMMPS_) by themselves in order to compute harmonic and anharmonic force constants.

.. _VASP: http://www.vasp.at
.. _OpenMX: http://www.openmx-square.org
.. _QUANTUM-ESPRESSO: http://www.quantum-espresso.org
.. _xTAPP: http://frodo.wpi-aimr.tohoku.ac.jp/xtapp/index.html
.. _LAMMPS: http://lammps.sandia.gov


Optional requirements
~~~~~~~~~~~~~~~~~~~~~~~

* Python (>= 3.x), Numpy, and Matplotlib
* XcrySDen_ or VMD_

We provide some small scripts written in Python for visualizing phonon dispersion relations, phonon DOSs, etc.
To use these scripts, one needs to install the above Python packages.
Additionally, XcrySDen is necessary to visualize the normal mode directions and animate the normal mode.
VMD may be more useful to make an animation, but it may be replaced by any other visualization software which supports the XYZ format.

.. _XcrySDen: http://www.xcrysden.org
.. _VMD: http://www.ks.uiuc.edu/Research/vmd/


Install using conda (recommended for non-experts)
-------------------------------------------------

This option is recommended for users who want a reproducible build with minimal
manual setup. It follows the same conda environment used by the GitHub Actions
tests. If you need the best performance for large production runs, especially
large sparse least-squares fits, see the :ref:`native installation <install_native>`
section and the :ref:`linear-algebra backend notes <install_backends>`.


Step 1. Preparing build tools by conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At first, prepare a conda environment named ``alamode``.
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

Download source files from GitHub repository::

  % git clone https://github.com/alamode-team/alamode.git
  % cd alamode
  % git checkout 2.0dev
  % conda env update -n alamode -f etc/alamode-environment.yml
  % conda activate alamode

Use the branch that corresponds to the documentation you are reading. For the
development documentation and the current GitHub workflow, this is ``2.0dev``.
The directory structure supposed in this document is shown as below::

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

The recommended priority is:

1. **Intel MKL / PARDISO** on Linux or Intel oneAPI systems.
2. **SuiteSparse / SPQR + CHOLMOD** when SuiteSparse is available.
3. **Apple Accelerate** on macOS when MKL is unavailable.

.. raw:: html

   <details>
   <summary><strong>Intel MKL / PARDISO</strong></summary>
   <p><code>-DUSE_MKL_BACKEND=yes</code> enables Intel MKL and uses
   <strong>PARDISO</strong> (<code>Eigen::PardisoLDLT</code>) for the sparse
   KKT system used by the numerically constrained sparse OLS path
   (<code>LMODEL = ols</code>, <code>SPARSE = 1</code>,
   <code>ICONST = 1, 2, 3, 4</code>). This is the preferred backend for large
   constrained sparse fits. ALAMODE currently supports only the LP64 MKL
   interface.</p>
   <p>Example with Intel oneAPI compilers:</p>
   <pre><code>% source /path/to/oneapi/setvars.sh
   % mkdir _build-mkl; cd _build-mkl
   % cmake -DCMAKE_BUILD_TYPE=Release \
           -DUSE_MKL_BACKEND=yes -DMKL_INTERFACE=lp64 \
           -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
           -DCMAKE_CXX_FLAGS="-O2 -xHOST" ..
   % make -j</code></pre>
   <p>If CMake cannot find MKL automatically, make sure the oneAPI environment
   has been loaded, or add MKL's CMake package location to
   <code>CMAKE_PREFIX_PATH</code>. The MKL backend also routes Eigen dense
   products through MKL (<code>EIGEN_USE_MKL_ALL</code>).
   <code>-DUSE_EIGEN_BLAS=yes</code> is ignored when the MKL backend is
   enabled.</p>
   </details>

   <details>
   <summary><strong>SuiteSparse / SPQR and CHOLMOD</strong></summary>
   <p><code>-DUSE_SUITESPARSE_BACKEND=yes</code> enables SuiteSparse support.
   This is an add-on, not an exclusive BLAS backend: it can be combined with the
   default BLAS, MKL, or Accelerate. It makes two additional
   <code>SPARSESOLVER</code> values available for the unconstrained or
   algebraically constrained sparse OLS path: <code>SuiteSparseQR</code> and
   <code>CHOLMOD</code>. It also makes SuiteSparseQR the preferred QR fallback
   in the numerically constrained sparse KKT path after any MKL or Accelerate
   LDLT backend.</p>
   <p>The implementation calls the SuiteSparseQR C interface directly for SPQR,
   rather than Eigen's <code>SPQR</code> wrapper, and uses Eigen's CHOLMOD
   wrapper for CHOLMOD. Point CMake to the SuiteSparse installation prefix, i.e.
   the directory that contains <code>lib/cmake/SPQR</code> and
   <code>lib/cmake/CHOLMOD</code>.</p>
   <p>Example:</p>
   <pre><code>% mkdir _build-suitesparse; cd _build-suitesparse
   % cmake -DCMAKE_BUILD_TYPE=Release \
           -DUSE_SUITESPARSE_BACKEND=yes \
           -DSUITESPARSE_ROOT=/path/to/suitesparse/prefix ..
   % make -j</code></pre>
   <p>When SuiteSparse is installed by a package manager in a standard prefix,
   the <code>-DSUITESPARSE_ROOT</code> option may not be necessary. For large
   constrained sparse fits, SuiteSparseQR is usually more robust than Eigen's
   built-in serial <code>SparseLU</code>/<code>SparseQR</code> fallback, but
   MKL/PARDISO remains the first choice when available.</p>
   </details>

   <details>
   <summary><strong>Apple Accelerate</strong></summary>
   <p><code>-DUSE_ACCEL_BACKEND=yes</code> uses Apple's Accelerate framework on
   macOS. It enables <code>Eigen::AccelerateLDLT</code> for the same
   numerically constrained sparse KKT path where MKL/PARDISO is used, and routes
   Eigen dense products through Accelerate. This is the recommended native
   backend on Apple systems where MKL is not available.</p>
   <p>Example:</p>
   <pre><code>% mkdir _build-accel; cd _build-accel
   % cmake -DCMAKE_BUILD_TYPE=Release -DUSE_ACCEL_BACKEND=yes ..
   % make -j</code></pre>
   <p>This backend requires an Eigen version that provides
   <code>Eigen/AccelerateSupport</code> (Eigen 3.4.90 or later).</p>
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
Instead of installing from source, you can install the Boost library with `Homebrew <http://brew.sh>`_ on macOS and the ``apt-get`` or ``yum`` command on unix.

In the same way, please install the Eigen3 include files as follows::

  % cd
  % mkdir etc; cd etc
  (Download a source file and mv it to ~/etc)
  % tar xvf eigen-eigen-*.tar.bz2 (* is an array of letters and digits)
  % cd ../
  % cd include
  % ln -s ../etc/eigen-eigen-*/Eigen .  


If you have followed the instruction, you will see the following results::

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

  % git clone https://github.com/alamode-team/alamode.git
  % cd alamode
  % git checkout 2.0dev

Use the branch that corresponds to the documentation you are reading. For the
development documentation and the current GitHub workflow, this is ``2.0dev``.

The directory structure supposed in this section is shown as below::

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

To build Makefiles with CMake, please issue the following commands::

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
