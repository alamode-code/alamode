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

This option is recommended for all users who want to build working binaries. 
If you want to build highly-optimized binaries using the Intel compiler and other optimized libraries, 
you will need to change the cmake settings below. 


Step 1. Preparing build tools by conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At first, it is recommended `to prepare a conda environment
<https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_ by::

   % conda create --name alamode -c conda-forge python=3
   % conda activate alamode

Here the name of the conda environment is chosen ``alamode``. The detailed
instruction about the conda environment is found `here
<https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.
For linux and macOS, we recommend using the conda `compiler tools
<https://conda.io/docs/user-guide/tasks/build-packages/compiler-tools.html>`_.
To build binaries on linux or macOS, the conda packages need to be installed by

::

   % conda install -c conda-forge compilers openmpi boost eigen cmake spglib hdf5 scipy numpy h5py ipython


Step 2. Download source 
~~~~~~~~~~~~~~~~~~~~~~~

Download source files from GitHub repository::

  % git clone https://github.com/ttadano/alamode.git
  % cd alamode
  % git checkout develop

If git command doesn't exist in your system, it is also obtained from conda by ``conda install git``.
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


.. _install_backends:

Optional linear-algebra backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, ALAMODE links against the LAPACK/BLAS libraries detected by CMake
(the conda OpenBLAS is detected automatically). The following options select an
alternative backend for the dense/sparse linear algebra:

* ``-DUSE_MKL_BACKEND=yes`` : Use the Intel MKL (PARDISO) backend. Only the LP64
  interface is supported, so it is combined with ``-DMKL_INTERFACE=lp64`` (the default).
  This enables Intel MKL **PARDISO** (``Eigen::PardisoLDLT``) for factorizing the sparse
  KKT system that arises in the constrained sparse least-squares fit
  (``LMODEL = ols`` with ``SPARSE = 1`` and numerically imposed constraints
  ``ICONST = 1, 2, 3``). PARDISO is a multi-threaded sparse direct solver and is the
  fastest and most memory-scalable choice for such large constrained sparse fits; it is
  recommended whenever Intel MKL is available. It does not affect the dense solvers or
  the Eigen ``SPARSESOLVER`` options.
* ``-DUSE_ACCEL_BACKEND=yes`` : Use the Apple Accelerate framework (macOS only). This uses
  ``Eigen::AccelerateLDLT`` for the same sparse KKT factorization as the PARDISO backend
  above, and is the recommended choice on Apple silicon where MKL is unavailable.
* ``-DUSE_SUITESPARSE_BACKEND=yes`` : Enable the `SuiteSparse <https://people.engr.tamu.edu/davis/suitesparse.html>`_
  sparse solvers. Unlike the MKL/Accelerate options above this is *not* an exclusive BLAS backend;
  it only adds two extra sparse-solver choices and therefore composes with whichever BLAS backend is
  selected. It makes **SuiteSparseQR** (a multithreaded, rank-revealing multifrontal sparse QR) and
  **CHOLMOD** (supernodal sparse Cholesky) available as ``SPARSESOLVER`` options for the
  unconstrained / algebraically constrained sparse fit, and, for the numerically constrained
  (``ICONST = 1, 2, 3``) KKT path, makes the multithreaded SuiteSparseQR the **preferred** direct
  solver (tried before Eigen's serial ``SparseLU``, after any ``LDLT`` backend). Point CMake at the
  installation prefix (the directory that contains ``lib/cmake/SPQR``) with
  ``-DSUITESPARSE_ROOT=<prefix>``. See the :ref:`SPARSESOLVER <alm_sparsesolver>` tag for when to
  prefer each solver.

  .. note::

     QR fill-in on the symmetric-indefinite KKT matrix is heavy only when the constraint matrix has
     dense, (near-)rank-deficient rows; with the rank-revealing reduction of the constraint matrix
     (full row rank, well scaled) the KKT stays sparse and the multithreaded SuiteSparseQR is fast and
     scalable. A symmetric-indefinite ``LDLT`` backend (``-DUSE_MKL_BACKEND`` / ``-DUSE_ACCEL_BACKEND``)
     remains the most efficient option for very large constrained fits when available;
     ``-DUSE_ACCEL_BACKEND`` additionally requires Eigen >= 3.4.90 (the version that introduced
     ``Eigen/AccelerateSupport``).
* ``-DUSE_EIGEN_BLAS=yes`` : Route Eigen's dense products through the detected BLAS
  (``EIGEN_USE_BLAS``). The vendor can be chosen with, e.g., ``-DBLA_VENDOR=OpenBLAS``.

When none of ``-DUSE_MKL_BACKEND``, ``-DUSE_ACCEL_BACKEND``, nor ``-DUSE_SUITESPARSE_BACKEND`` is
given, the constrained sparse KKT system is factorized with Eigen's built-in ``SparseLU`` (serial),
which is portable but slower for large problems.


Install using native environment (optional for experts)
-------------------------------------------------------

If you are familiar with unix OS and you want to use the Intel compiler,
please follow the instruction below.
Here, the Intel C++ compiler and the Intel MKL backend (``-DUSE_MKL_BACKEND=yes``) will be used for the demonstration.


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

From Sourceforge::

  % (visit https://sourceforge.net/projects/alamode/files/latest/download?source=files to download the latest version source)
  % tar xvzf alamode-x.y.z.tar.gz
  % cd alamode-x.y.z

From GitHub repository::

  % git clone https://github.com/ttadano/alamode.git
  % cd alamode
  % git checkout develop

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
  % cmake -DUSE_MKL_BACKEND=yes -DMKL_INTERFACE=lp64 -DSPGLIB_ROOT=${SPGLIB_ROOT} \
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
          -DBOOST_INCLUDE=${HOME}/include -DEIGEN3_INCLUDE=${HOME}/include \
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
      % export LD_LIBRARY_PATH=$SPGLIB_ROOT/lib:$LD_LIBRARY_PATH


Step 3-2. Build by Makefile (deprecated)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

    The legacy ``Makefile.{linux,osx,...}`` files under ``alm/``, ``anphon/``, and ``tools/``
    are no longer maintained and are not kept in sync with the current build system. For
    example, they still target the C++11 standard and do not link the now-mandatory HDF5
    library, so they will not produce working binaries without manual editing. Please build
    ALAMODE with CMake as described above.
