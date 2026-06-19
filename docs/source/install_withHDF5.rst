Installation
============

Requirement
-----------

Mandatory requirements
~~~~~~~~~~~~~~~~~~~~~~

* C++ compiler (Intel compiler is recommended.)
* LAPACK library
* MPI library (OpenMPI, MPICH2, IntelMPI, etc.)
* `Boost C++ library <http://www.boost.org>`_ (version >= 1.66)
* FFTW3 library (not necessary when Intel MKL is available)
* `Eigen3 library <http://eigen.tuxfamily.org/>`_
* `spglib <https://atztogo.github.io/spglib/>`_

No worries! All of these libraries can be installed easily by using conda.

In addition to the above requirements, users have to get and install a first-principles package 
(such as VASP_, QUANTUM-ESPRESSO_, OpenMX_, or xTAPP_) or another force field package (such as
LAMMPS_) by themselves in order to compute harmonic and anharmonic force constants.

.. _VASP : http://www.vasp.at
.. _OpenMX : http://www.openmx-square.org
.. _QUANTUM-ESPRESSO : http://www.quantum-espresso.org
.. _xTAPP : http://frodo.wpi-aimr.tohoku.ac.jp/xtapp/index.html
.. _LAMMPS : http://lammps.sandia.gov


Optional requirements
~~~~~~~~~~~~~~~~~~~~~

* Python (> 2.6), Numpy, and Matplotlib
* XcrySDen_ or VMD_
* HDF5 library (as of alamode >= 1.2.0)

We provide some small scripts written in Python for visualizing phonon dispersion relations, phonon DOSs, etc.
To use these scripts, one need to install the above Python packages.
Additionally, XcrySDen is necessary to visualize the normal mode directions and animate the normal mode.
VMD may be more useful to make an animation, but it may be replaced by any other visualization software which supports the XYZ format.

.. _XcrySDen : http://www.xcrysden.org
.. _VMD : http://www.ks.uiuc.edu/Research/vmd/


Install using conda (recommended for non-experts)
-------------------------------------------------

.. highlight:: bash

This option is recommended for all users who want to build working binaries. 
If you want to build highly-optimized binaries using the Intel compiler and other optimized libraries, 
you will need to change the cmake settings below. 


Step 1. Preparing build tools by conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At first, it is recommended `to prepare a conda environment
<https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_ by::

   % conda create --name alamode -c conda-forge python=3.7
   % conda activate alamode

Here the name of the conda environment is chosen ``alamode``. The detailed
instruction about the conda environment is found `here
<https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.
For linux and macOS, compilers provided by conda are `different
<https://conda.io/docs/user-guide/tasks/build-packages/compiler-tools.html>`_.
Then to build binaries on linux or macOS, the conda packages need to be installed by

For linux
+++++++++
::

   % conda install -c conda-forge gcc_linux-64 gxx_linux-64 openmpi h5py scipy numpy boost eigen cmake spglib ipython fftw

For macOS
+++++++++
::

   % conda install -c conda-forge clang_osx-64 clangxx_osx-64 openmpi llvm-openmp cmake boost eigen numpy scipy h5py spglib ipython fftw


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

If you want to bulid all binaries (**alm**, **anphon**, and the others), please use ``CMakeLists.txt`` in the ``$HOME/alamode`` directory.
::

  % pwd
  * $HOME/alamode
  % mkdir _build; cd _build
  % cmake -DUSE_MKL_FFT=no ..

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
      
      % cmake -DUSE_MKL_FFT=no -DSPGLIB_ROOT=$CONDA_PREFIX ..
    Also, when using the binaries, it may be necessary to set ``$LD_LIBRARY_PATH`` as
    ::

      % export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH
          


Install using native environment (optional for experts)
-------------------------------------------------------

If you are familier with unix OS and you want to use the Intel compiler, 
please follow the instruction below.
Here, the Intel C++ compiler and the Intel MKL, including the FFTW3 wrapper, will be used for the demonstration.


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
Instead of installing from source, you can install the Boost library with `Homebrew <http://brew.sh>`_ on macOS and ``apt-get`` or ``yum`` command on unix..

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

FFTW
+++++

If you use the MKL wrapper of FFT, this step can be skipped. 
If you want to use the native FFTW library, please follow the instruction on the `FFTW webpage <http://www.fftw.org>`_.

HDF5 (optional)
+++++++++++++++

Please install the HDF5 library following the instruction on the `HDF5 webpage <https://www.hdfgroup.org/downloads/hdf5/source-code/>`_.
If you don't need to activate the HDF5 format support in ALAMODE, please skip this.

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

   $HDF5_ROOT (optional)
    ├── include
    └── lib

   $FFTW3_ROOT (optional)
    ├── include
    └── lib

Step 3-1. Build by CMake 
~~~~~~~~~~~~~~~~~~~~~~~~

Building by CMake is recommended as of version 1.2.0 of alamode. To use this approach, 
you need to install cmake version 3.1 or later.

To build Makefiles with CMake, please issue the following commands::

  % cd alamode
  % mkdir _build; cd _build
  % cmake -DUSE_MKL_FFT=yes -DSPGLIB_ROOT=${SPGLIB_ROOT} \ 
    -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS="-O2 -xHOST" ..

If you want to activate the HDF5 support, please add ``-DWITH_HDF5_SUPPORT=yes`` in the cmake option.

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

    If cmake cannot find Boost, Eigen3, HDF5, or FFTW automatically, you need to tell where these libraries are installed by
    are installed by using ``-DBOOST_INCLUDE``, ``-DEIGEN3_INCLUDE``, ``-DHDF5_ROOT``, and ``-DFFTW3_ROOT`` options.
    For example, if the directory structure of Step 2 is used, the cmake option will be::
    
        % cmake -DUSE_MKL_FFT=yes -DWITH_HDF5_SUPPORT=yes -DSPGLIB_ROOT=${SPGLIB_ROOT} \
          -DBOOST_INCLUDE=${HOME}/include -DEIGEN3_INCLUDE=${HOME}/include -DHDF5_ROOT=${HDF5_ROOT} \
          -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS="-O2 -xHOST" .. 

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
      % export HDF5_ROOT=/path/to/hdf5/installdir (when compiled with -DWITH_HDF5_SUPPORT=yes in cmake)
      % export LD_LIBRARY_PATH=$SPGLIB_ROOT/lib:$HDF5_ROOT/lib:$LD_LIBRARY_PATH


Step 3-2. Build by Makefile 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of using CMake, you can build each binary of ALAMODE by using the Makefile.{linux,osx,..}. 

In directories ``alm/``, ``anphon/``, and ``tools``, we provide sample Makefiles for Linux (Intel compiler) and Mac OSX (gcc, clang).
Please copy either of them, edit the options appropriately, and issue ``make`` command as follows::

    % export SPGLIB_ROOT=/path/to/spglib/installdir
    % export HDF5_ROOT=/path/to/hdf5/installdir

    % cd alm/
    % cp Makefile.linux Makefile
    (Edit Makefile here)
    % make -j

    % cd ../anphon/
    % cp Makefile.linux Makefile
    (Edit Makefile here)
    % make -j

    % cd ../tools/
    % cp Makefile.linux Makefile
    (Edit Makefile here)
    % make -j

An example of the Makefiles is shown below:

.. literalinclude:: ../../alm/Makefile.linux
    :caption: **ALM Makefile.linux**
    :language: makefile
    :linenos:
    :lines: 7-17

The default options are expected to work with modern Intel compilers.

.. note::

    When using the binaries, it may be necessary to set ``$LD_LIBRARY_PATH`` as
    ::
      % export SPGLIB_ROOT=/path/to/spglib/installdir
      % export HDF5_ROOT=/path/to/hdf5/installdir (when compiled with -D_HDF5 in CXXFLAGS)
      % export LD_LIBRARY_PATH=$SPGLIB_ROOT/lib:$HDF5_ROOT/lib:$LD_LIBRARY_PATH
