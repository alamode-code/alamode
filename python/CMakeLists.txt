cmake_minimum_required(VERSION 3.17)
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()
project(alamode_python CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MACOSX_RPATH 1)

# Inside a conda/pixi environment, search that prefix first (same hook as the
# top-level CMakeLists.txt) so spglib/Boost/Eigen/HDF5/MPI are found without
# passing -C cmake.define.*_ROOT options to pip.
if(DEFINED ENV{CONDA_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    if(NOT CMAKE_BUILD_RPATH)
        set(CMAKE_BUILD_RPATH "$ENV{CONDA_PREFIX}/lib")
    endif()
    if(NOT CMAKE_INSTALL_RPATH)
        set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
    endif()
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

# ---- nanobind (provided by the build requirement; scikit-build-core exposes its CMake dir) ----
# STABLE_ABI (below) builds against the limited API on Python >= 3.12, which needs Development.SABIModule.
# With wheel.py-api = "cp312" in pyproject.toml this gives one abi3 wheel for 3.12 and later. Older
# Pythons get a regular build.
find_package(Python 3.9 COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# ---- C++ libraries ----
# The modules link the static libraries the CLI build defines: almcxx_static (../alm) and
# anphoncxx_static (../anphon). Dependency detection is theirs, so the location options are the CLI ones:
# SPGLIB_ROOT, BOOST_ROOT / BOOST_INCLUDE, EIGEN3_ROOT / EIGEN3_INCLUDE, HDF5_ROOT, USE_ACCEL_BACKEND
# (alm only; anphon picks its LAPACK through FindLAPACK / BLA_VENDOR), USE_EIGEN_BLAS, ...
# EXCLUDE_FROM_ALL builds only what the modules need (no CLIs, no unit tests) and keeps the
# subdirectories' install rules out of the wheel.
if(EIGEN3_INCLUDE_DIR AND NOT EIGEN3_INCLUDE) # spelling accepted by earlier versions of this file
    set(EIGEN3_INCLUDE ${EIGEN3_INCLUDE_DIR})
endif()

# Boost: alm/ and anphon/ need only the headers (>= 1.66) and fall back to BOOST_INCLUDE when their
# find_package(Boost) fails. Header-only installs (e.g. conda-forge "boost" without BoostConfig.cmake) are
# located here by boost/version.hpp and handed down as BOOST_INCLUDE.
if(NOT BOOST_INCLUDE)
    find_package(Boost 1.66.0 QUIET CONFIG)
    if(NOT Boost_FOUND)
        find_path(BOOST_VERSION_HPP_DIR boost/version.hpp HINTS ${BOOST_ROOT} ENV BOOST_ROOT PATH_SUFFIXES include)
        if(BOOST_VERSION_HPP_DIR)
            file(STRINGS "${BOOST_VERSION_HPP_DIR}/boost/version.hpp" _boost_version_line
                 REGEX "^#define BOOST_VERSION [0-9]+")
            if(_boost_version_line MATCHES "BOOST_VERSION ([0-9]+)")
                math(EXPR _boost_major "${CMAKE_MATCH_1} / 100000")
                math(EXPR _boost_minor "${CMAKE_MATCH_1} / 100 % 1000")
                if("${_boost_major}.${_boost_minor}" VERSION_LESS 1.66)
                    message(FATAL_ERROR "Boost ${_boost_major}.${_boost_minor} found in ${BOOST_VERSION_HPP_DIR}; "
                                        "ALAMODE requires Boost >= 1.66")
                endif()
                message(STATUS "Boost headers ${_boost_major}.${_boost_minor} in ${BOOST_VERSION_HPP_DIR}")
            endif()
            set(BOOST_INCLUDE "${BOOST_VERSION_HPP_DIR}" CACHE PATH "Boost include directory")
        endif()
    endif()
endif()

set(ALM_BUILD_STATIC_LIB ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../alm alm EXCLUDE_FROM_ALL)

# anphon needs MPI on top of what ALM needs (HDF5, LAPACK). ON by default so that a missing MPI fails the
# configure step instead of silently producing a package without a working `anphon`.
option(ALAMODE_PYTHON_ANPHON "Build the anphon module _anphon (needs MPI)" ON)
if(ALAMODE_PYTHON_ANPHON)
    find_package(MPI COMPONENTS CXX)
    if(NOT MPI_CXX_FOUND)
        message(FATAL_ERROR "MPI (C++) not found, which the anphon module needs. Install an MPI library or "
                            "build ALM only with -DALAMODE_PYTHON_ANPHON=OFF "
                            "(pip: -C cmake.define.ALAMODE_PYTHON_ANPHON=OFF).")
    endif()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../anphon anphon EXCLUDE_FROM_ALL)
endif()

# NB_STATIC: link the nanobind runtime statically into each module.
nanobind_add_module(_alm NB_STATIC STABLE_ABI src/_alm.cpp)
target_link_libraries(_alm PRIVATE almcxx_static)
install(TARGETS _alm LIBRARY DESTINATION alm)

if(ALAMODE_PYTHON_ANPHON)
    nanobind_add_module(_anphon NB_STATIC STABLE_ABI src/_anphon.cpp)
    target_link_libraries(_anphon PRIVATE anphoncxx_static)
    install(TARGETS _anphon LIBRARY DESTINATION anphon)
endif()
