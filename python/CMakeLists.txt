cmake_minimum_required(VERSION 3.17)
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()
project(alm_python CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MACOSX_RPATH 1)

# Inside a conda/pixi environment, search that prefix first (same hook as the
# top-level CMakeLists.txt) so spglib/Boost/Eigen/HDF5 are found without
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

set(ALM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../alm)
set(ALM_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)
set(ALM_EXT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../external)

# ---- nanobind (provided by the build requirement; scikit-build-core exposes its CMake dir) ----
# For an abi3/limited-API (STABLE_ABI) build on Python >= 3.12, also:
#   (1) add OPTIONAL_COMPONENTS Development.SABIModule to find_package(Python ...)
#   (2) pass STABLE_ABI to nanobind_add_module(...) below
#   (3) set  [tool.scikit-build.wheel] py-api = "cp312"  in pyproject.toml
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# ---- dependencies (mirror ../alm/CMakeLists.txt) ----
if (SPGLIB_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${SPGLIB_ROOT}")
endif()
find_path(SPGLIB_INCLUDE_DIR spglib.h HINTS ${SPGLIB_ROOT} PATH_SUFFIXES include)
find_library(SPGLIB_LIBRARY NAMES symspg spglib HINTS ${SPGLIB_ROOT} PATH_SUFFIXES lib lib64)
if (SPGLIB_INCLUDE_DIR AND SPGLIB_LIBRARY)
    set(spglib_include ${SPGLIB_INCLUDE_DIR})
    set(spglib ${SPGLIB_LIBRARY})
else ()
    set(spglib "-lsymspg")
endif ()

find_package(OpenMP REQUIRED COMPONENTS CXX)

if (BOOST_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${BOOST_ROOT}")
endif()
# ALM only needs the Boost headers (boost::sort, >= 1.66). With CMP0167 NEW,
# find_package(Boost) is config-mode only and needs BoostConfig.cmake, which
# header-only installs do not ship (e.g. conda-forge "boost"/"libboost-headers"
# without libboost-devel). Fall back to locating boost/version.hpp directly.
find_package(Boost 1.66.0 QUIET)
if (Boost_FOUND)
    if (TARGET Boost::boost)
        set(boost_target Boost::boost)
    elseif (TARGET Boost::headers)
        set(boost_target Boost::headers)
    else()
        set(boost_includes ${Boost_INCLUDE_DIRS})
    endif()
    message(STATUS "Found Boost ${Boost_VERSION} (config: ${Boost_DIR})")
else ()
    if (BOOST_INCLUDE)
        set(boost_includes ${BOOST_INCLUDE})
    else ()
        find_path(BOOST_INCLUDE_DIR boost/version.hpp
            HINTS ${BOOST_ROOT} ENV BOOST_ROOT
            PATH_SUFFIXES include)
        if (NOT BOOST_INCLUDE_DIR)
            message(FATAL_ERROR "Boost (>=1.66) not found; pass -DBOOST_INCLUDE=... or -DBOOST_ROOT=...")
        endif()
        set(boost_includes ${BOOST_INCLUDE_DIR})
    endif ()
    list(GET boost_includes 0 _boost_inc0)
    if (NOT EXISTS "${_boost_inc0}/boost/version.hpp")
        message(FATAL_ERROR "boost/version.hpp not found under ${_boost_inc0}; check BOOST_INCLUDE / BOOST_ROOT")
    endif()
    file(STRINGS "${_boost_inc0}/boost/version.hpp" _boost_version_line
        REGEX "^#define BOOST_VERSION [0-9]+")
    if (_boost_version_line MATCHES "BOOST_VERSION ([0-9]+)")
        math(EXPR _boost_major "${CMAKE_MATCH_1} / 100000")
        math(EXPR _boost_minor "${CMAKE_MATCH_1} / 100 % 1000")
        set(_boost_version "${_boost_major}.${_boost_minor}")
        if (_boost_version VERSION_LESS 1.66)
            message(FATAL_ERROR "Boost ${_boost_version} found in ${_boost_inc0}; ALM requires Boost >= 1.66")
        endif()
        message(STATUS "Found Boost headers ${_boost_version} in ${_boost_inc0} (no BoostConfig.cmake)")
    else ()
        message(WARNING "Could not determine the Boost version from ${_boost_inc0}/boost/version.hpp")
    endif ()
endif ()

if (EIGEN3_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${EIGEN3_ROOT}")
endif()
find_package(Eigen3 QUIET)
if (Eigen3_FOUND AND TARGET Eigen3::Eigen)
    set(eigen_target Eigen3::Eigen)
elseif (EIGEN3_INCLUDE)
    set(eigen_includes ${EIGEN3_INCLUDE})
elseif (EIGEN3_INCLUDE_DIR)
    set(eigen_includes ${EIGEN3_INCLUDE_DIR})
else ()
    message(FATAL_ERROR "Eigen3 not found; pass -DEIGEN3_INCLUDE=... or -DEIGEN3_ROOT=...")
endif ()

if (HDF5_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${HDF5_ROOT}")
endif()
find_package(HDF5 COMPONENTS CXX)
if (HDF5_FOUND)
    set(hdf5library ${HDF5_LIBRARIES})
    if (TARGET HDF5::HDF5)
        set(hdf5_target HDF5::HDF5)
    endif()
    set(hdf5_include ${HDF5_INCLUDE_DIRS})
elseif (HDF5_ROOT)
    set(hdf5_include "${HDF5_ROOT}/include")
    set(hdf5library "-L${HDF5_ROOT}/lib -lhdf5_cpp -lhdf5")
else ()
    message(FATAL_ERROR "HDF5 (CXX) not found; pass -DHDF5_ROOT=...")
endif ()

# BLAS/LAPACK backend (generic; Apple Accelerate is found here on macOS).
option(USE_ACCEL_BACKEND "Enable Apple Accelerate" OFF)
if (USE_ACCEL_BACKEND)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    add_compile_definitions(USE_ACCEL_BACKEND EIGEN_USE_ACCELERATE)
    set(BACKEND_LIBS ${ACCELERATE_FRAMEWORK})
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    if (TARGET BLAS::BLAS)
        set(blas_target BLAS::BLAS)
    else()
        set(blas_target ${BLAS_LIBRARIES})
    endif()
    if (TARGET LAPACK::LAPACK)
        set(lapack_target LAPACK::LAPACK)
    else()
        set(lapack_target ${LAPACK_LIBRARIES})
    endif()
    set(BACKEND_LIBS ${blas_target} ${lapack_target})
endif()

# ---- ALM library sources (same list as ../alm/CMakeLists.txt, minus the CUI driver) ----
set(ALM_SOURCES
    ${ALM_SRC_DIR}/alm.cpp
    ${ALM_SRC_DIR}/cluster.cpp
    ${ALM_SRC_DIR}/constraint.cpp
    ${ALM_SRC_DIR}/fcs.cpp
    ${ALM_SRC_DIR}/files.cpp
    ${ALM_SRC_DIR}/input_parser.cpp
    ${ALM_SRC_DIR}/input_setter.cpp
    ${ALM_SRC_DIR}/optimize.cpp
    ${ALM_SRC_DIR}/patterndisp.cpp
    ${ALM_SRC_DIR}/rref.cpp
    ${ALM_SRC_DIR}/symmetry.cpp
    ${ALM_SRC_DIR}/system.cpp
    ${ALM_SRC_DIR}/timer.cpp
    ${ALM_SRC_DIR}/writer.cpp
    ${ALM_SRC_DIR}/smith.cpp
    ${ALM_SRC_DIR}/svd.cpp
    ${ALM_SRC_DIR}/least_squares.cpp
    ${ALM_SRC_DIR}/accelerate_solver.cpp)

# NB_STATIC: link the nanobind runtime statically into this single module.
# (Add STABLE_ABI to also produce an abi3/limited-API wheel on Python >= 3.12.)
nanobind_add_module(_alm NB_STATIC src/_alm.cpp ${ALM_SOURCES})

target_compile_definitions(_alm PRIVATE _HDF5)
target_include_directories(_alm PRIVATE
    ${ALM_SRC_DIR} ${ALM_INC_DIR} ${ALM_EXT_DIR}
    ${spglib_include} ${boost_includes} ${eigen_includes} ${hdf5_include})
target_link_libraries(_alm PRIVATE
    ${boost_target} ${eigen_target} ${Boost_LIBRARIES} ${BACKEND_LIBS}
    ${spglib} OpenMP::OpenMP_CXX ${hdf5_target} ${hdf5library})

# Install the compiled module into the python package directory.
install(TARGETS _alm LIBRARY DESTINATION alm)
