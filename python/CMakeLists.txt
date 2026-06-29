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
find_package(Boost 1.66.0)
if (Boost_FOUND)
    if (TARGET Boost::boost)
        set(boost_target Boost::boost)
    elseif (TARGET Boost::headers)
        set(boost_target Boost::headers)
    else()
        set(boost_includes ${Boost_INCLUDE_DIRS})
    endif()
elseif (BOOST_INCLUDE)
    set(boost_includes ${BOOST_INCLUDE})
else ()
    message(FATAL_ERROR "Boost (>=1.66) not found; pass -DBOOST_INCLUDE=... or -DBOOST_ROOT=...")
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
