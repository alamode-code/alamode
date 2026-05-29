if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()
cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_MACOSX_RPATH 1)

project(anphon)

# Version numbers
file(READ ${PROJECT_SOURCE_DIR}/../include/version.h version_file)
string(REGEX MATCH "ALAMODE_VERSION = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" alm_version ${version_file})
set(alm_version ${CMAKE_MATCH_1})
MESSAGE("Setting up for ANPHON code (ALAMODE version: ${CMAKE_MATCH_1})")

if (SPGLIB_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${SPGLIB_ROOT}")
endif()

find_path(SPGLIB_INCLUDE_DIR spglib.h
    HINTS ${SPGLIB_ROOT}
    PATH_SUFFIXES include
)
find_library(SPGLIB_LIBRARY
    NAMES symspg spglib
    HINTS ${SPGLIB_ROOT}
    PATH_SUFFIXES lib lib64
)

if (SPGLIB_INCLUDE_DIR AND SPGLIB_LIBRARY)
    set(spglib_include ${SPGLIB_INCLUDE_DIR})
    set(spglib ${SPGLIB_LIBRARY})
else()
    message("SPGLIB not found via hints. Falling back to -lsymspg.")
    set(spglib "-lsymspg")
endif()

set(FFTW_BACKEND "AUTO" CACHE STRING "FFTW backend: AUTO, FFTW3, or MKL")
set_property(CACHE FFTW_BACKEND PROPERTY STRINGS AUTO FFTW3 MKL)

set(fftw_include_dir "")
set(fftw3library "")
set(using_mkl_fftw FALSE)

if (FFTW_BACKEND STREQUAL "MKL" OR FFTW_BACKEND STREQUAL "AUTO")
    set(mkl_root "$ENV{MKLROOT}")
    find_path(MKL_FFTW_INCLUDE_DIR fftw3.h
        HINTS ${mkl_root}
        PATH_SUFFIXES include/fftw include
    )
    find_library(MKL_FFTW_LIBRARY
        NAMES mkl_rt
        HINTS ${mkl_root}
        PATH_SUFFIXES lib lib/intel64 lib64
    )

    if (MKL_FFTW_INCLUDE_DIR AND MKL_FFTW_LIBRARY)
        set(using_mkl_fftw TRUE)
        set(fftw_include_dir ${MKL_FFTW_INCLUDE_DIR})
        set(fftw3library ${MKL_FFTW_LIBRARY})
        message(STATUS "Using MKL FFTW wrappers: ${MKL_FFTW_LIBRARY}")
    elseif(FFTW_BACKEND STREQUAL "MKL")
        message(FATAL_ERROR "FFTW_BACKEND=MKL requested, but MKL FFTW wrappers were not found. Set MKLROOT or provide MKL libs/headers.")
    endif()
endif()

if (NOT using_mkl_fftw)
    if (FFTW3_ROOT)
        find_path(FFTW3_INCLUDE_DIR fftw3.h HINTS "${FFTW3_ROOT}" PATH_SUFFIXES include)
        find_library(FFTW3_LIBRARY NAMES fftw3 HINTS "${FFTW3_ROOT}" PATH_SUFFIXES lib lib64)
    else()
        find_path(FFTW3_INCLUDE_DIR fftw3.h)
        find_library(FFTW3_LIBRARY NAMES fftw3)
    endif()

    if (FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY)
        set(fftw_include_dir ${FFTW3_INCLUDE_DIR})
        set(fftw3library ${FFTW3_LIBRARY})
    else()
        message(FATAL_ERROR "FFTW3 was not found. Set FFTW3_ROOT or use FFTW_BACKEND=MKL with MKLROOT.")
    endif()
endif()

if (fftw_include_dir)
    include_directories(${fftw_include_dir})
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (openmplibrary OpenMP::OpenMP_CXX)
endif()

if (BOOST_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${BOOST_ROOT}")
endif()
find_package(Boost)
if (Boost_FOUND)
    if (TARGET Boost::boost)
        set(boost_target Boost::boost)
    elseif (TARGET Boost::headers)
        set(boost_target Boost::headers)
    else()
        set(boost_includes ${Boost_INCLUDE_DIRS})
    endif()
else()
    if (BOOST_INCLUDE)
        set(boost_includes ${BOOST_INCLUDE})
    else()
        message(FATAL_ERROR "Boost was not found.
        Please specify the location of boost include directories
        via -DBOOST_INCLUDE option.")
    endif()
endif()

if (EIGEN3_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${EIGEN3_ROOT}")
endif()
option(EIGEN3_ROOT "Path to Eigen3 installation" "")
if (EIGEN3_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${EIGEN3_ROOT}")
endif()
find_package(Eigen3)
if (Eigen3_FOUND)
    if (TARGET Eigen3::Eigen)
        set(eigen_target Eigen3::Eigen)
    else()
        set(eigen_includes ${EIGEN3_INCLUDE_DIR})
    endif()
else()
    if (EIGEN3_INCLUDE)
        set(eigen_includes ${EIGEN3_INCLUDE})
    else()
        message(FATAL_ERROR "Eigen3 was not found.
        Please specify the location of boost include directories
        via -DEIGEN3_INCLUDE option.")
    endif()
endif()

if (NOT BLA_VENDOR AND DEFINED ENV{CONDA_PREFIX})
    find_library(CONDA_OPENBLAS_LIB
        NAMES openblas openblas64
        HINTS "$ENV{CONDA_PREFIX}"
        PATH_SUFFIXES lib
    )
    if (CONDA_OPENBLAS_LIB)
        set(BLA_VENDOR OpenBLAS CACHE STRING "BLAS/LAPACK vendor" FORCE)
    endif()
endif()
find_package(LAPACK REQUIRED)
if (TARGET LAPACK::LAPACK)
    set(lapack_target LAPACK::LAPACK)
else()
    set(lapack_target ${LAPACK_LIBRARIES})
endif()
find_package(MPI REQUIRED)
if (TARGET MPI::MPI_CXX)
    set(mpi_target MPI::MPI_CXX)
else()
    set(mpi_target ${MPI_CXX_LIBRARIES})
endif()

if (HDF5_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${HDF5_ROOT}")
endif()
find_package(HDF5 COMPONENTS CXX)
if (HDF5_FOUND)
    set(hdf5library ${HDF5_LIBRARIES})
    if (TARGET HDF5::HDF5)
        set(hdf5_target HDF5::HDF5)
    endif()
else()
    if (HDF5_ROOT)
        set(hdf5_include "${HDF5_ROOT}/include")
        set(hdf5library "-L${HDF5_ROOT}/lib -lhdf5_cpp -lhdf5")
    else()
        message(FATAL_ERROR "HDF5 was not found.
        Please specify the HDF5 install directory
        via -DHDF5_ROOT option.")
    endif()
endif()

# Source code
set(SOURCES ${PROJECT_SOURCE_DIR}/anharmonic_core.cpp
            ${PROJECT_SOURCE_DIR}/conductivity.cpp
            ${PROJECT_SOURCE_DIR}/dielec.cpp
            ${PROJECT_SOURCE_DIR}/diis.cpp
            ${PROJECT_SOURCE_DIR}/dynamical.cpp
            ${PROJECT_SOURCE_DIR}/ewald.cpp
            ${PROJECT_SOURCE_DIR}/fcs_phonon.cpp
            ${PROJECT_SOURCE_DIR}/ifc_derivative.cpp
            ${PROJECT_SOURCE_DIR}/gruneisen.cpp
            ${PROJECT_SOURCE_DIR}/integration.cpp
            ${PROJECT_SOURCE_DIR}/interpolation.cpp
            ${PROJECT_SOURCE_DIR}/isotope.cpp
            ${PROJECT_SOURCE_DIR}/iterativebte.cpp
            ${PROJECT_SOURCE_DIR}/kpoint.cpp
            ${PROJECT_SOURCE_DIR}/main.cpp
            ${PROJECT_SOURCE_DIR}/mode_analysis.cpp
            ${PROJECT_SOURCE_DIR}/mpi_common.cpp
            ${PROJECT_SOURCE_DIR}/parsephon.cpp
            ${PROJECT_SOURCE_DIR}/phonon_dos.cpp
            ${PROJECT_SOURCE_DIR}/phonon_velocity.cpp
            ${PROJECT_SOURCE_DIR}/phonons.cpp
            ${PROJECT_SOURCE_DIR}/optimizers.cpp
            ${PROJECT_SOURCE_DIR}/qha.cpp
            ${PROJECT_SOURCE_DIR}/relaxation.cpp
            ${PROJECT_SOURCE_DIR}/scph.cpp
            ${PROJECT_SOURCE_DIR}/scph_bubble.cpp
            ${PROJECT_SOURCE_DIR}/scph_derivatives.cpp
            ${PROJECT_SOURCE_DIR}/scph_io.cpp
            ${PROJECT_SOURCE_DIR}/scph_qha_common.cpp
            ${PROJECT_SOURCE_DIR}/scph_v3v4_elements.cpp
            ${PROJECT_SOURCE_DIR}/selfenergy.cpp
            ${PROJECT_SOURCE_DIR}/symmetry_core.cpp
            ${PROJECT_SOURCE_DIR}/system.cpp
            ${PROJECT_SOURCE_DIR}/thermodynamics.cpp
            ${PROJECT_SOURCE_DIR}/timer.cpp
            ${PROJECT_SOURCE_DIR}/write_phonons.cpp
            ${PROJECT_SOURCE_DIR}/../external/niggli.c
            )

set(HEADERS ${PROJECT_SOURCE_DIR}/anharmonic_core.h
            ${PROJECT_SOURCE_DIR}/conductivity.h
            ${PROJECT_SOURCE_DIR}/dielec.h
            ${PROJECT_SOURCE_DIR}/diis.h
            ${PROJECT_SOURCE_DIR}/dynamical.h
            ${PROJECT_SOURCE_DIR}/error.h
            ${PROJECT_SOURCE_DIR}/ewald.h
            ${PROJECT_SOURCE_DIR}/fcs_phonon.h
            ${PROJECT_SOURCE_DIR}/ifc_derivative.h
            ${PROJECT_SOURCE_DIR}/gruneisen.h
            ${PROJECT_SOURCE_DIR}/integration.h
            ${PROJECT_SOURCE_DIR}/interpolation.h
            ${PROJECT_SOURCE_DIR}/isotope.h
            ${PROJECT_SOURCE_DIR}/iterativebte.h
            ${PROJECT_SOURCE_DIR}/kpoint.h
            ${PROJECT_SOURCE_DIR}/mode_analysis.h
            ${PROJECT_SOURCE_DIR}/mpi_common.h
            ${PROJECT_SOURCE_DIR}/parsephon.h
            ${PROJECT_SOURCE_DIR}/phonon_dos.h
            ${PROJECT_SOURCE_DIR}/phonon_velocity.h
            ${PROJECT_SOURCE_DIR}/phonons.h
            ${PROJECT_SOURCE_DIR}/pointers.h
            ${PROJECT_SOURCE_DIR}/optimizers.h
            ${PROJECT_SOURCE_DIR}/qha.h
            ${PROJECT_SOURCE_DIR}/relaxation.h
            ${PROJECT_SOURCE_DIR}/scph.h
            ${PROJECT_SOURCE_DIR}/selfenergy.h
            ${PROJECT_SOURCE_DIR}/symmetry_core.h
            ${PROJECT_SOURCE_DIR}/system.h
            ${PROJECT_SOURCE_DIR}/thermodynamics.h
            ${PROJECT_SOURCE_DIR}/timer.h
            ${PROJECT_SOURCE_DIR}/write_phonons.h
            ${PROJECT_SOURCE_DIR}/../include/progress_bar.h
            ../include/lapack_wrapper.h
            ../include/logger.h
            )

message("MPI_C_LIBRARIES = ${MPI_C_LIBRARIES}")
message("MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}")
# Executable
add_executable(anphon ${SOURCES} ${HEADERS})
# Warnings on for all build types. -Wno-sign-compare silences ~1500 benign
# signed/unsigned loop-counter comparisons so that the genuinely useful warnings
# (unused/uninitialized variables, VLAs, deprecated copies, ...) are visible.
# Drop -Wno-sign-compare once the loop counters have been cleaned up.
target_compile_options(anphon PRIVATE
    -Wall -Wextra -Wno-sign-compare
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g -O0>
)
target_compile_definitions(anphon PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    _HDF5
)
target_include_directories(anphon PRIVATE
    ${PROJECT_SOURCE_DIR}/../include
    ${PROJECT_SOURCE_DIR}/../external
    ${spglib_include}
    ${boost_includes}
    ${eigen_includes}
    ${fftw_include_dir}
    ${hdf5_include}
)

target_link_libraries(anphon
    ${boost_target}
    ${eigen_target}
    ${Boost_LIBRARIES}
    ${lapack_target}
    ${mpi_target}
    ${spglib}
    ${openmplibrary}
    ${hdf5_target}
    ${hdf5library}
)
if (fftw3library)
    target_link_libraries(anphon ${fftw3library})
endif()
