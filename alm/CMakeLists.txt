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

project(alm)

# Version numbers
file(READ ${PROJECT_SOURCE_DIR}/../include/version.h version_file)
string(REGEX MATCH "ALAMODE_VERSION = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" alm_version ${version_file})
set(alm_version ${CMAKE_MATCH_1})
MESSAGE("Setting up for ALM code (ALAMODE version: ${CMAKE_MATCH_1})")

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
else ()
    message("SPGLIB not found via hints. Falling back to -lsymspg.")
    set(spglib "-lsymspg")
endif ()
#
# Add openmp flag if available
#
find_package(OpenMP REQUIRED COMPONENTS CXX)
if (OPENMP_FOUND)
    set(openmplibrary OpenMP::OpenMP_CXX)
endif ()

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
else ()
    if (BOOST_INCLUDE)
        set(boost_includes ${BOOST_INCLUDE})
    else ()
        message(FATAL_ERROR "Boost was not found.
        Please specify the location of boost include directories
        via -DBOOST_INCLUDE option.")
    endif ()
endif ()

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
else ()
    if (EIGEN3_INCLUDE)
        set(eigen_includes ${EIGEN3_INCLUDE})
    else ()
        message(FATAL_ERROR "Eigen3 was not found.
        Please specify the location of boost include directories
        via -DEIGEN3_INCLUDE option.")
    endif ()
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
else ()
    if (HDF5_ROOT)
        set(hdf5_include "${HDF5_ROOT}/include")
        set(hdf5library "-L${HDF5_ROOT}/lib -lhdf5_cpp -lhdf5")
    else ()
        message(FATAL_ERROR "HDF5 was not found.
        Please specify the HDF5 install directory
        via -DHDF5_ROOT option.")
    endif ()
endif ()

option(USE_MKL_BACKEND     "Enable Intel MKL Pardiso" OFF)
option(USE_ACCEL_BACKEND   "Enable Apple Accelerate LDLT" OFF)
# Route Eigen's dense matrix products through the detected (Fortran) BLAS -- e.g. OpenBLAS or Apple
# Accelerate -- via EIGEN_USE_BLAS. Layered on top of the generic find_package(BLAS/LAPACK) path
# below; choose the vendor with -DBLA_VENDOR=... (OpenBLAS, Apple, ...). Ignored under the MKL or
# Accelerate-sparse backends. Relies on the guarded BLAS prototypes in include/blas_wrapper.h.
option(USE_EIGEN_BLAS      "Use Eigen's BLAS backend (EIGEN_USE_BLAS) with the detected BLAS" OFF)

if (USE_MKL_BACKEND)
#    set(MKL_INTERFACE LP64 CACHE STRING "Use LP64 interface for MKL" FORCE)
    find_package(MKL REQUIRED)
    add_compile_definitions(USE_MKL_BACKEND EIGEN_USE_MKL_ALL)
    add_compile_definitions(EIGEN_USE_MKL_ALL)
    add_compile_definitions(EIGEN_MKL_BACKEND)
    add_compile_definitions(USE_MKL_BACKEND)
    # Prevent Eigen from enabling MKL's MKL_DIRECT_CALL when including mkl.h.
    # Otherwise BLAS routines (dtrsm/ztrsm/...) become direct-call macros that
    # clash with Eigen's own typed BLAS backend, breaking compilation.
    add_compile_definitions(EIGEN_MKL_NO_DIRECT_CALL)
#    if(MKL_INTERFACE STREQUAL "ILP64")
#        message(WARNING "MKL was detected as ILP64; switching to explicit LP64 interface.")

    set(BACKEND_LIBS mkl_intel_lp64 mkl_intel_thread mkl_core iomp5)
#    else()
#        set( BACKEND_LIBS MKL::MKL )
#    endif()
elseif (USE_ACCEL_BACKEND)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    add_compile_definitions(USE_ACCEL_BACKEND EIGEN_USE_ACCELERATE)
    set( BACKEND_LIBS ${ACCELERATE_FRAMEWORK} )
else()
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
    set ( BACKEND_LIBS ${blas_target} ${lapack_target} )
    if (USE_EIGEN_BLAS)
        add_compile_definitions(EIGEN_USE_BLAS)
        message(STATUS "EIGEN_USE_BLAS enabled: Eigen dense products routed through BLAS (${BLAS_LIBRARIES})")
    endif()
endif()

if (USE_EIGEN_BLAS AND (USE_MKL_BACKEND OR USE_ACCEL_BACKEND))
    message(WARNING "USE_EIGEN_BLAS is ignored when USE_MKL_BACKEND or USE_ACCEL_BACKEND is enabled "
                    "(those backends already route BLAS through MKL / Accelerate).")
endif()


# Source code
set(SOURCES ${PROJECT_SOURCE_DIR}/alm.cpp
        ${PROJECT_SOURCE_DIR}/cluster.cpp
        ${PROJECT_SOURCE_DIR}/constraint.cpp
        ${PROJECT_SOURCE_DIR}/fcs.cpp
        ${PROJECT_SOURCE_DIR}/files.cpp
        ${PROJECT_SOURCE_DIR}/input_parser.cpp
        ${PROJECT_SOURCE_DIR}/input_setter.cpp
        ${PROJECT_SOURCE_DIR}/optimize.cpp
        ${PROJECT_SOURCE_DIR}/patterndisp.cpp
        ${PROJECT_SOURCE_DIR}/rref.cpp
        ${PROJECT_SOURCE_DIR}/symmetry.cpp
        ${PROJECT_SOURCE_DIR}/system.cpp
        ${PROJECT_SOURCE_DIR}/timer.cpp
        ${PROJECT_SOURCE_DIR}/writer.cpp
        ${PROJECT_SOURCE_DIR}/smith.cpp
        ${PROJECT_SOURCE_DIR}/svd.cpp
        ${PROJECT_SOURCE_DIR}/least_squares.cpp)

set(HEADERS ${PROJECT_SOURCE_DIR}/alm.h
        ${PROJECT_SOURCE_DIR}/cluster.h
        ${PROJECT_SOURCE_DIR}/constraint.h
        ${PROJECT_SOURCE_DIR}/fcs.h
        ${PROJECT_SOURCE_DIR}/files.h
        ${PROJECT_SOURCE_DIR}/input_parser.h
        ${PROJECT_SOURCE_DIR}/input_setter.h
        ${PROJECT_SOURCE_DIR}/optimize.h
        ${PROJECT_SOURCE_DIR}/patterndisp.h
        ${PROJECT_SOURCE_DIR}/rref.h
        ${PROJECT_SOURCE_DIR}/symmetry.h
        ${PROJECT_SOURCE_DIR}/system.h
        ${PROJECT_SOURCE_DIR}/timer.h
        ${PROJECT_SOURCE_DIR}/writer.h
        ${PROJECT_SOURCE_DIR}/smith.h
        ${PROJECT_SOURCE_DIR}/svd.h
        ${PROJECT_SOURCE_DIR}/least_squares.h
        ../include/logger.h)

# # Trick to use gcc to compile *.cpp: This doesn't work because of boost
# SET_SOURCE_FILES_PROPERTIES(${SOURCES} PROPERTIES LANGUAGE C)
# set(CMAKE_C_COMPILER "gcc")

# Executable
add_executable(alm ${PROJECT_SOURCE_DIR}/main.cpp
        ${PROJECT_SOURCE_DIR}/alm_cui.cpp
        ${SOURCES} ${HEADERS})

# Warnings on for all build types. -Wno-sign-compare silences ~1500 benign
# signed/unsigned loop-counter comparisons so that the genuinely useful warnings
# (unused/uninitialized variables, VLAs, deprecated copies, ...) are visible.
# Drop -Wno-sign-compare once the loop counters have been cleaned up.
target_compile_options(alm PRIVATE
    -Wall -Wextra -Wno-sign-compare
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-g -O2>
)
target_compile_definitions(alm PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    _HDF5
)
target_include_directories(alm PRIVATE
    ${PROJECT_SOURCE_DIR}/../include
    ${PROJECT_SOURCE_DIR}/../external
    ${spglib_include}
    ${boost_includes}
    ${eigen_includes}
    ${hdf5_include}
)
target_link_libraries(alm
#        PRIVATE Eigen3::Eigen
    ${boost_target}
    ${eigen_target}
        ${Boost_LIBRARIES}
        ${BACKEND_LIBS}
        ${spglib}
        ${openmplibrary}
    ${hdf5_target}
        ${hdf5library}
)

if (BUILD_LIBRARIES)
    set(serial "${alm_version}")
    set(soserial "1")
    # Shared library
    add_library(almcxx SHARED ${SOURCES})
    target_compile_definitions(almcxx PRIVATE _HDF5)
    target_include_directories(almcxx PRIVATE
        ${PROJECT_SOURCE_DIR}/../include
        ${PROJECT_SOURCE_DIR}/../external
        ${spglib_include}
        ${boost_includes}
        ${eigen_includes}
        ${hdf5_include}
    )
    target_link_libraries(almcxx ${boost_target} ${eigen_target} ${BACKEND_LIBS} ${spglib} ${openmplibrary} ${hdf5_target} ${hdf5library})
    set_property(TARGET almcxx PROPERTY VERSION ${serial})
    set_property(TARGET almcxx PROPERTY SOVERSION ${soserial})
    install(TARGETS almcxx DESTINATION ${PROJECT_SOURCE_DIR}/lib)

    # Static link library
    add_library(almcxx_static STATIC ${SOURCES})
    target_compile_definitions(almcxx_static PRIVATE _HDF5)
    target_include_directories(almcxx_static PRIVATE
        ${PROJECT_SOURCE_DIR}/../include
        ${PROJECT_SOURCE_DIR}/../external
        ${spglib_include}
        ${boost_includes}
        ${eigen_includes}
        ${hdf5_include}
    )
    target_link_libraries(almcxx_static ${boost_target} ${eigen_target} ${BACKEND_LIBS} ${spglib} ${openmplibrary} ${hdf5_target} ${hdf5library})
    set_property(TARGET almcxx_static PROPERTY VERSION ${serial})
    set_property(TARGET almcxx_static PROPERTY SOVERSION ${soserial})
    set_property(TARGET almcxx_static PROPERTY OUTPUT_NAME almcxx)
    install(TARGETS almcxx_static ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/lib)
    # Header file
    install(FILES ${PROJECT_SOURCE_DIR}/alm.h DESTINATION ${PROJECT_SOURCE_DIR}/include)
endif ()
