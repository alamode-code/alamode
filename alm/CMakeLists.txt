cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -O2 -D_DEBUG")

project(alm)

# Version numbers
file(READ ${PROJECT_SOURCE_DIR}/../include/version.h version_file)
string(REGEX MATCH "ALAMODE_VERSION = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" alm_version ${version_file})
set(alm_version ${CMAKE_MATCH_1})
MESSAGE("Setting up for ALM code (ALAMODE version: ${CMAKE_MATCH_1})")

if (SPGLIB_ROOT)
    include_directories("${SPGLIB_ROOT}/include")
    set(spglib "-L${SPGLIB_ROOT}/lib -L${SPGLIB_ROOT}/lib64 -lsymspg")
else()
    message("SPGLIB_ROOT is not given by -DSPGLIB_ROOT option. Just add -lsymspg.")
    set(spglib "-lsymspg")
endif()
#
# Add openmp flag if available
#
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (openmplibrary OpenMP::OpenMP_CXX)
endif()

find_package(Boost)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
else()
    if (BOOST_INCLUDE)
        include_directories(${BOOST_INCLUDE})
    else()
        message(FATAL_ERROR "Boost was not found.
        Please specify the location of boost include directories
        via -DBOOST_INCLUDE option.")
    endif()
endif()

find_package(Eigen3)
if (Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    if (EIGEN3_INCLUDE)
        include_directories(${EIGEN3_INCLUDE})
    else()
        message(FATAL_ERROR "Eigen3 was not found.
        Please specify the location of boost include directories
        via -DEIGEN3_INCLUDE option.")
    endif()
endif()

add_definitions(-D_HDF5)
find_package(HDF5 COMPONENTS CXX)
if (HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    set(hdf5library ${HDF5_LIBRARIES})
else()
    if (HDF5_ROOT)
        include_directories("${HDF5_ROOT}/include")
        set(hdf5library "-L${HDF5_ROOT}/lib -lhdf5_cpp -lhdf5")
    else()
        message(FATAL_ERROR "HDF5 was not found.
        Please specify the HDF5 install directory
        via -DHDF5_ROOT option.")
    endif()
endif()

include_directories("${PROJECT_SOURCE_DIR}/../external")


find_package(LAPACK REQUIRED)
include_directories(${Lapack_INCLUDE_DIRS})

# Source code
include_directories("${PROJECT_SOURCE_DIR}/../include")
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
            ${PROJECT_SOURCE_DIR}/svd.cpp)

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
            ${PROJECT_SOURCE_DIR}/svd.h)

# # Trick to use gcc to compile *.cpp: This doesn't work because of boost
# SET_SOURCE_FILES_PROPERTIES(${SOURCES} PROPERTIES LANGUAGE C)
# set(CMAKE_C_COMPILER "gcc")

# Executable
add_executable(alm ${PROJECT_SOURCE_DIR}/main.cpp
                   ${PROJECT_SOURCE_DIR}/alm_cui.cpp
                   ${SOURCES} ${HEADERS})
target_link_libraries(alm ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} ${spglib} ${openmplibrary} ${hdf5library})

if (BUILD_LIBRARIES)
    set(serial "${alm_version}")
    set(soserial "1")
    # Shared library
    add_library(almcxx SHARED ${SOURCES})
    target_link_libraries(almcxx ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} ${spglib} ${hdf5library})
    set_property(TARGET almcxx PROPERTY VERSION ${serial})
    set_property(TARGET almcxx PROPERTY SOVERSION ${soserial})
    install(TARGETS almcxx DESTINATION ${PROJECT_SOURCE_DIR}/lib)

    # Static link library
    add_library(almcxx_static STATIC ${SOURCES})
    set_property(TARGET almcxx_static PROPERTY VERSION ${serial})
    set_property(TARGET almcxx_static PROPERTY SOVERSION ${soserial})
    set_property(TARGET almcxx_static PROPERTY OUTPUT_NAME almcxx)
    install(TARGETS almcxx_static ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/lib)
    # Header file
    install(FILES ${PROJECT_SOURCE_DIR}/alm.h DESTINATION ${PROJECT_SOURCE_DIR}/include)
endif()
