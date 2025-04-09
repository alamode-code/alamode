cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -O0 -D_DEBUG")

project(tools)

# Version numbers
file(READ ${PROJECT_SOURCE_DIR}/../include/version.h version_file)
string(REGEX MATCH "ALAMODE_VERSION = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" alm_version ${version_file})
set(alm_version ${CMAKE_MATCH_1})
MESSAGE("Setting up for TOOLS (ALAMODE version: ${CMAKE_MATCH_1})")

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

find_package(Boost)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
else ()
    if (BOOST_INCLUDE)
        include_directories(${BOOST_INCLUDE})
    else ()
        message(FATAL_ERROR "Boost was not found.
        Please specify the location of boost include directories
        via -DBOOST_INCLUDE option.")
    endif ()
endif ()

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
        ${PROJECT_SOURCE_DIR}/writer.cpp)

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
        ${PROJECT_SOURCE_DIR}/writer.h)

# Executable
add_executable(analyze_phonons
        ${PROJECT_SOURCE_DIR}/analyze_phonons.cpp
        ${PROJECT_SOURCE_DIR}/analyze_phonons.h)

add_executable(dfc2
        ${PROJECT_SOURCE_DIR}/dfc2.cpp
        ${PROJECT_SOURCE_DIR}/dfc2.h)
target_link_libraries(dfc2 ${Boost_LIBRARIES})

add_executable(qe2alm
        ${PROJECT_SOURCE_DIR}/qe2alm.cpp
        ${PROJECT_SOURCE_DIR}/qe2alm.h)
target_link_libraries(qe2alm ${Boost_LIBRARIES})

add_executable(fc_virtual
        ${PROJECT_SOURCE_DIR}/fc_virtual.cpp
        ${PROJECT_SOURCE_DIR}/fc_virtual.h)
target_link_libraries(qe2alm ${Boost_LIBRARIES})

add_executable(parse_fcsxml
        ${PROJECT_SOURCE_DIR}/parse_fcsxml.cpp
        ${PROJECT_SOURCE_DIR}/parse_fcsxml.h)
target_link_libraries(parse_fcsxml ${Boost_LIBRARIES})
