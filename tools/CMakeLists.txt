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

project(tools)

# Version numbers
file(READ ${PROJECT_SOURCE_DIR}/../include/version.h version_file)
string(REGEX MATCH "ALAMODE_VERSION = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" alm_version ${version_file})
set(alm_version ${CMAKE_MATCH_1})
MESSAGE("Setting up for TOOLS (ALAMODE version: ${CMAKE_MATCH_1})")

find_package(OpenMP)
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
else()
    if (EIGEN3_INCLUDE)
        set(eigen_includes ${EIGEN3_INCLUDE})
    else()
        message(FATAL_ERROR "Eigen3 was not found.
        Please specify the location of boost include directories
        via -DEIGEN3_INCLUDE option.")
    endif()
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
target_compile_options(analyze_phonons PRIVATE
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-g -Wall -O0>
)
target_compile_definitions(analyze_phonons PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
)
target_include_directories(analyze_phonons PRIVATE
    ${PROJECT_SOURCE_DIR}/../include
    ${boost_includes}
    ${eigen_includes}
)
if (boost_target)
    target_link_libraries(analyze_phonons ${boost_target})
endif()
if (eigen_target)
    target_link_libraries(analyze_phonons ${eigen_target})
endif()
if (openmplibrary)
    target_link_libraries(analyze_phonons ${openmplibrary})
endif()

add_executable(dfc2
        ${PROJECT_SOURCE_DIR}/dfc2.cpp
        ${PROJECT_SOURCE_DIR}/dfc2.h)
target_compile_options(dfc2 PRIVATE
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-g -Wall -O0>
)
target_compile_definitions(dfc2 PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
)
target_include_directories(dfc2 PRIVATE
    ${PROJECT_SOURCE_DIR}/../include
    ${boost_includes}
    ${eigen_includes}
)
target_link_libraries(dfc2 ${boost_target} ${eigen_target} ${Boost_LIBRARIES})
if (openmplibrary)
    target_link_libraries(dfc2 ${openmplibrary})
endif()

add_executable(qe2alm
        ${PROJECT_SOURCE_DIR}/qe2alm.cpp
        ${PROJECT_SOURCE_DIR}/qe2alm.h)
target_compile_options(qe2alm PRIVATE
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-g -Wall -O0>
)
target_compile_definitions(qe2alm PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
)
target_include_directories(qe2alm PRIVATE
    ${PROJECT_SOURCE_DIR}/../include
    ${boost_includes}
    ${eigen_includes}
)
target_link_libraries(qe2alm ${boost_target} ${eigen_target} ${Boost_LIBRARIES})
if (openmplibrary)
    target_link_libraries(qe2alm ${openmplibrary})
endif()

add_executable(fc_virtual
        ${PROJECT_SOURCE_DIR}/fc_virtual.cpp
        ${PROJECT_SOURCE_DIR}/fc_virtual.h)
target_compile_options(fc_virtual PRIVATE
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-g -Wall -O0>
)
target_compile_definitions(fc_virtual PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
)
target_include_directories(fc_virtual PRIVATE
    ${PROJECT_SOURCE_DIR}/../include
    ${boost_includes}
    ${eigen_includes}
)
target_link_libraries(fc_virtual ${boost_target} ${eigen_target} ${Boost_LIBRARIES})
if (openmplibrary)
    target_link_libraries(fc_virtual ${openmplibrary})
endif()

add_executable(parse_fcsxml
        ${PROJECT_SOURCE_DIR}/parse_fcsxml.cpp
        ${PROJECT_SOURCE_DIR}/parse_fcsxml.h)
target_compile_options(parse_fcsxml PRIVATE
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-g -Wall -O0>
)
target_compile_definitions(parse_fcsxml PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
)
target_include_directories(parse_fcsxml PRIVATE
    ${PROJECT_SOURCE_DIR}/../include
    ${boost_includes}
    ${eigen_includes}
)
target_link_libraries(parse_fcsxml ${boost_target} ${eigen_target} ${Boost_LIBRARIES})
if (openmplibrary)
    target_link_libraries(parse_fcsxml ${openmplibrary})
endif()
