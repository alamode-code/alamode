cmake_minimum_required(VERSION 3.17)

if(POLICY CMP0144)
	cmake_policy(SET CMP0144 NEW)
endif()
if(POLICY CMP0167)
	cmake_policy(SET CMP0167 NEW)
endif()

project(ALAMODE)

# Standard install locations (bin/, lib/, include/) under CMAKE_INSTALL_PREFIX,
# used by the install() rules in the subdirectories.
include(GNUInstallDirs)
message(STATUS "Install prefix (make install): ${CMAKE_INSTALL_PREFIX}")

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

add_subdirectory(alm)
add_subdirectory(anphon)
add_subdirectory(tools)

#enable_testing()
#message(${CMAKE_CURRENT_SOURCE_DIR})
#message(${CMAKE_BINARY_DIR})
#message($ENV{LD_LIBRARY_PATH})
#set(CTEST_ENVIRONMENT "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}")
#add_test(NAME silicon_test COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test/test_si.py")
