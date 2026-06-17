cmake_minimum_required(VERSION 3.10)

project(ALAMODE)

add_subdirectory(alm)
add_subdirectory(anphon)
add_subdirectory(tools)

#enable_testing()
#message(${CMAKE_CURRENT_SOURCE_DIR})
#message(${CMAKE_BINARY_DIR})
#message($ENV{LD_LIBRARY_PATH})
#set(CTEST_ENVIRONMENT "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}")
#add_test(NAME silicon_test COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test/test_si.py")
