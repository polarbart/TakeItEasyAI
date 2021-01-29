cmake_minimum_required(VERSION 3.10)
project(TakeItEasyC)

find_package(Git QUIET)
if(GIT_FOUND AND NOT EXISTS "${PROJECT_SOURCE_DIR}/lib/pybind11/CMakeLists.txt")
    message(STATUS "Cloning pybind11")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive)
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/lib/pybind11/CMakeLists.txt")
    message(FATAL_ERROR "pybind11 has to be cloned into lib/pybind11")
endif()

find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "Couldn't find OpenMP!")
endif()

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(lib/pybind11)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/$<0:>)

pybind11_add_module(TakeItEasyC Bindings.cpp TakeItEasy.cpp BatchedTakeItEasy.cpp)
target_link_libraries(TakeItEasyC PUBLIC OpenMP::OpenMP_CXX)
