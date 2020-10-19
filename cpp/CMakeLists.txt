cmake_minimum_required(VERSION 3.15)
project(TakeItEasyC)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "../.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(
                COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/.."
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/lib/pybind11/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "Couldn't find OpenMP!")
endif()

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(lib/pybind11)

pybind11_add_module(TakeItEasyC Bindings.cpp TakeItEasy.cpp)
target_link_libraries(TakeItEasyC PUBLIC OpenMP::OpenMP_CXX)
