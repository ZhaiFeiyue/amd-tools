cmake_minimum_required(VERSION 3.15)


# Specify name of project to build, install and package
set(PROJECT_NAME "auto_topo")
set(TEST_NAME "${PROJECT_NAME}")
project(${PROJECT_NAME})

include(GNUInstallDirs)
# Build is not supported on Windows plaform
if(WIN32)
  message("Windows platfom is not supported")
  return()
endif()

# Flag to enable / disable verbose output.
option(CMAKE_VERBOSE_MAKEFILE "Enable verbose output" ON)
option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile commands for linters and autocompleters" ON)

# Compiler Preprocessor definitions.
add_definitions(-DUNIX_OS)
add_definitions(-DLINUX)
add_definitions(-D__AMD64__)
add_definitions(-DAMD_INTERNAL_BUILD)
add_definitions(-DLITTLEENDIAN_CPU=1)
add_definitions(-DHSA_LARGE_MODEL=)
add_definitions(-DHSA_DEPRECATED=)

# Enable debug trace
if(DEFINED ENV{CMAKE_DEBUG_TRACE})
  add_definitions(-DDEBUG_TRACE=1)
endif()

# Linux Compiler options
# Allow compiler flags to inherit any set by env
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

add_compile_definitions(USE_ROCM)
add_compile_definitions(__HIP_PLATFORM_AMD__)

# Add sources that belong to the project
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} Src)

include_directories(/opt/rocm/include/hip)
link_directories(/opt/rocm/lib)

add_executable(${TEST_NAME} auto_topo.cpp)

target_link_libraries(${TEST_NAME} PUBLIC amdhip64)
