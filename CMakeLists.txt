# this minimum version hasn't been
# tested extensively
cmake_minimum_required(VERSION 2.8.10)

# setup our project
project(ForWaRD
  LANGUAGES CXX)

# and a version
set(CMAKE_PROJECT_VERSION 0.1.0)

###################### CREATE LIBRARY ######################
add_library(ForWaRD INTERFACE)  # header only

# and set the include directory
target_include_directories(ForWaRD INTERFACE "include/")

#################### START FIND PACKAGES ####################
find_package(FFTW3 REQUIRED)
#target_include_directories(ForWaRD PUBLIC ${FFTW_INCLUDES})

# build the test/ directory into a shared library
add_subdirectory(test)

# add an option to compile the Python interface
option(WITH_PYTHON "Compile the Python wrapper" OFF)

# if requested, build the Python interface
if (WITH_PYTHON)
  add_subdirectory(python)
endif()

# add an option to compile the AnitaTools interface
option(ANITA_TOOLS "Compile the AnitaTools interface" OFF)

# if requested, build the Python interface
if (ANITA_TOOLS)
  add_subdirectory(anita)
endif()
