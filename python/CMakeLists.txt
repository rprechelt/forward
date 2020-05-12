# we want to support C++11 with PyBind11
set(PYBIND11_CPP_STANDARD -std=c++17)

# find and load PyBind11
find_package(pybind11)

# check that we found PyBind11
IF(NOT pybind11_FOUND)

  # update the user
  message(WARNING "PyBind11 not found. Downloading local PyBind11 copy...")

  # load the external project module
  include(${CMAKE_ROOT}/Modules/FetchContent.cmake)

  # fetch PyBind11
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.4.3)

  # make PyBind11 available
  FetchContent_MakeAvailable(pybind11)

  # load the properties of PyBind11
  FetchContent_GetProperties(pybind11)
  if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
  endif()

  # and now
ENDIF()

# and wrap each of these sources into our app
# use SYSTEM to include PyBind11 as a system library
pybind11_add_module(forward SYSTEM "PyForward.cpp")

# add our include directory for header files
target_include_directories(forward PRIVATE "${CMAKE_HOME_DIRECTORY}/include")

# and we need FFTW3
find_package(FFTW3 REQUIRED)
target_include_directories(forward PUBLIC ${FFTW3_INCLUDES})
target_link_libraries(forward PUBLIC ${FFTW3_LIBRARIES})

set(COMPILE_OPTIONS -Wall -Wextra -Wdisabled-optimization -fconcepts
  -pedantic -Wshadow -Wno-sign-conversion -Wno-sign-compare -Wold-style-cast
  -Wno-strict-overflow -Wundef -Wno-unused -Wstack-protector -Wcast-align
  -Wpacked -Wformat-y2k -Warray-bounds -Wreorder -Wmissing-braces
  -Wstrict-overflow=4 -Wwrite-strings -Wpointer-arith
  -Wcast-qual -Wswitch-enum -Wswitch-default -Wunreachable-code
  -Wmaybe-uninitialized -march=native -fmax-errors=5 -O3 -flto -Wfloat-equal)

# add the compile options to the library
target_compile_options(forward PRIVATE ${COMPILE_OPTIONS})

# add our install target
install(TARGETS forward
  CONFIGURATIONS Release
  LIBRARY
  DESTINATION "forward")
