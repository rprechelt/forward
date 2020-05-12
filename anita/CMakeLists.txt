# create a static library
add_library(waveletdeconv STATIC "WaveletDeconvolution.cc")

# add our include directory for header files
target_include_directories(waveletdeconv PRIVATE "${CMAKE_HOME_DIRECTORY}")

# we also need access to ANITA tools to build waveletdeconv
target_include_directories(waveletdeconv PRIVATE "${ANITA_UTIL_INSTALL_DIR}/include")

# add the ANITA_UTIL_INSTALL_DIR to our library search path
target_link_directories(waveletdeconv PUBLIC $ENV{ANITA_UTIL_INSTALL_DIR}/lib)

# and we need ROOT
find_package(ROOT REQUIRED)

# link against forward
target_include_directories(waveletdeconv PUBLIC "${CMAKE_HOME_DIRECTORY}/include")#

# setup our include directories for ROOT
target_include_directories(waveletdeconv PUBLIC "${ROOT_INCLUDE_DIRS}")

# and link against ROOT and FFTW
target_link_libraries(waveletdeconv PUBLIC "${ROOT_LIBRARIES}")

# we need to link against AnitaEvent
find_library(ANITA_EVENT_LIB AnitaEvent
  PATHS $ENV{ANITA_UTIL_INSTALL_DIR}/lib
  REQUIRED
  )

# we also need to link against AnitaAnalysis
find_library(ANITA_ANALYSIS_LIB AnitaAnalysis
  PATHS $ENV{ANITA_UTIL_INSTALL_DIR}/lib
  REQUIRED
  )

# link against the needed analysis tools
target_link_libraries(waveletdeconv PUBLIC ${ANITA_EVENT_LIB})
target_link_libraries(waveletdeconv PUBLIC ${ANITA_ANALYSIS_LIB})

set(COMPILE_OPTIONS -Wall -Wextra -Wdisabled-optimization -fconcepts
  -pedantic -Wshadow -Wno-sign-conversion -Wno-sign-compare -Wold-style-cast
  -Wno-strict-overflow -Wundef -Wno-unused -Wstack-protector -Wcast-align
  -Wpacked -Wformat-y2k -Warray-bounds -Wreorder -Wmissing-braces
  -Wstrict-overflow=4 -Wwrite-strings -Wpointer-arith
  -Wcast-qual -Wswitch-enum -Wswitch-default -Wunreachable-code
  -Wmaybe-uninitialized -march=native -fmax-errors=5 -O3 -Wfloat-equal)

# add the compile options to the library
target_compile_options(waveletdeconv PRIVATE ${COMPILE_OPTIONS})
