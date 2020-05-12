###################### CREATE LIBRARY ######################
add_executable(test_forward "test_forward.cpp")
set_target_properties(test_forward PROPERTIES VERSION "${PROJECT_VERSION}")

# and make sure we use C++-11
target_compile_features(test_forward PRIVATE cxx_std_17)

#################### START INCLUDES ####################
target_include_directories(test_forward PUBLIC "${CMAKE_HOME_DIRECTORY}/include")

#################### WE NEED FFTW3 ####################
find_package(FFTW3 REQUIRED)
target_include_directories(test_forward PUBLIC ${FFTW3_INCLUDES})
target_link_libraries(test_forward PUBLIC ${FFTW3_LIBRARIES})


#################### START COMPILE FLAGS ####################
# set(COMPILE_OPTIONS -Wall -Wextra -Wdisabled-optimization -fconcepts
#   -pedantic -Wshadow -Wno-sign-conversion -Wno-sign-compare -Wold-style-cast
#   -Wno-strict-overflow -Wundef -Wno-unused -Wstack-protector -Wcast-align
#   -Wpacked -Wformat-y2k -Warray-bounds -Wreorder -Wmissing-braces
#   -Wstrict-overflow=4 -Wwrite-strings -Wpointer-arith -ggdb3
#   -Wcast-qual -Wswitch-enum -Wswitch-default -Wunreachable-code
#   -Wmaybe-uninitialized -march=x86-64)
#-march=native -fmax-errors=5 -O2 -flto -Wfloat-equal)
set(COMPILE_OPTIONS -ggdb3)

# add the compile options to the library
target_compile_options(test_forward PRIVATE ${COMPILE_OPTIONS})

# and install our header files
install(TARGETS test_forward DESTINATION "bin")
