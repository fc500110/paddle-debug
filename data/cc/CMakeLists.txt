CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

project(PADDLE_DEBUG)
set(cmake_c_compiler gcc)
set(cmake_cxx_compiler g++)
add_compile_options(-std=c++11 -g -O0)

function(add_paddle)
    include_directories(${FLUID_INFER_LIB}/paddle/include)
    link_directories(${FLUID_INFER_LIB}/paddle/lib)
endfunction()

function(add_third_party libs)
    foreach (lib ${libs})
        set(lib_abs_path ${FLUID_INFER_LIB}/third_party/${lib})
        include_directories(${lib_abs_path}/include)
        link_directories(${lib_abs_path}/lib)
    endforeach()
endfunction()

add_paddle()
add_third_party(glog gflags mkldnn mklml)
include_directories($ENV{PWD}/..)

file(GLOB src_files debug_reader.cc)
add_library(debug_reader SHARED ${src_files})
target_link_libraries(debug_reader dl paddle_fluid glog pthread)

file(GLOB ut_files ./tests/*.cc)

foreach(test ${ut_files})
    string(REGEX REPLACE ".*/tests/(.*).cc$" "\\1" ut_app ${test})
    add_executable(${ut_app} ${test})
    target_link_libraries(${ut_app} debug_reader gtest boost_filesystem boost_system)
endforeach()

# gtest_add_tests(TARGET)
