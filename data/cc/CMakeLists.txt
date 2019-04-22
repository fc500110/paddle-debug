CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

project(PADDLE_DEBUG)
set(cmake_c_compiler gcc)
set(cmake_cxx_compiler g++)
add_compile_options(-std=c++11)

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

file(GLOB src_files debug_reader.cc)
add_library(debug_reader ${src_files})
target_link_libraries(debug_reader dl paddle_fluid glog pthread)

