CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

project(PADDLE_DEBUG)
set(cmake_c_compiler gcc)
set(cmake_cxx_compiler g++)
add_compile_options(-std=c++11 -g -O0 -D_GLIBCXX_USE_CXX11_ABI=0)

function(add_paddle)
  include_directories(${FLUID_INFER_LIB}/paddle/include)
  link_directories(${FLUID_INFER_LIB}/paddle/lib)
endfunction()

function(add_third_party libs)
  foreach (lib ${libs})
    set(lib_abs_path ${FLUID_INFER_LIB}/third_party/install/${lib})
    include_directories(${lib_abs_path}/include)
    link_directories(${lib_abs_path}/lib)
  endforeach()
endfunction()

add_paddle()
add_third_party(mkldnn mklml)

include_directories(.)

# libpaddle_debug
file(GLOB src_files debug_reader.cc)
add_library(paddle_debug SHARED ${src_files})
target_link_libraries(paddle_debug dl paddle_fluid glog pthread)

# app
file(GLOB app_files ./app/*.cc)
add_executable(app ${app_files})
target_link_libraries(app gflags glog paddle_debug paddle_fluid dl pthread)

# unittest
file(GLOB ut_files ./tests/*.cc)
foreach(test ${ut_files})
  string(REGEX REPLACE ".*/tests/(.*).cc$" "\\1" ut_app ${test})
  add_executable(${ut_app} ${test})
  target_link_libraries(${ut_app} paddle_debug gtest boost_filesystem boost_system)
endforeach()

# gtest_add_tests(TARGET)
