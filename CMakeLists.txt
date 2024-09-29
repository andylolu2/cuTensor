cmake_minimum_required(VERSION 3.26.4)

include(FetchContent)

project(cuTensor VERSION 1.0)

enable_language(CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")
add_library(cuTensor SHARED ${SOURCES})
target_include_directories(cuTensor PUBLIC src)
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
set_property(TARGET cuTensor PROPERTY CUDA_ARCHITECTURES native)
set_target_properties(cuTensor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuTensor fmt::fmt)

add_executable(scratchpad test/scratchpad.cpp)
set_source_files_properties(test/scratchpad.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(scratchpad PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Optionally, you can add include directories if you have headers in a specific directory
target_include_directories(scratchpad PRIVATE src)

# Optionally, you can link libraries if your project depends on them
target_link_libraries(scratchpad cuTensor)
