cmake_minimum_required(VERSION 3.26.4)

include(FetchContent)

project(cuTensor VERSION 1.0)

enable_language(CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")
add_executable(scratchpad test/scratchpad.cu ${SOURCES})

# Optionally, you can add include directories if you have headers in a specific directory
target_include_directories(scratchpad PRIVATE src)

# Optionally, you can link libraries if your project depends on them
target_link_libraries(scratchpad fmt::fmt)
