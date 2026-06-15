cmake_minimum_required(VERSION 3.16)
project(MetalGEMM VERSION 1.0.0 LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

find_library(METAL_LIB Metal REQUIRED)
find_library(FOUNDATION_LIB Foundation REQUIRED)
find_library(ACCELERATE_LIB Accelerate REQUIRED)

include_directories(include)

file(GLOB SOURCES src/*.cpp src/*.mm)

add_executable(metalgemm ${SOURCES})

target_link_libraries(metalgemm
    ${METAL_LIB}
    ${FOUNDATION_LIB}
    ${ACCELERATE_LIB}
)

target_compile_options(metalgemm PRIVATE -O3)
