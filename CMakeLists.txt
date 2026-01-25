cmake_minimum_required(VERSION 3.15)
project(river-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(river-cpp INTERFACE)
target_include_directories(river-cpp INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_executable(quick_start example/ex_phishing.cpp)
    target_link_libraries(quick_start PRIVATE river-cpp)
endif()