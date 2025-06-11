cmake_minimum_required(VERSION 3.10)

project(PONCA-IGL)

add_executable(${PROJECT_NAME} ./src/main.cpp)

add_subdirectory(external/ponca)
add_subdirectory(external/libigl)