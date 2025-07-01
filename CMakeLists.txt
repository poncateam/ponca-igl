cmake_minimum_required(VERSION 3.14)
project(PONCA_IGL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use Eigen from Ponca
add_subdirectory(external/ponca)
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/ponca/external/eigen")

# Tell libigl it's not top-level
set(LIBIGL_PARENT_DIR ON)

# Force-enable modules needed for viewer
set(LIBIGL_USE_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
set(LIBIGL_OPENGL ON CACHE BOOL "" FORCE)
set(LIBIGL_GLFW ON CACHE BOOL "" FORCE)
set(LIBIGL_IMGUI ON CACHE BOOL "" FORCE)
set(LIBIGL_VIEWER ON CACHE BOOL "" FORCE)

include(FetchContent)

FetchContent_Declare(
        glad
        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
        GIT_TAG        v0.1.36
)

FetchContent_MakeAvailable(glad)

# Add libigl
add_subdirectory(external/libigl)

# Executable
add_executable(${PROJECT_NAME} src/main.cpp)


# Include Eigen and libigl
target_include_directories(${PROJECT_NAME}
        PRIVATE ${EIGEN3_INCLUDE_DIR}
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/external/libigl/include"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/external/ponca"
)


target_link_libraries(${PROJECT_NAME}
        PRIVATE glad
        PRIVATE igl::core
        PRIVATE igl::glfw
        PRIVATE igl::opengl
        PRIVATE igl::imgui
)
