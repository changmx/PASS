cmake_minimum_required(VERSION 3.29)
project(PASS 
        VERSION 0.0.1 
        LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python COMPONENTS Interpreter Development)

# get build time and define it in config.h
string(TIMESTAMP CMAKE_BUILD_TIME "%Y-%m-%d %H:%M:%S")
configure_file(${PROJECT_SOURCE_DIR}/config/config.h.in ${PROJECT_SOURCE_DIR}/config/config.h)

# set source files
set(SOURCES
    src/main.cpp
    src/general.cu
)

# set header files
set(HEADERS
    include/general.h
    include/particle.h
)

# generate executable file
# CMake compiles only the source files according to their extensions. 
# In most cases, the header files will be ignored, the only reason to list them is so that they can be displayed in the IDE.
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# set include files' directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/config
    ${Python_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include/nlohmann
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    OpenMP::OpenMP_CXX
    ${Python_LIBRARIES}
    CUDA::cufft
    CUDA::curand
    CUDA::cusparse
)

# set CUDA ARCHITECTURE. 70 for V100, 75 for RTX2060, 80 for A100, 89 for RTX4080
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;89"
    CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)
target_compile_features(${PROJECT_NAME} PRIVATE cuda_std_14)