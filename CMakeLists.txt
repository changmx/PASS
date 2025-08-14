cmake_minimum_required(VERSION 3.29)
project(PASS 
        VERSION 0.0.1 
        LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)

option (PASS_CAL_PHASE "Record the coordinates of the particle at the previous position/turn to calculate the phase shift" ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python COMPONENTS Interpreter Development)

# get build time and define it in config.h
string(TIMESTAMP CMAKE_BUILD_TIME "%Y-%m-%d %H:%M:%S")
configure_file(${PROJECT_SOURCE_DIR}/config/config.h.in ${PROJECT_SOURCE_DIR}/config/config.h)

# set cuDSS installation path
#set(CUDSS_ROOT "C:/Program Files/NVIDIA cuDSS/v0.6")
find_package(cudss REQUIRED PATHS "C:/Program Files/NVIDIA cuDSS/v0.6/lib/12/cmake/cudss")
message(STATUS "CUDSS include dir: ${cudss_INCLUDE_DIR}")
message(STATUS "CUDSS library dir: ${cudss_LIBRARY_DIR}")

# set AMGX installation path
# set(AMGX_ROOT ${PROJECT_SOURCE_DIR}/libs/amgx CACHE PATH "AMGX installation directory")
# message(STATUS "Using AMGX from: ${AMGX_ROOT}")

# find_path(AMGX_INCLUDE_DIR
#     NAMES amgx_c.h
#     PATHS "${AMGX_ROOT}/include"
#     REQUIRED
# )
# find_library(AMGX_LIBRARY
#     NAMES amgxsh
#     PATHS "${AMGX_ROOT}/lib"
#     NO_DEFAULT_PATH
#     REQUIRED
# )

# set(AMGX_LIBRARIES ${AMGX_LIBRARY}) 

# # Display the found AMGX information
# message(STATUS "AMGX include dir: ${AMGX_INCLUDE_DIR}")
# message(STATUS "AMGX library: ${AMGX_LIBRARIES}")

# set source files
set(SOURCES
    src/main.cpp
    src/particle.cu
    src/general.cu
    src/parameter.cpp
    src/readCommand.cu
    src/injection.cu
    src/twiss.cu
    src/element.cu
    src/monitor.cu
    src/cutSlice.cu
    src/pic.cu
    src/spaceCharge.cu
    src/aperture.cu
)

# set header files
set(HEADERS
    include/constant.h
    include/particle.h
    include/general.h
    include/parameter.h
    include/command.h
    include/readCommand.h
    include/injection.h
    include/twiss.h
    include/element.h
    include/parallelPlan.h
    include/monitor.h
    include/cutSlice.h
    include/pic.h
    include/spaceCharge.h
    include/aperture.h
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
    ${PROJECT_SOURCE_DIR}/include/cmdline
    ${PROJECT_SOURCE_DIR}/include/tabulate
    ${PROJECT_SOURCE_DIR}/include/spdlog
    ${AMGX_INCLUDE_DIR}
    ${cudss_INCLUDE_DIR}
)

target_link_directories(${PROJECT_NAME}
    PRIVATE
    ${Python_LIBRARIES}
    ${cudss_LIBRARY_DIR}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    OpenMP::OpenMP_CXX
    CUDA::cufft
    CUDA::curand
    CUDA::cusparse
    # ${AMGX_LIBRARIES}
    cudss
)

# set CUDA ARCHITECTURE. (https://developer.nvidia.com/cuda-gpus)
# 70 for V100, 75 for RTX2060~2080Ti, 80 for A100, 86 for RTX3050~RTX3090Ti, 89 for RTX4050~RTX4090, 120 for RTX5060-5090
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "80;86;89;120"
    CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    # --ptxas-options=-v
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)
target_compile_features(${PROJECT_NAME} PRIVATE cuda_std_17)

# set spdlog level: [SPDLOG_LEVEL_TRACE, SPDLOG_LEVEL_DEBUG, SPDLOG_LEVEL_INFO, SPDLOG_LEVEL_WARN, SPDLOG_LEVEL_ERROR, SPDLOG_LEVEL_CRITICAL, SPDLOG_LEVEL_OFF]
target_compile_definitions(${PROJECT_NAME} PRIVATE PASS_SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)