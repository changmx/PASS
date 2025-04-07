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

# set source files
set(SOURCES
    src/main.cpp
    src/particle.cpp
    src/general.cu
    src/parameter.cpp
    src/readCommand.cu
    src/injection.cu
    src/twiss.cu
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

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)
target_compile_features(${PROJECT_NAME} PRIVATE cuda_std_17)

# set spdlog level: [SPDLOG_LEVEL_TRACE, SPDLOG_LEVEL_DEBUG, SPDLOG_LEVEL_INFO, SPDLOG_LEVEL_WARN, SPDLOG_LEVEL_ERROR, SPDLOG_LEVEL_CRITICAL, SPDLOG_LEVEL_OFF]
target_compile_definitions(${PROJECT_NAME} PRIVATE PASS_SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)