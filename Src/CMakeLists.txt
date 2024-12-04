set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Include directories
include_directories(
  ${SOURCE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_INCLUDE}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
)

# Collect headers and source files
file(GLOB HEADERS ${SOURCE_DIR}/*.h ${SOURCE_DIR}/*.cuh)
file(GLOB SOURCES ${SOURCE_DIR}/*.cpp ${SOURCE_DIR}/*.cu ${SOURCE_DIR}/*.c)

# Group headers in IDEs
source_group("Headers" FILES ${HEADERS})

# Find CUDA
find_package(CUDA REQUIRED)

# CUDA
cuda_add_library(naive_lib STATIC naive.cu OPTIONS -arch=sm_52)
cuda_add_library(opt_lib STATIC opt.cu OPTIONS -arch=sm_52)
cuda_add_library(cpu_lib STATIC cpu.cu OPTIONS -arch=sm_52)
cuda_add_library(cublas_lib STATIC cublas.cu OPTIONS -arch=sm_52)

# C
add_library(cblas_lib STATIC cblas.c)

# CUDA properties for libraries
set_target_properties(naive_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(opt_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cpu_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cublas_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Executable
cuda_add_executable(benchmark 
  ${HEADERS}
  benchmark.cu
  OPTIONS --cudart=shared  -lineinfo -arch sm_52 -use_fast_math
  DEBUG -Xptxas -v -g -G 
)

# Set CUDA properties for benchmark executable
set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link libraries to the benchmark executable
target_link_libraries(benchmark
  naive_lib
  opt_lib
  cpu_lib
  cublas_lib
  cblas_lib
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_CURAND_LIBRARY}
  ${NVML_DEBUG_LIBRARIES}
  ${NVTX_DEBUG_LIBRARIES}
  ${FREEGLUT_DEBUG_LIBRARIES}
  ${GLEW_DEBUG_LIBRARIES}
)
