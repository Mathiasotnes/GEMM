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
file(GLOB SOURCES ${SOURCE_DIR}/*.cpp ${SOURCE_DIR}/*.cu)

# Group headers in IDEs
source_group("Headers" FILES ${HEADERS})

# Find CUDA (modern CMake can find CUDA without the FindCUDA module)
find_package(CUDA REQUIRED)

# Libraries
cuda_add_library(gemm_naive_lib STATIC Src/gemm_naive.cu OPTIONS -arch=sm_52)
cuda_add_library(gemm_opt_lib STATIC Src/gemm_opt.cu OPTIONS -arch=sm_52)
cuda_add_library(gemm_cpu_lib STATIC Src/gemm_cpu.c)

# CUDA properties for libraries
set_target_properties(gemm_naive_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gemm_opt_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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
  gemm_naive_lib
  gemm_opt_lib
  gemm_cpu_lib
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_CURAND_LIBRARY}
  ${NVML_DEBUG_LIBRARIES}
  ${NVTX_DEBUG_LIBRARIES}
  ${FREEGLUT_DEBUG_LIBRARIES}
  ${GLEW_DEBUG_LIBRARIES}
)
