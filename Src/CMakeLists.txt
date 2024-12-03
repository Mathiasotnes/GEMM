# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/Src
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_INCLUDE}
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
)

# Collect headers and source files
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/Src/*.h ${CMAKE_CURRENT_SOURCE_DIR}/Src/*.cuh)
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Src/*.cu)

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
  Src/benchmark.cu 
  ${HEADERS}
  OPTIONS -arch=sm_52 --cudart=shared -lineinfo -use_fast_math
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
  $<$<CONFIG:Debug>:${NVML_DEBUG_LIBRARIES} ${NVTX_DEBUG_LIBRARIES}>
  $<$<CONFIG:Release>:${FREEGLUT_RELEASE_LIBRARIES} ${GLEW_RELEASE_LIBRARIES}>
)

# Add compile options
target_compile_options(benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo -arch=sm_52>
)

# Add definitions if necessary
target_compile_definitions(benchmark PRIVATE
    -DUSE_FAST_MATH
)
