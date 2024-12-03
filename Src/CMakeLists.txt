# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
)

# Collect headers and source files
file(GLOB HEADERS *.h *.cuh)
file(GLOB SOURCES *.cpp *.cu)

# Group headers in IDEs
source_group("Headers" FILES ${HEADERS})

# Libraries
add_library(gemm_naive_lib STATIC Src/gemm_naive.cu)
add_library(gemm_opt_lib STATIC Src/gemm_opt.cu)
add_library(gemm_cpu_lib STATIC Src/gemm_cpu.c)
set_target_properties(gemm_naive_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gemm_opt_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gemm_cpu_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Executable
add_executable(benchmark Src/benchmark.cu)

# Set CUDA as the language for all targets using CUDA files
set_target_properties(gemm_naive_lib PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(gemm_opt_lib PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(gemm_cpu_lib PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(benchmark PROPERTIES LINKER_LANGUAGE CUDA)

# Set properties for benchmark executable
set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 52
)

# Link libraries to the benchmark executable
target_link_libraries(benchmark
  gemm_naive_lib
  gemm_opt_lib
  gemm_cpu_lib
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
)

# Add any necessary definitions
target_compile_definitions(benchmark PRIVATE
    -DUSE_FAST_MATH
)

# Add compile options
target_compile_options(benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)
