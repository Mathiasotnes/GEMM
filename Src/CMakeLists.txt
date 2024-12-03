# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDA_INCLUDE_DIRS}
)

# Collect headers and source files
file(GLOB HEADERS *.h *.cuh)
file(GLOB SOURCES *.cpp)

# Group headers in IDEs
source_group("Headers" FILES ${HEADERS})

# Libraries
add_library(gemm_naive_lib STATIC gemm_naive.cu)
add_library(gemm_lib STATIC gemm.cu)
set_target_properties(gemm_naive_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gemm_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Executable
add_executable(benchmark benchmark.cu)

# Set properties for benchmark executable
set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 52
)

# Link libraries to the benchmark executable
target_link_libraries(benchmark
  gemm_naive_lib
  gemm_lib
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
