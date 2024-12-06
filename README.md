# GEMM
General Matrix Multiplication (GEMM) optimization in Cuda.

### Notes
- I'm using square matrixes
- I'm setting alpha and beta to 1 and C to 0 to simplify
- The functions include the memory allocation
- When using the profiler at stream shmem I saw that it launched hundreds of different kernel instances. The other
  methods only had a single kernel instance.
- CuBLAS uses 3D grid (8, 16, 5), and a blockSize of (128,1,1). When I tried to use this in my shmem implementation
  I got the wrong answer, but it reduced the amount of cycles.

### Talking points:

1. Introduce problem:
    - Simple matrix multiplication variation of GEMM.

2. Go through implementations:
    - CPU:              To compare with GPU implementation.
    - naive:            Basic implementation of parallell matrix multiplication.
    - shmem:            Utilizing shared memory in the same way as explaned in lecture (tile-based).
    - stream:           Tried to split the A-matrix into different tiles. Unsuccesfully.
    - stream_shmem:     Stream combined with shared memory.
    - cublas:           CuBLAS library wrapper.

3. Go through results:
    - results tile/block size 16:
        - CPU was fastest on the small matrixes because it doesn't have to copy memory.
        - Naive and shmem were close on all the sizes, but shmem turned out better when the size increased.
        - CuBLAS excelled when the sizes became large enough.
    - results tile/block size 32:
        - naive and shmem were a lot faster on 2048, but slower on 1024. Kinda surprising since 32x32=1024.
    - shmem nsight compute analysis:
        - Close to zero bank conflicts.
        - LSU bottleneck (load and store operations).
    - cublas nsight compute analysis:
        - Not bottlenecked in the same way as shmem by LSU.
    - Profile summary:
        - CuBLAS dimension
        - CuBLAS using a lot less cycles.


