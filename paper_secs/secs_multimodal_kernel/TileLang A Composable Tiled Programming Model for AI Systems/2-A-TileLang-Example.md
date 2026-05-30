# 2 A TileLang Example

Existing machine learning compilers that separate scheduling from computation, such as TVM, require users to explicitly distinguish between computation and scheduling. Additionally, users must manually register new tensor instructions and specify buffer layouts to achieve optimal performance. However, writing and understanding scheduling programs remains challenging. Although modern frameworks like Triton allow users to focus on tile-level programming, their dataflow representation is often unclear, and they require the use of certain workarounds—such as masked conditional loads—or hardware-specific features like Tensor Memory Accelerator (TMA). While frameworks such as ThunderKitten abstract programs into a tile-granular combination of load,

<span id="page-2-0"></span><sup>1</sup><https://github.com/tile-ai/tilelang>

```
a⊤.prim_func
 1 import tilelang.language as T
                                                                                                                         Matmul(A: T.Tensor, B: T.Tensor, C: T.Tensor):
                                                                                                                         Buffer Allocation
shared = T.decl_buffer((4096,), "float16", "shared")
shared = T.decl_buffer((4096,), "float16", "shared")
      M, N, K = 1024, 1024, 1024
block_M, block_N, block_K = 128, 128, 32
                                                                                                                                                                                                                                          global void main kernel(
                                                                                                                                                                                                                                            global___void main_kernel(
const __half* __restrict__ A,
const __half* __restrict__ B,
__half* __restrict__ C,\nint m, int n, int k){
       num stages =
                                                                                                                      "Lower tragment butter to threads
C_local = T.decl_buffer((128), "float16", "local")
                                                                                                                      # Thread binding
bx = T.thread_binding(128, "blockIdx.x")
by = T.thread_binding(128, "blockIdx.y")
tid = T.thread_binding(128, "threadIdx.x")
     def Matmul(A: T.Tensor, B: T.Tensor, C: T.Tensor):
   with T.Kernel(
                                                                                                                                                                                                                                             extern __shared__
half* AShared = ...;
                                                                                                                                                                                                                                                                                   align (1024) uchar buf dyn shmem[];
         with T.Kernel(
   N // block_N, M // block_M, threads=threads
) as (bx, by):
                                                                                                                                                                                                                                             half* BShared = ...
float C_local[128];
              # Buffer Allocation
A_shared = T.alloc_shared(block_M, block_K)
B_shared = T.alloc_shared(block_K, block_N)
C_local = T.alloc_fragment(block_M, block_N)
                                                                                                                     # ... Initialize C_location in T.unroll(128):
                                                                                                                                                                                                                                            // Main Loop with Pipeline
tl::cp_async_gs<16>(AShared, A);
tl::cp_async_gs<16>(BShared, B);
tl::cp_async_commit();
for (int ko * 0; ko < 31; ++ko) {
    tl::cp_async_wait<0>();
    _syncthreads();
                                                                                                                    # Main Loop with Expanded Pipe
for i in T.unroll(4):
   T.cp_async(A, A_shared, 16)
              # Initialize C_local
T.clear(C local)
                                                                                                                      # ... Copy B to B_sha
T.cp_async_commit(0)
             # Main Loop with Pipeline Annotation
for k in T.Pipelined(K // block_K, num_stages):
T.copy(A[by * block_M, k * block_K], A_shared)
T.copy(B[k * block_K, bx * block_N], B_shared)
T.gemm(A_shared, B_shared, C_local)
                                                                                                                                                                                                                                                _synctnreads();
tl::gemm_s<128, 128, 32, 2, 2>(AShared, BShared, C_local);
_syncthreads();
tl::cp_async_gs<15>(AShared, A);
tl::cp_async_gs<15>(BShared, B);
                                                                                                                          T.cp_async_wait(0)
T.gemm_ss(A_shared, B_shared, C_local,
                                                                                                                         for i in T.unroll(4):
                                                                                                                                                                                                                                           }
tl::cp_async_wait<0>();
__syncthreads();
tl::gemm_ss<128, 128, 32, 2, 2>(AShared, BShared, C_local);
                                                                                                                          T.cp_async(A, A_shared, 16)

#... Copy B to B_shared

T.cp_async_commit(0)

... Compute the last stage
      program = Matmul(A, B, C)
                                                                                                                                                                                                                                             // Copy C_local to C
31 kernel = tilelang.compile(program, target="cuda")
                                                                                                                     T.conv(C local, C)
                                                                                                                                                                                                                                                  (c) The generated CUDA code
      (a) An example TileLang Program
                                                                                                                      (b) Intermediate Tensor IR
```

Fig. 1. An example TILELANG program and the corresponding lowered ir and generated cuda c code. The code snippets are simplified for demonstration purposes.

compute, store, and synchronization operations, their dataflow remains insufficiently transparent, limiting users' ability to apply further optimizations. Moreover, with the widespread adoption of Python-based deep learning frameworks [3, 22], manually translating models into C++ for optimization is impractical. Therefore, in designing Tilelang, we emphasize three key principles: (1) **Pythonic design**, which integrates seamlessly with the Python ecosystem, providing a familiar coding experience and reducing the learning curve; (2) **Dataflow-centric**, which enables users to focus primarily on dataflow while abstracting away low-level scheduling complexities. It decouples scheduling aspects—such as thread binding, memory layout, tensorization, and pipelining—from dataflow, encapsulating them as a set of customizable annotations and primitives to enhance both programmability and maintainability; and (3) **Composability**, ensuring that kernels, primitives, and scheduling strategies can be seamlessly combined to construct complex designs.

In the following, we implement a general matrix multiplication (GEMM) kernel in TILELANG to illustrate its basic syntax and demonstrate how it enhances productivity. As shown in Figure 11(a), the implementation begins by defining the GEMM kernel's inputs and outputs (Line 8), specifying their shapes and data types. Subsequently, we initialize the kernel context (Lines 9-11), which determines the grid size and total number of threads, followed by the kernel body (Lines 12–27), which includes on-chip memory allocations and data flow management. Since TileLang is a Pythonembedded programming language, it supports all imperative constructs of Python (e.g., if-else, for, and while), with the key distinction that users must provide explicit type annotations for function arguments and variable declarations. This requirement arises due to Python's dynamic typing, which may not be inherently suitable for device code generation (e.g., CUDA/HIP), where static data types are essential for determining precise data bitwidths. In TILELANG, type annotations explicitly define element types and tensor shapes, ensuring correctness and efficient code generation. Additionally, TileLang allows explicit memory allocation, providing greater control over data placement and access patterns. In the given implementation, TILELANG employs T. alloc\_shared to store submatrices of A and B in shared memory, while T.alloc\_fragments is used to allocate accumulators in register files at the block level. Furthermore, the use of pipelined execution (T. Pipelined) enables the overlapping of memory transfers with computation, effectively hiding memory latency and improving overall throughput. The T. gemm operation leverages NVIDIA

CUTLASS or manually written HIP code to perform tile-level matrix computation efficiently. By automating low-level scheduling and synchronization, TileLang allows developers to focus on algorithm design rather than hardware-specific optimizations, thereby enhancing productivity while maintaining computational efficiency.

Finally, we invoke tilelang.compile (Line 31) to lower the tilelang program into an intermediate representation (IR), as illustrated in Figure 11(b). This IR is then further compiled into an executable, generating the final optimized code, as shown in Figure 11(c).

