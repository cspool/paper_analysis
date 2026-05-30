# IV. IMPLEMENTATION

## A. Overall Implementation

μShare uses PyTorch [40] as the GPU inference framework. The components of μShare are compiled as shared libraries (i.e., .so files) and loaded into the PyTorch process using the LD\_PRELOAD environment variable to interact with it. To support custom kernel blocksize, we set PyTorch's blocksize limit (i.e., defined in the C10\_LAUNCH\_BOUNDS (blocksize) macro) to be consistent with the CUDA limit, i.e., 1024.

In particular, the *kernel interceptor* uses the <code>dlopen()</code> function from the <code>libdl[13]</code> library to open CUDA's dynamic link libraries (e.g., CUDA's <code>libcublas.so()</code> at runtime and return handles to the CUDA's kernel launch functions. It then uses the <code>dlsym()</code> function from the <code>libdl</code> library to obtain the addresses of the library functions and their input parameters based on the handles. The <code>block shaper</code> uses the <code>shm\_open()</code> function from the <code>libc[16]</code> library to create a shared memory area. The <code>block shaper</code> exposes an interface to the <code>kernel interceptor</code>, with <code>kernel\_process()</code> used for uploading parameters such as kernel blocksize. This interface uses the <code>mmap()</code> function from the <code>libc</code> library to map and modify values in shared memory. After modifying the uploaded kernel parameters, the <code>block shaper</code> returns the

parameters to the address of the shared library function obtained by the dlsym() function, restoring kernel execution.

#### B. Support for Different NVIDIA GPUs

uShare provides support for different NVIDIA GPUs. Since the number of threads per SM affects the blocksize configuration strategy,  $\mu Share$  categorizes recent GPUs into two types: (1) **GPUs with 1,536 threads per SM:** Such as NVIDIA A40, RTX 4090, and RTX 3080 Ti. Under the CUDA constraint, where the maximum blocksize is 1024, a half-plus block can be deployed on these GPUs with 1,536 threads per SM. Moreover, setting the blocksize as a multiple of 32 (i.e., the number of threads in a warp) can avoid thread resource fragmentation. Therefore, the range of the half-plus blocksize b is given by:  $\{b \mid 1536/2 < b \le 1024, b \equiv 0 \pmod{32}\}$ . For example, the minimum blocksize is 800.

To determine a specific blocksize b, the block shaper reads the kernel's launch slack value from shared memory using the mmap () function. If the slack value is negative, b is set to the previous kernel's blocksize incremented by 32; otherwise, it defaults to 800. The block shaper then injects b into the kernel's launch address intercepted using the dlsym() function, thereby resuming kernel execution.

(2) GPUs with 2,048 threads per SM: Such as NVIDIA A100, A800, and H200. The half-plus strategy is not suitable for these GPUs. Even with a maximum blocksize of 1024, the CUDA scheduler still performs stacked co-location of two large blocks within the SM cores.

In this case, the optimal configuration is to set the blocksize to 1/3-plus of 2,048. This allows the CUDA scheduler to deploy two large 1/3-plus blocks from the same kernel within one SM. Subsequently, since the remaining threads in the SM are less than 1/3 of 2,048, it is not possible to deploy another large block. These remaining threads can be allocated to small blocks from another kernel with complementary hardware resource demands, thereby achieving scattered co-location.

Thus, the range of the 1/3-plus blocksize b is given by:  $\{b \mid 2048/3 < b \le 1024, b \equiv 0 \pmod{32}\}$ . For example, the minimum blocksize is 704. After determining the range of blocksize, the subsequent operations are the same as in the previous case.

