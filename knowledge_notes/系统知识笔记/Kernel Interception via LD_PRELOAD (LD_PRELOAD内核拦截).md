## Kernel Interception via LD_PRELOAD (LD_PRELOAD内核拦截)

术语是什么？

Kernel Interception via LD_PRELOAD（LD_PRELOAD内核拦截）是μShare实现非侵入式CUDA kernel参数修改的核心系统机制。在Linux userspace层面，通过LD_PRELOAD环境变量将μShare的共享库（.so文件）在CUDA runtime库之前加载，利用同名函数覆盖（function shadowing）劫持CUDA kernel launch函数（如cudaLaunchKernel、cublasSgemm）。拦截函数通过dlopen打开真正的CUDA动态库（libcudart.so、libcublas.so、libcudnn.so），使用dlsym获取原始函数的地址和输入参数（blocksize、gridsize、sharedMem、stream），在转发到原始函数前修改blocksize等参数，实现完全不修改kernel代码或GPU硬件的kernel-level控制。

从系统架构角度拆解术语：

μShare kernel interception的工作流程：

```
// 1. 编译时：μShare组件编译为.so shared library
//    kernel_interceptor.so  — 包含cudaLaunchKernel等wrapper函数
//    block_shaper.so        — 包含blocksize计算和shm_open/mmap逻辑

// 2. 运行时：通过LD_PRELOAD加载
//    LD_PRELOAD=kernel_interceptor.so:block_shaper.so python inference_server.py

// 3. PyTorch调用cudaLaunchKernel时，kernel_interceptor的wrapper先被调用：
extern "C" cudaError_t cudaLaunchKernel(
    const void *func, dim3 gridDim, dim3 blockDim,
    void **args, size_t sharedMem, cudaStream_t stream) {
    
    // Step 1: 获取原始CUDA函数指针 (仅在首次调用时)
    static cudaError_t (*original_launch)(...) = nullptr;
    if (!original_launch) {
        void *handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_LOCAL);
        original_launch = (cudaError_t(*)(...)) dlsym(handle, "cudaLaunchKernel");
    }
    
    // Step 2: 提取原始参数
    kernel_id = get_kernel_name(func);
    original_blocksize = blockDim.x;
    
    // Step 3: 通过共享内存与block_shaper通信
    int shm_fd = shm_open("/ushare_shaper", O_RDWR, 0666);
    kernel_params_t *shm = (kernel_params_t*) mmap(0, sizeof(kernel_params_t), 
        PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    
    // 写入kernel信息到共享内存
    shm->kernel_func = func;
    shm->blocksize = blockDim.x;
    shm->gridsize = gridDim.x;
    shm->intercept_time = get_current_time_ns();
    
    // Step 4: 等待block_shaper修改blocksize (60.35ns avg overhead)
    // block_shaper: 读取slack → 计算half-plus → 写回shm->blocksize
    while (!shm->ready) { _mm_pause(); }
    
    // Step 5: 使用修改后的参数调用原始函数
    dim3 newBlockDim = {shm->blocksize, blockDim.y, blockDim.z};
    return original_launch(func, gridDim, newBlockDim, args, sharedMem, stream);
}

// 4. 对unmodifiable kernel (cuBLAS wrapper):
extern "C" cublasStatus_t cublasSgemm(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float *alpha, const float *A, int lda,
    const float *B, int ldb, const float *beta, float *C, int ldc) {
    
    // 类似拦截，但blocksize参数隐藏在wrapper内部，不可直接修改
    // 改用time-shifted launching：通过共享内存通知block_shaper
    // block_shaper决定是否延迟启动或直接转发
    static cublasStatus_t (*original_sgemm)(...) = nullptr;
    if (!original_sgemm) {
        void *handle = dlopen("libcublas.so", RTLD_LAZY | RTLD_LOCAL);
        original_sgemm = (cublasStatus_t(*)(...)) dlsym(handle, "cublasSgemm");
    }
    // ... time-shifted launch logic via shm_open/mmap ...
}
```

术语一般如何实现？如何使用？

μShare的kernel interception实现：
1. **共享库编译**：C++实现，编译为.so文件（使用-fPIC -shared）
2. **LD_PRELOAD机制**：Linux dynamic linker在加载程序依赖库前先加载LD_PRELOAD指定的库，同名符号优先使用preload版本，实现透明拦截
3. **dlopen+dlsym**：POSIX标准API，dlopen以RTLD_LAZY模式打开CUDA库获得handle，dlsym通过symbol name查找原始函数地址
4. **shm_open+mmap**：POSIX共享内存API，用于kernel_interceptor与block_shaper之间的高速通信（60.35ns avg per kernel）
5. **kernel分类**：通过拦截函数的signature区分modifiable (cudaLaunchKernel, 51.63% executions) vs unmodifiable (cuBLAS/cuDNN wrapper functions, 48.37%)
6. **适用性**：论文仅在PyTorch 2.2.0 + CUDA 11.8/12.1环境下验证，但原理上适用于任何使用标准CUDA runtime API的推理框架（TVM/TensorRT/TF-Serving）
7. **安全性**：LD_PRELOAD在public cloud环境下可行，因为不修改任何系统文件或内核代码，仅影响当前进程的symbol resolution

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs
