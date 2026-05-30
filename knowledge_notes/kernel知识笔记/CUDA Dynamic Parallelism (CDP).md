## CUDA Dynamic Parallelism (CDP)

术语是什么？
CUDA Dynamic Parallelism（CDP，CUDA 动态并行）是 NVIDIA 自 Kepler 架构（SM 3.5+）起支持的 GPU 硬件特性，允许 GPU kernel 在设备端动态发射子 kernel，无需 CPU 参与。父 kernel（parent kernel）可以通过 `cudaLaunchDevice` 或直接调用 `kernel<<<grid, block, smem, stream>>>(args)` 发射子 kernel。CDP 天然支持父 kernel 与多个子 kernel 之间的数据依赖（子 kernel 在父 kernel 发射后执行，父 kernel 可等待子 kernel 完成），但不支持一个 kernel 依赖多个父 kernel 的向无环图（DAG）依赖模式——这在 Deep RL 仿真和动态 DNN 中很常见。

从kernel调度角度拆解术语：
```
// CDP的基本使用模式
__global__ void parent_kernel(float* data, int N) {
    // ... 计算第一阶段 ...
    
    // 动态发射子kernel (无需CPU参与)
    child_kernel<<<grid, block, 0, 0>>>(data, N);
    // 子kernel在父kernel的stream中执行，默认串行
    
    cudaDeviceSynchronize();  // 等待所有子kernel完成
    
    // ... 使用子kernel的结果继续计算 ...
}

// CDP的限制:
// 1. 仅支持父子依赖 (1 parent → N children)
// 2. 不支持: child需要等待多个parent kernel
//    (这在Dynamic DNN和物理仿真中很常见)
// 3. 嵌套深度受限于设备runtime堆大小
```

CDP 的典型应用：(1) 基于运行时数据决定是否/如何发射子 kernel（如根据数据稀疏性跳过计算）；(2) 递归算法（如快速排序、八叉树遍历）；(3) 减少 GPU-CPU 往返（父 kernel 不返回 CPU 即发射子 kernel）。但在 ACS 的目标 workload（Deep RL 仿真、动态 DNN）中，kernel 间的依赖是多对多的——一个 kernel 可能依赖多个 upstream kernel，CDP 无法表达此类依赖关系。ACS 通过调度窗口的依赖管理解决了这一限制。

术语一般如何实现？如何使用？
CDP 需要：(1) GPU compute capability ≥ 3.5；(2) 编译时链接 `cudadevrt` 库（`nvcc -rdc=true -lcudadevrt`）；(3) 设置设备 runtime 堆大小（`cudaDeviceSetLimit(cudaLimitMallocHeapSize, size)`）。CDP 的开销包括子 kernel launch 延迟（几百 ns 到几 μs）和额外的设备端内存使用（runtime 堆、pending launch buffer）。ACS 论文评估了 CDP 在目标 workload 上的适用性，结论为不适用于有多对多依赖关系的不规则计算图。

涉及论文标题：
- ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs
