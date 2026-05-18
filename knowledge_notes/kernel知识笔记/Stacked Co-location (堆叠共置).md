## Stacked Co-location (堆叠共置)

术语是什么？

Stacked Co-location（堆叠共置）是指NVIDIA GPU硬件调度器的默认行为：当一个CUDA kernel被launch后，其多个blocks被顺序调度到SM cores，同一kernel的blocks因具有相同的资源需求而被"堆叠"在相同的SM内。由于kernel的blocksize通常远小于SM的总thread容量（如A40: 1536 threads, 默认blocksize 512-1024），一个SM可以同时容纳多个同kernel blocks，但这些blocks的hardware resource需求完全相同，导致SM内仅一种硬件资源被充分利用，其余资源闲置。"Stacked"一词强调同kernel blocks在SM内的垂直堆叠关系。在μShare的实验中，vectorized kernel (dominant LDST, blocksize 1024) 和 roll kernel (dominant INT32, blocksize 512) 虽在不同CUDA stream上并发launch，但vectorized kernel的所有blocks先占满所有SM并执行完毕，roll kernel的blocks才被调度，无法在SM内实现不同kernel的并行。

从kernel调度角度拆解术语：

Stacked co-location的执行过程（以PyTorch launch two kernels on NVIDIA A40为例）：

```
// Stage 1: Launch — PyTorch launch两个kernel到不同CUDA stream
cudaLaunchKernel(vectorized_kernel, gridDim, 1024, ..., stream1);
cudaLaunchKernel(roll_kernel, gridDim, 512, ..., stream2);

// Stage 2: Dispatch — GPU hardware dispatch unit按顺序调度block
// vectorized_kernel: 每个block 1024 threads
//   SM0: block_0(1024 threads) — 剩余512 threads
//   SM0: block_1(1024 threads) — 无法，因为 512 < 1024
//   但可以：SM0: block_0, SM1: block_1, ...
//   当线程数超过总thread容量时：SM0: block_0 + block_84, SM1: block_1 + block_85, ...
//   → 每个SM内都放着同kernel的两个block

// roll_kernel的blocks全部排队等待，直到vectorized_kernel blocks接近完成
// 原因：当kernel总线程数 > GPU总thread容量(129024)，需等待前序blocks释放SM资源

// Stage 3: Execution — SM内
// vectorized_kernel blocks在SM0内：
//   LDST: 58.02% active (layer normalization的主要操作)
//   FP32: 13.43%, INT32: ≈0%, FP64: 0%, SFU: 11.03%, Tensor: 0%
//   → "1 more, 5 less" 模式
```

μShare的实验数据显示：在max batch下，61.85%的kernel执行时线程数超过GPU总容量（129024 threads），这些kernel占70.83%的总执行时间，因此stacked co-location是主要瓶颈。6802次kernel执行的统计显示NVIDIA-SMI报告81.16%利用率，但Nsight Compute仅报告9.28% low-level hardware利用率。

术语一般如何实现？如何使用？

Stacked co-location是NVIDIA GPU硬件调度器的固有行为，由闭源的hardware dispatch unit实现。GPU的left-over scheduling策略：只要SM的剩余thread capacity ≥ blocksize，block就可以被调度到该SM。由于同一kernel的所有block有相同blocksize和资源需求，硬件scheduler自然倾向将它们填入相同的SM集合。现有系统（INFless、Orion、MPS、MIG）在inter-SM层面做spatial/temporal sharing，但无法解决intra-SM的stacked co-location问题。避免stacked co-location的方法包括：kernel fusion（Tacker/T3/Rammer/COMBO，将互补kernel合并为一个）和persistent kernel（ISPA/Plasticine/Elastic kernel，空核驻留），但这些都需要侵入性修改。μShare选择非侵入路径：通过修改kernel launch参数（blocksize）间接影响硬件scheduler的decision，实现从stacked到scattered co-location的转变。

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

