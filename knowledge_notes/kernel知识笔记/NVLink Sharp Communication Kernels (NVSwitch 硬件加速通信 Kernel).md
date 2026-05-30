## NVLink Sharp Communication Kernels (NVSwitch 硬件加速通信 Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

NVLink Sharp Communication Kernels 是 LongCat-Flash 推理系统中基于 NVSwitch 硬件加速的 collective communication 实现。传统 NCCL all-gather/reduce-scatter 依赖 GPU SM（Streaming Multiprocessor）执行数据搬移和 reduction，占用 SM 资源和内存带宽。NVLink Sharp (NVSwitch) 提供硬件加速的原语——broadcast (multimem.st) 和 in-switch reduction (multimem.ld_reduce)——在 NVSwitch 内部完成数据传输和规约，无需大量占用 GPU SM。

LongCat-Flash 使用 inline PTX assembly 直接调用这些硬件指令，实现仅需 4 个 thread blocks 的 all-gather/reduce-scatter kernel。性能在 4KB-96MB message size 全范围超越 NCCL 和 MSCCL++。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// NVLink Sharp All-Gather Kernel (简化 PTX 伪代码)

// 输入: local_data [local_size] 在 GPU i 上
// 输出: full_data [total_size] 在所有 GPU 上

// 1. 每个 GPU 发布自己的数据 (multimem.st)
for offset in range(0, local_size, chunk):
    // inline PTX: 存储本地数据到 NVSwitch 共享内存
    asm volatile(
        "multimem.st [%0], %1;"
        :: "l"(switch_addr + gpu_id * local_size + offset),
           "r"(local_data[offset:offset+chunk])
    );

// 2. 等待所有 GPU 完成发布 (barrier in NVSwitch)
__syncwarp();

// 3. 每个 GPU 读取所有数据 (load from NVSwitch)
for gpu in range(num_gpus):
    for offset in range(0, per_gpu_size, chunk):
        asm volatile(
            "ld.global.ca.b32 %0, [%1];"
            : "=r"(full_data[gpu * per_gpu_size + offset])
            : "l"(switch_addr + gpu * per_gpu_size + offset)
        );

// 仅需 4 thread blocks 执行——其余 SM/thread blocks 可用于计算
```

```
// NVLink Sharp Reduce-Scatter Kernel (multimem.ld_reduce)

// 输入: full_data [total_size] 在 GPU i 上 (部分数据)
// 输出: reduced_data [local_size] 在 GPU i 上 (对应分片的规约结果)

for offset in range(0, local_size, chunk):
    // inline PTX: NVSwitch 从所有 GPU 读取数据并执行 in-switch reduction
    asm volatile(
        "multimem.ld_reduce.add.f32 %0, [%1];"
        : "=f"(reduced_data[offset:offset+chunk])
        : "l"(switch_addr + gpu_id * local_size + offset)
    );
    // NVSwitch 内部: 从所有 GPU 的 offset 位置读取 →
    //   执行 FP32/BF16 reduction (add/min/max) →
    //   返回规约结果到请求 GPU
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. **PTX inline assembly**：需要手动管理寄存器分配和内存对齐。CUDA C++ 层面难以表达 multimem 操作，需要 PTX 级别控制。
2. **Uniform vs Non-uniform token distribution**：LongCat-Flash 的 kernel 支持均匀和非均匀两种 token 分布。非均匀场景（如 imbalanced EP routing）下需要额外的 metadata 传递（各 GPU 的 per-rank 数据量）。
3. **4 thread blocks 效率**：相比 NCCL 需要数十个 thread blocks，仅 4 个 block 大幅减少 SM 占用——剩余 SM 可用于计算（如 Dense FFN），提高 overlap 效率。
4. **适用场景**：(1) ScMoE Dense FFN 的 TP all-gather/reduce-scatter（intra-node NVLink）；(2) 与 MoE inter-node RDMA 通信并发执行（GPUDirect RDMA），最大化网络总利用率。
5. **硬件依赖**：仅在 Hopper 架构 (H100/H800) 且有 NVSwitch 的系统中可用（如 HGX H100 8-GPU baseboard）。论文未明确说明是否支持其他架构。

涉及论文标题：
- LongCat-Flash Technical Report
