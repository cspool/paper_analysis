## Two-Dimensional Resource Inefficiency in Tiled GPU Attention（GPU Tiled Attention中的二维资源效率低下）

术语是什么？通过联网搜索让回答具体和精准。
Two-dimensional resource inefficiency是PAT论文对已有attention kernel实现（包括query-centric和KV-centric）在共享prefix场景下GPU硬件资源利用问题的系统性刻画。由于decode attention使用tiling pipeline（将K/V沿sequence length切成小tile，在shared memory中执行QK^T和PV计算，同时异步预取下一tile），tile size（Q tile size m和KV tile size n）的选择直接决定了CTA的shared memory和register使用量，进而影响per-SM resident CTA concurrency和memory bandwidth utilization。当tile size不能适配实际workload时，会产生两维度的资源效率低下：(1) **Memory Waste (I_mem)**：当CTA内实际query数少于Q tile size m时，需要padding填充shared memory和register slots，浪费on-chip memory。例如flash attention固定m=64，当共享某prefix的query仅2个时浪费62个query slot。(2) **Execution Bubble (I_exe)**：不同CTA的KV length差异很大（从几十到数万tokens），固定KV tile size n使得某些CTA执行时间远长于其他，造成最后完成的CTA拖慢整个kernel，其余SM在tail stage空转。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在A100 GPU上的具体表现：

```
GPU硬件架构要素（以A100-SXM4-80GB为例）:
- 108 SM, 每个SM有192KB shared memory, 65536 registers (256 threads × 255 regs)
- 40MB L2 cache, 80GB HBM (1935 GB/s bandwidth)
- 每个SM最多resident CTA数取决于shared memory和register用量

I_mem (Memory Waste) 示例:
无prefix场景: batch 64, 每query独立CTA, 每个CTA q=1
使用FlashAttention固定m=64 → 每个CTA浪费63个query slot
  shared memory: Q tile = 64×128×2B = 16KB
  register: 实际1个query只用1/64的register, 但编译器仍分配全部
  → per-SM resident CTA数被迫降低, memory bandwidth无法打满

I_exe (Execution Bubble) 示例:
batch中CTA有KV length = [4096, 2048, 512, 256, 128, 64]
使用固定n=128:
  CTA_0 (4096): 32 tiles → ~32× tile_time
  CTA_5 (64):   1 tile  → ~1× tile_time
  SM执行时序:
  SM0: [CTA_0][CTA_0]...[CTA_0][CTA_0] ← 最后完成
  SM1: [CTA_1][CTA_1]...[CTA_1]       ← 早完成，空转
  SM2: [CTA_2][CTA_3][CTA_4][CTA_5]    ← 最早完成，长时间空转
  实际SM utilization = 平均执行时间/最长执行时间 < 60%

PAT的解决方案 (A100上):
Multi-tile kernel的11组可行配置降低I_mem:
  Q tile m ∈ {16, 32, 64, 128}, KV tile n ∈ {16, 32, 64, 128, 256}
  q=2 → m=16(浪费14) vs 固定m=64(浪费62) → 减少77% shared memory waste

Long-KV split + multi-stream降低I_exe:
  KV length > batch均值 → split into parts ≤ mean_kv_len
  Multi-stream: 不同tile config并行执行, 减少tail等待
  PTX profiling: execution bubble显著减少, SM idle时间压缩
```

关键硬件约束关系：
- 更大n → 更多in-flight data → 更高bandwidth utilization → 但更少CTA per SM → 更多bubble
- 更大m → 更多shared memory per CTA → 更少CTA per SM → lower concurrency
- Tile selector的核心trade-off是在bandwidth utilization和execution bubble间找到per-CTA最优平衡点

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
I_mem和I_exe的量化测量需要GPU profiling工具：NCU (NVIDIA Nsight Compute)测量global memory traffic（识别redundant loads）和SM occupancy（识别低利用率时间段）；PTX profiling可视化CTA execution timeline。PAT通过offline profiling+online selection解决两维度问题：offline在目标GPU上profiling每种(m,n)配置在不同KV length下的实际bandwidth utilization和latency来建立kernel equivalence set和decision tree；online则为每个CTA做constant-time tile lookup。该分析方法可推广到其他memory-bound GPU kernel的tile size选择问题，不限于attention。

涉及论文标题：
- PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

