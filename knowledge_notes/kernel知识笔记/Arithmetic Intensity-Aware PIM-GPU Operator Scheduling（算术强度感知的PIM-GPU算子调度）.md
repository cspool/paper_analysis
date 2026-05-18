## Arithmetic Intensity-Aware PIM-GPU Operator Scheduling（算术强度感知的PIM-GPU算子调度）

术语是什么？通过联网搜索让回答具体和精准。
Arithmetic Intensity-Aware Operator Scheduling是SADDLE提出的一种运行时operator-to-device动态映射机制，用于PIM+GPU异构系统上的speculative decoding。核心思想：speculative decoding中variable draft lengths和changing effective micro-batch sizes会动态改变operator的arithmetic intensity (FLOPs/Byte)，使离线静态mapping失效（如SpecPIM的offline genetic algorithm-based mapping）。SADDLE的Scheduler在运行时快速估算operator CI，与预标定的PIM compute-bound和GPU memory-bound ridge point比较，动态决定operator在PIM或GPU执行。初始固定映射：DLM attention → PIM（每iteration 1 token/request，算术强度极低）；TLM FC → GPU（Shared Pool聚合token后变为compute-intensive GEMM）。动态remap对象：DLM FC和TLM attention。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。
Scheduler的per-iteration决策伪代码：
```
// 预标定阶段（offline）
PIM_peak_compute = measure_pim_tflops()     // e.g. HBM-PIM PE array peak
PIM_peak_bandwidth = measure_pim_bw()       // internal bandwidth ~144 TB/s
GPU_peak_compute = measure_gpu_tflops()     // A100 Tensor Core peak
GPU_peak_bandwidth = measure_gpu_bw()       // HBM2e bandwidth ~1.5 TB/s

PIM_ridge = PIM_peak_compute / PIM_peak_bandwidth   // ~1 FLOP/Byte
GPU_ridge = GPU_peak_compute / GPU_peak_bandwidth    // ~100 FLOP/Byte

// === 每次prediction后: schedule DLM FC ===
function schedule_DLM_FC():
    active_reqs = count(H_t > τ for all requests)
    eff_bs = active_reqs
    
    // DLM FC: [eff_bs, d_model] × [d_model, d_model]
    FLOPs = 2 * eff_bs * d_model^2
    Bytes = (eff_bs * d_model + d_model^2) * 2  // FP16
    CI_dlm_fc = FLOPs / Bytes  // ≈ eff_bs (approx, when eff_bs << d_model)
    
    if CI_dlm_fc < PIM_ridge:   return "PIM"
    else:                        return "GPU"

// === 每次verification前: schedule TLM Attention ===
function schedule_TLM_attention():
    total_tokens = SharedPool.count()
    FLOPs_attn = 4 * total_tokens * d_head^2
    Bytes_attn = 2 * total_tokens * d_head * 2  // KV reads
    CI_attn = FLOPs_attn / Bytes_attn  // ≈ 2 * d_head (per-token)
    
    if CI_attn > GPU_ridge:  return "GPU"
    else:                     return "PIM"
```

关键动态：当micro-batch从12请求降至4请求（短draft请求先完成），DLM FC的CI从~12降到~4 → 从GPU bandwidth-bound转到PIM compute-bound → optimal target从GPU变为PIM。当draft length从1增至8，TLM attention CI提升 → GPU超越PIM（即使operator still memory-bound on GPU） → optimal target从PIM变为GPU。

术语一般如何实现？如何使用？
SADDLE Scheduler在Manager中实现：offline预标定每个device的peak compute和bandwidth（一次性），runtime用活跃请求数和Shared Pool token count快速估算CI→与预标定阈值比较→决定operator映射。与SpecPIM的offline genetic algorithm/MCTS（执行前一次性mapping，推理中不变）和PAPI的dynamic profiling形成对比：SADDLE用轻量CI估算替代完整profiling，仅需预标定阈值和简单代数运算。消融：动态scheduling使SADDLE吞吐再提升1.13×（over static mapping），PIM ops占比从9.51%升至14.89%、GPU从90.49%降至85.11%，整体吞吐提升1.21×。

涉及论文标题：
- Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

