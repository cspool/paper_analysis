## Group GEMM for MoE Inference

术语是什么？
Group GEMM（Grouped General Matrix Multiplication）是MoE推理中将多个expert的矩阵乘法合并到单个统一kernel launch中并发执行的技术。每个expert的输入token batch被分配为group内的一个独立子任务（sub-task），所有子任务在同一kernel grid内并行执行。核心优势：(1) 减少kernel launch次数——所有active experts的FFN计算在单次kernel launch中完成；(2) 自然适应动态expert activation pattern——不同token激活不同expert集合，group内子任务数动态变化；(3) 通过offline profiling为不同activation pattern选择最优kernel tile sizes。

从kernel调度角度拆解术语：
以MoDES在Qwen3-VL-MoE-30B-A3B-Instruct上执行MoE FFN为例（128 experts/layer, k=8，跳过88%后平均约1 active expert/token）：

```
// Router Kernel (fused with thresholding, single launch):
r = Router(x)                              // [128] logits
π = softmax(r)
topk = topk_indices(π, k=8)                // 8 candidate expert IDs
// Branch-free thresholding:
for i in topk:
    s_i = α̃^{(l)} · π[i]
    topk[i] = (s_i < τ) ? M+1(sentinel) : topk[i]

// MoE Dispatch (filter sentinel entries):
for each expert_id in topk:
    if expert_id != M+1:
        dispatch token → expert_id's input buffer

// Group GEMM (single kernel launch):
GroupedMatMul(
    inputs: [X_e1, X_e2, ...],             // per-expert token batches
    weights: [W_gate_e1, W_up_e1, ...],    // expert FFN weights
    outputs: [gate_e1, up_e1, ...]
)
// Each expert's GEMM as independent sub-task
// Tile sizes: offline profiled for current activation pattern

// SiLU + Down Projection (also Group GEMM):
GroupedMatMul(
    inputs: [SiLU(gate_i) ⊙ up_i, ...],
    weights: [W_down_e1, W_down_e2, ...]
)

// Gather outputs, weighted sum:
for each token:
    y = Σ π_i · ExpertOutput_i
```

术语一般如何实现？如何使用？
MoDES使用custom CUDA kernel实现Group GEMM。配合offline profiling——对different representative activation patterns进行grid search确定最优tile sizes（BLOCK_M/N/K, num_warps, etc.）。Sentinel expert filtering在dispatch阶段即完成——跳过expert不分配输入buffer，不进入Group GEMM。Group GEMM性能高度依赖workload distribution的规律性——expert skipping导致的不规则子任务大小可能降低GPU利用率，但offline profiling和tile size tuning可最大化throughput。Kernel内仅需少量warp-level元素操作（masked comparison + sentinel filtering），overhead <1% of total compute time。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

---
