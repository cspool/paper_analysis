## Computation Affinity (计算亲和性)

术语解释
Computation Affinity 是 APTMoE 提出的概念，描述一个 expert 的计算工作负载对 GPU 或 CPU 的"适配程度"。由于 MoE 的 expert 热度偏斜，不同 expert 的输入 token 数量差异巨大，导致其计算强度（computational intensity）不同——高 token 数的 expert 适合 GPU 的并行计算能力（compute-bound），低 token 数的 expert 在 CPU 上执行时间与 GPU 可比甚至更优（memory-bound 时 CPU 更友好）。

术语是什么？
计算亲和性的核心判断依据：对于给定数量的输入 token，比较 expert 在 GPU 上的端到端时间（计算+数据移动）vs CPU 上的计算时间（无需数据移动，因 expert 权重留在 host memory）。当 token 数较少时，GPU 的计算优势被 kernel launch overhead 和 PCIe 数据传输时间抵消，CPU 就地计算反而更高效。

从算法pipeline角度拆解术语。
```
# Profiling 阶段（离线）
for each expert_config:
    for num_tokens in [1, 2, 4, 8, ..., max_tokens]:
        t_gpu = profile(expert.forward, device='cuda', input_tokens=num_tokens)
        t_cpu = profile(expert.forward, device='cpu', input_tokens=num_tokens)
        t_load = profile(cudaMemcpy, host_to_device, size=expert_size)
        lookup_table[expert][num_tokens] = {gpu_time, cpu_time, load_time}

# Runtime 决策（Equation 1）
def decide_allocation(experts, predicted_popularity):
    sorted_experts = sort(experts, key=predicted_popularity, reverse=True)
    cumulative_cpu_time = 0
    cumulative_load_time = Load_MHA + Load_Gate
    for expert in sorted_experts:  # 从高热度到低热度
        cumulative_load_time += lookup_table[expert][num_tokens].load_time
    for expert in reversed(sorted_experts):  # 从低热度到高热度
        cumulative_cpu_time += lookup_table[expert][num_tokens].cpu_time
        R = cumulative_cpu_time / cumulative_load_time
        if R >= 1:
            break  # 从此处及以上热度的 expert 在 GPU 执行
        allocate_to_cpu(expert)
```
关键发现（Figure 6）：当 token 数量 < ~64 时，A800 GPU 的 expert 计算时间与 Intel Xeon Gold 6348 CPU (28 cores) 可比；token > 256 时 GPU 优势显著。且低热度 expert 无法 saturate CPU 核心（减少核心数影响小），适合 C1+G4（7核/进程）场景。

术语一般如何实现？如何使用？
- 通过 PyTorch profiler 记录单层单 expert 在不同 token 数下的 forward/backward 时间
- Lookup table 在 static 阶段生成，runtime 阶段查表
- 适用于 batch size 固定、sequence length 固定的 fine-tuning 场景
- 需要权衡：expert 尺寸越大（如 MoE-M vs MoE-S），CPU 执行效率越低，affinity 效益递减

涉及论文标题：
- APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes
- Accelerating Distributed MoE Training and Inference with Lina

**Lina 的 Expert Popularity Estimation 方法**：
与 APTMoE 使用 Predictor 模块（learned model）不同，Lina 采用 **profiling-based statistical estimation**：
1. **Expert Selection Pattern**: 发现 tokens 在相邻 MoE layers 中选择同一 expert 的倾向性——选定同一 expert 的 tokens 在下一层中选择 top-1 same expert 的比率达 41.94%（k=2 时 54.59%）
2. **Sample Path Profiling**: 在 training 阶段采集 load balancing loss 稳定后的 expert selection results，按 sample path（从 layer i-l 到 layer i 的 expert 序列）分组，为每个 sample path j 计算到 layer i+1 的 expert 分布 `Ψ_j^{i+1}`
3. **Online Estimation**: inference 时对每个 batch，从 layer l 开始，根据每个 token 的 sample path j(t) 查找对应的 `Ψ_{j(t)}^{i+1}`，取 top-k expert 的概率 `P(e)` 作为 popularity estimate
4. **Resource Allocation Formula**: `n_e = N × Σ_t P_{j(t)}(e) / N_t`（expert e 应占设备数比例）
5. **Accuracy**: path length l=3 时 estimation accuracy 60.4%（Transformer-XL）和 63.5%（BERT-Large），l=6 时可达 71.4%

---
