## Expert Popularity in MoE Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Popularity（专家热门度）是 MoE 模型中各 expert 被不同输入 token 激活的频率分布。在 Mixtral-8x7B 等 MoE 模型中，不同 expert 学习不同的语言模式或 token 特征，导致某些 expert（如学习常见句法结构的 expert）被显著更频繁地激活。Fiddler 通过 offline profiling 量化了这种分布：在 256 个 expert 中 popularities 均值=0.71（相对于最热门 expert 的比值），std=0.08，25th percentile=0.67，75th percentile=0.76，最低值=0.22。分布相对均衡但存在足够差异，使得热门度导向的 GPU 放置比随机放置提升 3-5 个百分点的 hit rate。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Fiddler 的 expert popularity profiling 流程：

```
// Offline profiling (一次性, 使用 calibration data)
// 输入: MoE model, calibration dataset (ShareGPT)
// 输出: popularity[layer][expert] ∈ [0, 1]

for each sample in calibration_data:
    hidden_states = model.embed(sample)
    for layer in 0..31:
        gate_scores = softmax(W_gate[l] @ hidden_states)  // [tokens, 8]
        top2_indices = topk(gate_scores, k=2)              // [tokens, 2]
        for token_t in range(num_tokens):
            for idx in top2_indices[token_t]:
                activation_count[layer][idx] += 1

// 归一化 (vs 最热门 expert)
for layer in 0..31:
    max_count = max(activation_count[layer])
    for expert in 0..7:
        popularity[layer][expert] = activation_count[layer][expert] / max_count

// 全局排序 (所有 256 experts 统一比较)
all_experts = [(l, e, popularity[l][e]) for l in 0..31 for e in 0..7]
all_experts.sort(key=lambda x: -x[2])  // 降序

// GPU placement: select top-N_gpu
gpu_experts = set(all_experts[:N_gpu])
```

Fiddler 的 heat map 可视化（Figure 8, Appendix C）显示：
- 大部分 expert 的 popularity 在 0.6-0.9 之间（相对均衡）
- 仅 15/256 expert 的 popularity < 0.6
- 27/256 expert 的 popularity > 0.8
- 最低 popularity=0.22（某 expert 激活次数仅为最热门 expert 的 22%）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **Profiling data**：Fiddler 使用 ShareGPT 对话数据集；论文在 LMSYS-Chat-1M 上验证了跨数据集的鲁棒性
- **Calibration 效率**：offline profiling 仅需一次（每个 calibration sample forward pass 一次），不增加推理运行时开销
- **假设依据**：Expert 选择基于 token 语义/句法特征，该假设在 Mixtral 论文和 OpenMoE 论文中得到验证——expert popularity 在不同下游 domain 间近乎 universal
- **与其他 expert 选择方法的对比**：
  - Expert LRU Cache（Mixtral-Offloading）：runtime 动态调整，适合 expert locality 较强的场景
  - Expert Popularity（Fiddler）：init-time 固定，无 runtime 维护开销，适合 popularity 分布稳定的场景
  - 两者正交互补——可同时使用
- **局限性**：若模型 weight 更新且 expert specialization 改变，需重新 profiling

HarMoEny 的关键发现：Expert popularity skew 是**动态的（dynamic）**且**batch 间剧烈波动**的。对 Qwen MoE（60 experts）和 Switch128（128 experts, bookcorpus）的分析（Figure 1）表明：(1) 偏斜随层深累积——层 0 仅 3/128 expert 接收平均 19% token，最后层 3 expert 接收 60%；(2) 偏斜随输入 domain 变化——medical vs programming prompts 产生完全不同的 expert activation pattern；(3) Batch 间波动——连续 batch 间 throughput 可下降 37.6%。这一动态性使得 profiling-based 方案（ExFlow 需 8.5min profiling for Switch128, 45min for Qwen）完全失效——profiling 时间远超 batch 处理时间（289ms）。HarMoEny 通过 **online token rebalancing**（无需 profiling，per-batch adaptation）和 **async expert prefetching** 解决此动态偏斜。

涉及论文标题：
- Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
- HarMoEny: Efficient Multi-GPU Inference of MoE Models
