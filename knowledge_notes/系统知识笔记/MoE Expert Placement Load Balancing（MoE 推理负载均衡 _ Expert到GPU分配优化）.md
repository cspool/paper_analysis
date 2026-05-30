## MoE Expert Placement Load Balancing（MoE 推理负载均衡 / Expert到GPU分配优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE 推理中的 Load Balancing 是指基于运行时 expert 激活数据，优化 expert 到 GPU 的分配（placement），使得各 GPU 的计算负载尽可能均衡。Huang et al. (NeurIPS 2024) 将其形式化为多路数字划分问题（multi-way number partitioning, NP-hard），约束每 GPU 等量 experts。

训练时的 load balancing loss 鼓励均匀使用 experts，但推理时的 token 分布可能差异显著——训练分布 ≠ 推理分布。严重不均衡导致：(1) 热 expert GPU 过载，OOM 风险；(2) 冷 expert GPU 空闲等待。论文提出两种启发式算法。

从系统架构角度拆解术语：

```
问题形式化:
  P_{mn} ∈ {0,1}: expert m 是否放在 GPU n (m=1..E, n=1..D)
  A_{mb} ∈ [0,1]: expert m 在 batch b 中的 token 比例 (activation data)
  
  min max_{m,b} |Σ_n P_{mn} A_{mb} - 1/D|
  s.t. Σ_m P_{mn} = E/D （每 GPU 等量 experts）

Algorithm 1: Greedy Balancing (for LM, MT-Encoder, 独立激活)
  1. 排序 experts 按 Ã_m = mean_b A_{mb} 降序
  2. 循环 experts (高负载优先):
       分配 expert m 到当前负载最小的 GPU
       GPU 满 (E/D experts) 后排除
  3. 输出 P_{mn}

Algorithm 2: Anti-Correlation Balancing (for MT-Decoder, 相关激活)
  // MT-Decoder 某些 experts 趋于同时激活 (correlated)
  // 仅看 Ã 会低估负载（相关 experts 可能同时 spike）
  1. 计算 Pearson 相关系数 S_{ab} = corr(A_a, A_b), 基于历史数据
  2. 修改 GPU 负载估计:
       load_estimate[n] = Σ_m P_{mn} (Ã_m + 0.5 × Σ_{a in GPU_n} S_{am})
       // 惩罚项: 若与 GPU 上已放置 experts 高度相关，提高估计负载
  3. 贪心分配（同 Algorithm 1，但用修正的 load_estimate）

部署流程:
  [Profiling Phase]
    用少量 batches 运行推理 → 收集 A_{mb} (activation data)
  [Partitioning Phase]  
    运行 Greedy 或 Anti-Correlation → 输出 P (expert→GPU 映射)
  [Inference Phase]
    将 experts 按 P 分配到各 GPU
    可与 Expert Buffering 结合：相关 GPU 负载均衡 → cache 命中率更高
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Python 实现，作为预部署工具运行：(1) profiling 收集 activation matrix A_{mb}；(2) 调用 Greedy/Anti-Correlation 算法生成 placement map；(3) 加载模型时按 placement 分配 experts。与 Expert Buffering 协同使用时，balanced placement 提升 cache 命中率（因为 activation 分布更均匀）。论文实验：Greedy 在 LM 上减少 Max load 从 >0.6 降至 <0.4，吞吐提升至多 1.19×；Anti-Correlation 对 MT-Decoder 相关激活场景降低 Max/Avg-Max load，但吞吐增益 modest（1.02×）因 decoder 本身激活极稀疏。

涉及论文标题：
- Toward Efficient Inference for Mixture of Experts
- Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference
