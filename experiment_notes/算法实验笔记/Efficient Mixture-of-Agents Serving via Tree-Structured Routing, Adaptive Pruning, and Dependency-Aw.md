## Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **Faster-MoA** 的两个算法级创新：(1) **层次化树状 Agent 拓扑**：将 all-to-all 全连接 MoA 替换为三层树结构（9-3-1），每层 agent 被分组为 clusters，下一层 agent 仅连接其对应 cluster 的前驱 agent，形成局部信息聚合→全局聚合的层级结构。(2) **语义引导的运行时动态 Early-Exit**：在每层中，通过 FrobCosSim（Frobenius Cosine Similarity）+ 置信度几何平均计算早退概率 Q，在小 agent 输出足够高质量时以概率 Q 提前终止大 agent 运行。核心计算：先用 Qwen3-Embedding-4B 将各 agent 输出文本编码为 last-layer hidden states T_i ∈ R^{n×h}，计算 feature-wise correlation matrix U = T_i^T × T_i ∈ R^{h×h}，在两矩阵之间计算 FrobCosSim；再结合 token-level log-probability 的几何平均置信度 C_ℓ，计算合成质量分数 Q = √(C̄ · B)，B 为校准后的相似度。
  实验比较：(a) 模型激活分布（4B/8B/32B 各被调用的比例），更难任务（IFBench）更大模型被更多调用; (b) EE 开销（~5% 额外延迟换来 10-50% E2E 减少）; (c) Tree-only vs Tree+Incremental Prefill vs Fully-integrated Faster-MoA vs All-to-all Baseline 的 E2E 延迟和准确率; (d) 准确率在五个 benchmark 上对比（GSM8K/MATH-500/AIME2025/MMLU-ProX-Lite/IFBench）。

- 硬件平台是什么，配置是什么。
  6× NVIDIA H200 GPU（单台 H200 HGX Server），每个模型用两张 GPU（1 PE + 1 DE），总计三组模型（Qwen3-VL-4B-Instruct、Qwen3-VL-8B-Instruct、Qwen3-VL-32B-Instruct + 额外 Qwen3-Embedding-4B 用于动态 EE）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen3-VL-4B-Instruct、Qwen3-VL-8B-Instruct、Qwen3-VL-32B-Instruct（来自 Qwen model family），外加 Qwen3-Embedding-4B 做 embedding 用于动态 EE routing。三个模型共享相同 tokenizer 避免异构 tokenizer 编排问题。采样参数按 model card 推荐设置。
  数据集/benc"hmarks：五个——GSM8K（小学数学推理）、MATH-500（中等数学）、AIME2025（竞赛数学）、MMLU-ProX-Lite（STEM 综合科学）、IFBench（指令遵循测试）。覆盖从易到难的数学推理和通用科学 QA 任务。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文代码未公开（Georgia Tech + Peking University + Samsung，提交 DAC 2026）。以下基于论文 (Sec. 4.1-4.2) 给出算法 pipeline 伪代码：

  **=== 层次化树状拓扑 (Tree Topology) ===**
  ```
  输入: user query
  输出: final answer from root agent

  Layer 1 (9 leaf agents, 3 clusters):
    Cluster 1: agents {a_{1,1}, a_{1,2}, a_{1,3}} (各自用 4B/8B/32B Qwen3-VL)
    Cluster 2: agents {a_{1,4}, a_{1,5}, a_{1,6}}
    Cluster 3: agents {a_{1,7}, a_{1,8}, a_{1,9}}
    每个 cluster 独立并行执行

  Layer 2 (3 aggregation agents):
    a_{2,1} ← 仅依赖 Cluster 1 的输出 (C(a_{2,1}) = Cluster 1)
    a_{2,2} ← 仅依赖 Cluster 2 的出 (C(a_{2,2}) = Cluster 2)
    a_{2,3} ← 仅依赖 Cluster 3 的输出 (C(a_{2,3}) = Cluster 3)
    只需自己的 local precursors 完成即可启动，无需等待其他 cluster

  Layer 3 (1 root aggregator agent):
    a_{3,1} ← 依赖所有 Layer 2 agents 输出 → 生成最终答案

  Latency 优势:
    T_ℓ^{tree} ≈ max_{a_{ℓ,j}} max_{c∈C(a_{ℓ,j})} t_c
    vs T_ℓ^{all} = max_i t_{ℓ,i} (all-to-all)
    每个 successor 仅等待其连接的 precursors，无关子树可并发
  ```

  **=== 动态 Early-Exit (Algorithm 1) ===**
  参数：偏好相似度阈值 τ = 0.7（经验最优）
  ```
  输入: 当前层 ℓ 的已完成 LLM 输出 {O_1,...,O_ℓ} 和 token-level log-prob {log p_ℓ^i}_{i=1..n_a}
  输出: Early-exit probability Q

  1. 置信度计算:
     for i = 1..n_a:  // n_a = 当前已完成 LLM 数
       C_ℓ ← exp( (1/n_a) * Σ_{i=1}^{n_a} log p_ℓ^i )  // 几何平均置信度
     C̄ ← √( (1/ℓ) * Σ_{i=1}^{ℓ} C_i^2 )  // RMS 历史置信度

  2. 语义相似度计算:
     for i = 1..ℓ:
       T_i ← Embed(O_i) via Qwen3-Embedding-4B  // [n_i × h] last-layer hidden states
       T_ℓ ← Embed(O_ℓ)
       U ← T_i^T × T_i  ∈ R^{h×h}  // feature-wise correlation matrix
       V ← T_ℓ^T × T_ℓ  ∈ R^{h×h}
       Sim[i,ℓ] ← FrobCosSim(U, V)
         = trace(Corr(U)^T · Corr(V)) / (||Corr(U)||_F · ||Corr(V)||_F)
       Sim[ℓ,i] ← Sim[i,ℓ]

  3. 置信度加权相似度:
     W ← Σ_{i=1}^{ℓ} Σ_{j=1}^{i} C_i · C_j
     P ← (1/W) * Σ_{i=1}^{ℓ} Σ_{j=1}^{i} C_i · C_j · Sim[i,j] ∈ [0,1]

  4. 校准 (防止过度一致):
     B ← 1 - |P - τ| / τ  ∈ [0,1]  // τ=0.7 经验最优

  5. 合成质量分数:
     Q ← √(C̄ · B)^(1/τ)

  6. 以概率 Q 执行早退:
     终止当前层剩余未完成的 LLM
  ```

  关键数学公式：
  - Frobenius Cosine Similarity: FrobCosSim(U,V) = ⟨Corr(U), Corr(V)⟩_F / (||Corr(U)||_F · ||Corr(V)||_F)
  - 其中 Corr(U)_ij = U_ij / √(U_ii · U_jj)（将矩阵转为 correlation matrix 消除尺度/单位影响）
  - ⟨U,V⟩_F = trace(U^T V), ||U||_F = √⟨U,U⟩_F
  - 概率 Q 高 → 当前已完成 agent 输出置信度高且语义适度一致 → 可以早退
  - 概率 Q 低 → 需要等待更多（尤其是更大的）agent 完成
