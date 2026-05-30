## Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：OEA（Opportunistic Expert Activation）是一种无需重新训练的 batch-aware MoE 路由算法，用于降低 decode 阶段的延迟。核心是一个两阶段路由策略：
    - **Phase 1（Baseline Expert Selection）**：对 batch 中每个 token，激活其 top-k0 个最优先的专家，保证每个 token 的独立质量基线。
    - **Phase 2（Opportunistic Piggybacking）**：对每个 token，遍历其 top-k0 之后的低优先级专家，若该专家已在 Phase 1 中被其他 token 选入 S_base（即专家权重已被加载到 SRAM），则免费将该专家分配给当前 token（但不超过 k_max 上界），保持激活专家总数 T = |S_base| 不变，从而在不增加延迟的前提下恢复模型性能。
    - 最终路由权重按式 (1) 重归一化：moe(x) = sum_{i in S} (R(x)_i / sum_{j in S} R(x)_j) * E_i(x)。
    - 简化版本：消融实验表明 k_max=k=8、maxP=128（不限制）、p=1.0（固定 top-k0）效果最优，最终简化为 Algorithm 1 —— 仅在 Phase 1 用固定 k0 且 Phase 2 可填入 top-k 中的 S_base 专家。
  - 实验比较：(1) Cross-entropy vs. 平均激活专家数：FineWeb-Edu 子集上扫描 k0、k_max、p、maxP，batch size B ∈ {8,16,32,64}，对比 vanilla top-8 routing、Phase-1-only（pruned）和 OEA 的 Pareto 前沿；(2) 下游 Benchmark：AIME24、MATH500、GPQA、LiveCodeBench，对比 vanilla、pruned（top-k0）和 OEA 的准确率；(3) MoE 层延迟 vs. 激活专家数：测量所有 decode step 和所有 layer 的 (T, latency) 对，验证线性关系；(4) Qwen3-30B-A3B 和 Qwen3-235B-A22B 两个模型规模。

- 硬件平台是什么，配置是什么。
  - Qwen3-30B-A3B：单卡 NVIDIA H100 80GB，bfloat16 精度。
  - Qwen3-235B-A22B：8 张 H100 80GB，单节点 HGX H100，NVSwitch 互联（每 GPU pair 18 条 NVLink），tensor parallelism degree=8。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Qwen3-30B-A3B（48 layers, N=128 experts, k=8 activated per token, embedding dim=2048, expert hidden dim=768, SwiGLU FFN）；Qwen3-235B-A22B（96 layers, embedding dim=4096, expert hidden dim=1536, 同样 top-8/128 routing）。
  - 数据集：FineWeb-Edu 子集（2048 条 sequence，每条 ≥8192 token，用于 cross-entropy 评估）。
  - Benchmark：AIME24（数学竞赛）、MATH500、GPQA（研究生级问答）、LiveCodeBench（代码生成）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未提供独立开源仓库链接。路由算法集成进 SGLang 框架。
  - 伪代码（简化版 Algorithm 1）：
    ```
    输入: token embeddings x_{1..B}, 每个token的top-k0基线专家数k0,
          排序专家索引 e_{i,j}（按router score降序）
    
    Phase 1: 为每个token i, S_i_base = {e_{i,1}, ..., e_{i,k0}}
    Phase 2: S_base = union_i S_i_base  // 所有必需专家的并集
             对每个token i:
               S_i = S_i_base
               对 j = k0+1 到 N:
                 若 |S_i| >= k: break
                 若 e_{i,j} in S_base:
                   S_i = S_i ∪ {e_{i,j}}
    输出: 最终专家集合 S_1, ..., S_B
    ```
  - 张量计算示例：对 batch 中 B=16 个 token，k0=5，k=8，N=128。
    Phase 1 每 token 选 top-5，S_base 约含 30-40 个不同专家（远小于 128）。
    Phase 2 对每个 token 检查 S_base 中是否有其 6-8 位排名的专家，若有则免费附加。
    最终每 token 仍激活约 8 个专家，但 T ≈ |S_base| ≈ 30-40（而非 vanilla 的 ~48-82）。
    MoE 层输出 = sum_{i in S} softmax(R(x)_S) * E_i(x)，延迟从 b*43 + a*16*8 降至 b*30 + a*16*8（b >> a 时约降 30-50%）。
