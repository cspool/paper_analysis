## BrownoutMoE Architecture (BrownoutMoE 架构)

术语解释
BrownoutMoE 是 BrownoutServe 提出的 MoE 模块变体，在标准 MoE（含 shared experts 和 routed experts）基础上引入 united experts 和 brownout token routing，通过动态选择部分 token 走原始 experts、部分 token 走 united experts 来平衡精度和延迟。

术语是什么？
BrownoutMoE 的输出公式为（Eq. 5）：

$$\mathbf{h}_{t} = \mathbf{x}_{t} + \sum_{i=1}^{N_{s}} \text{FFN}_{i}^{(s)}(\mathbf{x}_{t}) + \sum_{i=1}^{N_{r}} p_{i,t} \text{FFN}_{i}^{(r)}(\mathbf{x}_{t}) + \sum_{i=1}^{N_{u}} q_{i,t} \text{FFN}_{f(i)}^{(u)}(\mathbf{x}_{t})$$

其中 N_s、N_r、N_u 分别为 shared experts、routed experts（原始）、united experts 的数量。p_{i,t} 为 token t 属于 S1（原 experts 处理）时对 expert i 的 routing weight，q_{i,t} 为 token t 属于 S2（united experts 处理）时的 routing weight。

与标准 MoE 的区别：第三项 Σq_{i,t}·FFN_{f(i)}^{(u)}(x_t) 是新增的 united expert 路径，f(i) 将原始 expert index 映射到其所在的 united expert group。这种设计使得即使是 cold expert 的 token 也能得到近似处理而非被忽略。

从算法pipeline角度拆解术语：
BrownoutMoE 的完整 forward 路径：
```
输入: x_t (token hidden state)

# 1. Gate 计算（与标准 MoE 相同）
for each token t, expert i:
    s_{i,t} = x_t^T @ e_i               # affinity score
g_{i,t} = softmax(TopK({s_{j,t}}, K))  # routing weight

# 2. Brownout 划分（新增）
expert_token_counts = count tokens routed to each expert
A = sort experts by token_count descending
T = total_tokens * threshold
S1 = top experts whose cumulative token count ≥ T
S2 = remaining experts

# 3. Shared expert（同标准 MoE）
h_shared = Σ FFN_i^{(s)}(x_t)

# 4. Routed experts - 分两路（Brownout 关键创新）
# Path A: S1 tokens → 原始 experts
h_original = Σ p_{i,t} * FFN_i^{(r)}(x_t)   # p_{i,t} = g_{i,t} if i∈S1 else 0

# Path B: S2 tokens → united experts
h_united = Σ q_{i,t} * FFN_{f(i)}^{(u)}(x_t) # q_{i,t} = g_{i,t} if i∈S2 else 0

# 5. 输出
h_t = x_t + h_shared + h_original + h_united
```

术语一般如何实现？如何使用？
- 实现约 5.5k 行 Python，基于 PyTorch。MoE 相关算子使用 Triton 重写。
- 与 DeepSeekMoE 等架构的区别：DeepSeekMoE 使用 shared experts + fine-grained routed experts，BrownoutMoE 在此基础上增加 united experts 作为第三类 expert 组件。
- 适用场景：可集成到任何 MoE transformer 模型中。

涉及论文标题：
- BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs
