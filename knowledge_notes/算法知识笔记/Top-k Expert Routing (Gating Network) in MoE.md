## Top-k Expert Routing (Gating Network) in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-k Expert Routing（也称 Gating Network 或 Router）是 MoE 架构中决定每个 token 由哪些专家处理的机制。Router 是一个小型线性层 W_r ∈ R^{d×n}，输入 hidden state x，输出 n 维路由 logits l = W_r^T x，通过 Softmax 转换为路由权重 w = Softmax(l)。然后取 w 中最大的 k 个值对应的专家索引作为该 token 的激活专家。Router 的参数占总参数比例极小（Mixtral 8x7B 中 Router 参数约 d×n ≈ 4096×8 = 32K per layer × 32 层 ≈ 1M，vs 总参数 47B）。路由权重的两个用途：(1) 选择 top-k 专家——决定哪些专家执行计算；(2) 加权聚合——归一化后的 w̃_{e_j} 作为各专家输出的混合权重。路由机制在训练和推理中均保持稀疏激活（仅 k 个专家计算），这是 MoE 高效性的核心——参数量大但计算量小。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Router 计算流程（per token）：
```
输入: x ∈ R^d, Router权重 W_r ∈ R^{d×n}, 专家数 n, top-k

l = W_r^T x                          # (n,) routing logits
w = Softmax(l)                       # (n,) routing weights, Σw_i = 1
indices = TopKIndices(w, k)          # 取 top-k 专家索引
weights = w[indices]                 # 对应路由权重
w̃ = weights / sum(weights)           # 归一化: Σw̃ = 1

# 门控输出: (indices, w̃) → 传给 Expert FFNs
```
在 Mixtral 8x7B 中 n=8, k=2。路由分布分析（论文 Fig.5）：不同数据集（C4 vs MATH）下路由偏好差异显著，同一数据集内也存在层间差异。路由坍塌是主要风险——部分专家路由权重始终为 0。缓解方法：负载均衡损失（辅助损失）、expert capacity 限制、noisy top-k gating（加噪声后取 top-k）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HuggingFace Transformers 中实现为 `MixtralSparseMoeBlock` 内的 `nn.Linear(hidden_dim, num_experts)`。Router 在 calibration 阶段用于生成校准数据分布分析路由偏好（如专家激活频率统计）。Router 权重在 expert pruning 中也被丢弃：剪枝后仅保留选中专家的对应路由权重通道（而非常见做法——剪枝后重新归一化路由）。Dynamic Skipping 利用路由权重比 w_{e1}/w_{e0} 做在线跳过决策，不需要修改 Router 本身。

涉及论文标题：
- MoEQuant Enhancing Quantization for Mixture-of-Experts Large Language Models
