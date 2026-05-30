## Expert Choice Routing (Expert-Choice MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Choice Routing（专家选择路由）是 Zhou et al. (2022) 提出的一种 MoE 路由策略，与传统的 Token-Choice Routing（每个 token 独立选择 top-k 个专家）相反，该方法让每个**专家**独立选择 top-k 个**token**。具体流程：(1) 计算 token-to-expert affinity matrix S = Softmax(X · W_g) ∈ R^{n×e}（与传统方式相同）；(2) 转置得到 S^T ∈ R^{e×n}，对每个专家（每一行）取 TopK，k = n×c/e（c 为容量系数，e 为专家数），得到 G, I = TopK(S^T, k)；(3) 通过排列矩阵 P = OneHot(I) 将 token 按专家分组：X_in = P · X ∈ R^{e×k×d}；(4) 各专家独立计算 FFN；(5) 反排列回原始顺序：X_out = Σ P[i,j,l] · G[i,j] · X_e[i,j,d]。核心创新：(a) 每个专家恰好处理 k 个 token，负载天然完美均衡，无需 auxiliary load balancing loss；(b) 每个 token 可被 0~e 个专家选中（实际分布：~77% tokens 被 1-2 个专家选中，~23% 被 3-4 个），实现可变计算分配。训练收敛速度比 GShard top-2 gating 快 2× 以上，每步 latency 快 20%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Expert Choice Routing 的完整计算流程：
```
输入: X ∈ R^{n×d}  (n = batch_size × seq_len)
      W_g ∈ R^{d×e}  (gate 参数)
      c (容量系数), e (专家数)
输出: X_out ∈ R^{n×d}

# Step 1: affinity 计算（与 token-choice 相同）
S = Softmax(X @ W_g)  ∈ R^{n×e}

# Step 2: 专家选 token（核心区别——对 S^T 的每行即每个专家取 top-k）
k = n × c / e
G, I = TopK(S^T, k)
# G ∈ R^{e×k}: 门控权重，G[i,j] = S[I[i,j], i]
# I ∈ R^{e×k}: I[i,j] = 第 i 个专家选的第 j 个 token 的全局索引

P = OneHot(I)  ∈ R^{e×k×n}  # 排列矩阵

# Step 3: shuffle — 按专家分组 token
X_in = P @ X  ∈ R^{e×k×d}

# Step 4: 专家 FFN（每个专家批量处理 k 个 token）
for i in 1..e:
    X_e[i] = GeLU(X_in[i] @ W_1[i]) @ W_2[i]^T
# X_e ∈ R^{e×k×d}

# Step 5: unshuffle — 反排列回原始 token 顺序，门控加权
X_out[l, d] = Σ_{i=1..e} Σ_{j=1..k} P[i,j,l] × G[i,j] × X_e[i,j,d]
```
与 Token-Choice 的关键区别：TopK 应用于 S^T 的行（专家维度）而非 S 的行（token 维度），k 由全局容量决定（k = n×c/e），而非每个 token 固定选 k 个专家。可选的约束版本 EC-CAP：通过熵正则化线性规划 + Dykstra 交替投影算法限制每个 token 最多 b 个专家，λ=0.001，max 100 iterations。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 官方实现：Google Research `sparse_mixers/routing.py`（JAX/Flax）包含 `ExpertsChoose` 路由；Flaxformer/T5X 也集成此机制。
- 第三方 PyTorch 实现：`pytorch-mixtures` 提供 `ExpertChoiceRouter`。
- 训练配置：c=2 匹配 GShard top-2 计算量，c=1 匹配 Switch Transformer top-1。无需 load balancing loss。最大模型 8B/64E 使用 512 TPU V4 chips，GSPMD 2D sharding。
- 局限：不直接适用于 auto-regressive 生成（需要 future tokens）；小 batch inference 时需改用 global top-k + cap 策略。

涉及论文标题：
- Mixture-of-Experts with Expert Choice Routing
