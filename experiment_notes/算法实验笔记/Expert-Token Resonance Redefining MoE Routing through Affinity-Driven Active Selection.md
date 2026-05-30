## Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是一种双向 MoE 路由算法 ETR（Expert-Token Resonance），包含三项算法创新：(1) **GrAP（Grouped Average Pooling）路由层**：将传统 MLP Router 的稠密权重矩阵替换为对角稀疏亲和力矩阵 W_aff，参数降为原来的 1/D（D 为分组因子，按 expert 数 n 分组），计算复杂度从 O(d²) 降至 O(d²/D)，且正交性天然防止 expert 同质化；(2) **TCR+ECR 双向选择机制**：token 先通过余弦相似度亲和力分数选择 top-ℓ experts（TCR），然后每个 expert 再按其亲和力分数从已分配的 token 中选择 top-C tokens（ECR），实现"共振效应"；(3) **自适应容量策略**：基于训练进度动态调整 expert capacity 下界，理论证明可将容量下界降低最多 40%，消除 All-to-All 通信气泡。

  实验比较：(a) ETR（LocMoE+）vs Baseline（标准 Top-1 MoE routing + capacity factor 1.1）vs LocMoE（TCR only）vs LocMoE（ECR only）；(b) 训练效率对比：32N/64N/256N Ascend NPU 集群下的每步耗时、各计算阶段（computation/communication/overlap/idle）分布、operator 级别耗时（FFN MatMul、TopK、IndexPutV2）、显存占用；(c) 路由质量对比：Calinski-Harabasz (CH) Index 衡量 token 聚类质量、不同 loss 函数下的 token 分配分布（CDF/ECDF）；(d) 下游任务对比：GDAD（含 GDAD-1/2/3 三个子任务及 16+13+18 项子能力）、GPQA、HumanEval、MMLU、TeleQnA；(e) SFT 后 GDAD 16 项子能力对比。

- 硬件平台是什么，配置是什么。
  Huawei Ascend 910B3 NPU 集群，三组规模：(1) 32N：TP=4, PP=4, DP=2, EP=2；(2) 64N：TP=8, PP=4, DP=2, EP=2；(3) 256N：TP=8, PP=8, DP=4, EP=2；全局 batch size=128。单颗 910B3 NPU：20 AI Cores @ 1.8GHz，fp16 理论算力 313T，HBM 64GB @ 1.6GHz，带宽 1.6T。每 8 颗 NPU 安装在同一 Atlas 800T A2 服务器内，全 mesh 互联。

- 模型是什么。数据集和bench分别是什么。
  模型：Mixtral 8×7B（46.7B 参数），GQA 注意力，32 层 sparse MoE block，每层 8 experts，top-2 选择（实验中统一用 top-1），序列长度扩展至 32768 tokens。
  预训练数据集：自建 300B tokens（150B ICT 领域 + 150B 通用数据），含中英双语，领域数据来自 iCase、blogs、Wiki、feature documents 等华为内部技术文档。通用数据含网页、书籍、代码、问答等 (详见 Table 3)。
  SFT 数据集：762,321 通用 QA + 11,048 领域 QA（比例 68:1），两阶段训练（Stage1 ~2M 样本增强逻辑推理，Stage2 ~3M 样本增强指令遵循）。
  Benchmark：GDAD（自建，含 16 类领域任务能力 2657 题 + 13 类领域认证考试 13968 题 + 18 类通用能力 1435 题）、GPQA、HumanEval、MMLU、TeleQnA。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文来自华为，未提供公开 GitHub 仓库。代码基于华为内部 MindSpeed-LLM / MindSpeed / AscendSpeed 生态，运行于 Ascend NPU。以下给出 ETR 算法 pipeline 的伪代码：

```
# === ETR 路由算法 (每层 MoE block) ===
# 输入: x ∈ R^{s×d}  (s 个 token，每个 d 维)
# 超参: n (expert 数), D=n (分组因子), ℓ (top-k), C (自适应容量)

# Step 1: GrAP 构建亲和力矩阵 W_aff
# W_aff ∈ R^{d×n} 为对角稀疏矩阵
for i in range(n):
    start = i * d // n
    end = (i+1) * d // n
    w_i[j] = n/d  for j in [start, end)   # 公式(3)
    # W_aff 仅对角线有值，参数量 = d, 而传统 MLP Router 为 d×n
    # 计算复杂度 O(d²/n) vs 传统 O(d²)

# Step 2: 计算 token-expert 亲和力分数
# δ_{t,i} = cos(x_t, w_i) = x_t^T w_i / (||x_t|| * ||w_i||)  # 公式(4)
for t in range(s):
    for i in range(n):
        delta[t][i] = cosine_similarity(x[t], W_aff[:, i])

# Step 3: TCR — Token 选择 Top-ℓ experts
for t in range(s):
    top_experts[t] = TopK(delta[t, :], k=ℓ)  # 公式(5)
    # 每个 token 分配至其亲和力分数最高的 ℓ 个 expert

# Step 4: ECR — Expert 选择 Top-C tokens（双向选择）
# 动态计算容量 C = max(C_min, adaptive_capacity(delta, training_progress))
# C_min = (1/n) * exp(d * δ_max² / (2 - δ_max²))  # Remark 7

for i in range(n):
    # 获取第一步中被分配至 expert i 的所有 token
    assigned = [t for t in range(s) if i in top_experts[t]]
    # Expert i 按亲和力分数从 assigned 中选择 Bottom-C（最低分数）
    # 即保留最高亲和力的 C 个 token
    selected[i] = BottomC(delta[t, i] for t in assigned, c=C)  # 公式(6)

# Step 5: MoE 计算（仅对选中的 token-expert 对）
output = zeros(s, d)
for i in range(n):
    for t in selected[i]:
        output[t] += G_i(x[t]) * E_i(x[t])  # gating weight * expert FFN

# === Locality Loss（负载均衡）===
# L_loc = μ * KL(D_c || D_l)
# D_c: 当前 token 分布, D_l: 完全本地化分布
# 鼓励 token 发送至同节点 expert，减少跨节点通信
total_loss = task_loss + alpha * L_aux + beta * L_loc

# === 训练阶段自适应 ===
# Phase 1 (早期，q_i ≈ Θ(1)): 偏向 TCR, C = Θ(s)
# Phase 2 (后期，s·q_i ≤ C*): 偏向 ECR, C = Θ(1)（容量降低~40%）
```

关键理论依据：Theorem 5 证明早期训练中 TCR 成功率为 Θ(C·∑p_i/s)，ECR 呈指数衰减 e^{-s}；后期当 expert 获得判别能力后 (q_i << 1)，ECR 接近 100% 成功率，TCR 仍受限于 C/s。因此动态从 TCR 过渡到 ECR 能最大化全程训练成功率。
