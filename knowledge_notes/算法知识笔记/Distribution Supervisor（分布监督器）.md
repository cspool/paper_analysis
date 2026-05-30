## Distribution Supervisor（分布监督器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Distribution Supervisor 是 ECRD 框架的第一个核心组件，负责在解码的每一步基于文本证据池对 base LVLM 的 token 分布进行重新评估和混合。其输入为：(a) base 模型当前的 next-token 分布 p_i；(b) 证据池 E_i 中的 N 条文本证据句。输出为协商混合后的分布 p_i^{mix}。核心功能：(1) 对每条证据计算 mean-over-prefix 概率 q_E(w)（替代 VDGD 的 min-over-prefix KL）；(2) 跨多条证据取平均支持度 S_i(w)；(3) 仅对 knee-selected 候选集 C_i 内归一化得到证据诱导分布 r_i(w)；(4) mass-matching：将 r_i 在 C_i 内的总 mass 缩放至与 p_i 匹配；(5) 通过自适应权重 α_i = p_{(1)} 混合 base 和 evidence 分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Distribution Supervisor 的核心计算
Input: base_dist p_i, evidence_pool E_i, candidate_set C_i

# Step 1: 证据评分
for each evidence sentence E_j in E_i:  # E_j length L
    q_Ej(w) = (1/L) * sum_{t=1..L} p_VLM(w | e_{<t})
    # 每条证据对所有候选 token 的 mean-over-prefix 条件概率
S_i(w) = -log( (1/N) * sum_{j} q_Ej(w) )
# 所有证据对 token w 的平均支持度（取负 log 转换为得分）

# Step 2: 证据诱导分布
r_i(w) = softmax_{w in C_i}(-S_i(w))  # 仅候选集内归一化

# Step 3: Mass-matching
mass_p = sum_{w in C_i} p_i(w)
mass_r = sum_{w in C_i} r_i(w)
r_tilde_i(w) = r_i(w) * (mass_p / mass_r)

# Step 4: 自适应混合
alpha_i = max(p_i)  # base 模型 top-1 概率
p_mix_i(w) = alpha_i * p_i(w) + (1-alpha_i) * r_tilde_i(w)  # w in C_i
p_mix_i(w) = alpha_i * p_i(w)                              # w not in C_i
```

关键设计：α_i = p_{(1)} 使 supervisor 在 base 模型自信时（p_{(1)} 大）保持其主导，在 base 模型犹豫时（p_{(1)} 小，即分布平坦、更易产生幻觉）给证据更大权重。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Supervisor 作为纯计算模块，运行在 CPU 上（precomputed log-likelihoods 存储为 FP16），计算复杂度 O(k*|E_i|)，k* 为个位数、|E_i| 增长缓慢，GPU 压力可忽略。Supervisor 是"始终在线"的防御层——即便不触发 visual decider，分布监督器也在每步进行证据约束重加权，提供稳定的幻觉抑制。Ablation 中 supervisor alone（无 visual decider）已在 TreeBench 上带来 +3.7 点提升（37.0%→40.7%）。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
