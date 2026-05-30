## Distribution Shift-based Expert Layer Selection / Extender (基于分布偏移的专家层选择 / 扩展器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Extender 是 MoExtend 中用于自适应决定"在哪些 MoE 层添加新专家"的机制。直接在所有 MoE 层添加新专家会增加参数和过拟合风险，但手动设计插入策略（如前半层、间隔层等）需要大量实验调优。Extender 通过度量各层在新模态数据下 expert 选择分布的偏移程度，自动选出最需要扩展的层。

从算法pipeline角度拆解术语：

**Extender 的完整流程：**
```
# 输入：
#   κ: Alignment stage 后得到的模型（router 未微调）
#   S_t: 子训练集（LLaVA 1.5-mix-665k，除去验证集后的数据）
#   S_e: 验证集（10,000 条随机抽样）

# Step 1: 微调 router（仅使 router 可训练，其余冻结）
κ' = copy(κ)
for step in range(1000):
    κ' = train_step(κ', S_t, trainable={all_routers})

# Step 2: 统计 expert 被选次数
R_κ  = count_expert_selections(κ, S_e)   # [m, L] 矩阵
R_κ' = count_expert_selections(κ', S_e)  # [m, L] 矩阵

# Step 3: 归一化为概率分布
R̄_κ  = normalize_by_column(R_κ)   # 每列（每层）归一化
R̄_κ' = normalize_by_column(R_κ')

# Step 4: 逐层计算分布差异
for j in range(L):
    diffs = [R̄_κ'[i,j] - R̄_κ[i,j] for i in range(m)]
    d_j = std(diffs)  # 标准偏差度量分布偏移程度

# Step 5: 选 top-⌊pL⌋ 层添加 expert（p=0.5）
# 对 Mixtral 8x7B：L=32, ⌊0.5×32⌋=16 层
selected_layers = top_k_by_d(layers, k=16)
```

注释：
- d_j 小 → MoE 层 j 对新模态数据响应变化小 → 无需新 expert
- d_j 大 → MoE 层 j 在新模态下路由分布发生显著变化 → 该层需新 expert 专门处理新模态
- p=0.5 是基于消融实验的默认值（16 层与手动最佳策略 First-half/Interval 性能相当，但训练时收敛更快）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **与手动搜索的对比**：消融实验显示，全层加（All layer, 32 layers）与最优手动策略（First-half/Interval, 16 layers）性能几乎相同（POPE 84.0 vs 84.5 vs 83.5），但 Extender 自动选出的 16 层在训练时收敛更快
- **选出的层分布**（Mixtral 8x7B）：集中在模型中部（layer 3-28），深层和极浅层变动小。具体为层 3,4,6,7,9,10,11,13,14,15,17,18,20,21,26,28
- **计算开销**：Extender 仅需 1,000 步 router 微调 + 一次验证集前向统计，相对于后续 Fine-tuning Stage (30h) 可忽略
- **推广性**：Extender 的设计理念（通过分布偏移度量"哪个组件对新数据最敏感"）可推广到任何"在预训练模型中选择性添加新组件"的场景，如增量学习、持续学习的模块扩展

涉及论文标题：
- MoExtend: Tuning New Experts for Modality and Task Extension
