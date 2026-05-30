## AdaAlloc (Adaptive Allocation / 自适应参数分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AdaAlloc 是 QWHA 提出的量化感知适配器参数选择策略，由两个层级组成：(1) 通道级自适应预算分配（Channel-wise Budget Allocation）：根据各输出通道的量化误差大小按比例分配可训练参数数量； (2) 通道内幅值选择（Intra-channel Magnitude-based Selection）：在每个通道内，选择 WHT 变换域中系数幅值最大的位置。核心公式为 p_i = floor(p × ||(ΔW_Q X)_{i,:}||_F^t / Σ_j ||(ΔW_Q X)_{j,:}||_F^t)，其中 t 为温度超参数（默认 t=1）。余数分配给当前分配最少的通道，保证所有通道 ≥ 2 个参数以维持 F 的 full-rank 性质。AdaAlloc 是首次同时兼顾"full-rank 表示（fine-tuning 能力）"和"低初始化误差（量化误差补偿）"的参数选择策略：纯幅值选择导致 F 低秩（参数过度集中在少量异常值通道），随机选择虽保持高秩但初始化误差大。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AdaAlloc 完整算法流程：

```
Input:  ΔW_Q ∈ R^{d_out × d_in}  # 权重量化误差
        X ∈ R^{d_in × (b·s)}     # 校准集激活
        p                          # 总参数 budget
        t = 1.0                    # 温度参数

# Step 1: 计算各通道的输出误差
for i in 0..d_out-1:
    error[i] = ||(ΔW_Q @ X)[i, :]||_F^t  # 第 i 个输出通道的误差 t 次幂

# Step 2: 按比例分配预算
total_error = sum(error)
p_i = floor(p × error[i] / total_error)  for each i

# Step 3: 余数分配（保证每通道 ≥2, sum(p_i) = p）
remainder = p - sum(p_i)
sort channels by p_i ascending
distribute remainder to channels with smallest p_i (+1 each)
ensure all p_i >= 2  # 满足 full-rank 条件

# Step 4: 通道内幅值选择
B = H^{-1} @ R   # 预计算投影基
for i in 0..d_out-1:
    v = (ΔW_Q)[i, :] @ R
    dense_sol = v @ B^{-1} = (ΔW_Q @ H)[i, :]  # 稠密 WHT 系数
    E_i = TopK_Index(|dense_sol|, p_i)   # 选幅值最大的 p_i 个位置
    B' = B[E_i, :]                         # 选中位置的基向量
    c_i = v @ B'^T @ inv(B' @ B'^T)       # Refinement: 最小二乘精化

Output: E (参数位置), c (参数值)

# AdaAlloc vs 其他策略的对比:
# Magnitude-based: p_i = |ΔW_Q @ H 的前 p 个最大系数| → low-rank F
# Random (LoCA): E 随机初始化 → 高秩但高初始化误差
# SSH: 50% 幅值 + 50% 随机 → 中间方案
# AdaAlloc: 通道级分配 + 通道内幅值 → 唯一同时 high-rank + low-init-error
```

Layer output error 对比 (Table 2): AdaAlloc avg 3.86 vs Magnitude 3.82 vs SSH 4.57 vs Random 5.96, None (no init) 7.21。AdaAlloc 与 Magnitude 误差接近但 rank 远高于 Magnitude（Figure 4：AdaAlloc rank≈r_max，Magnitude rank≈0）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AdaAlloc 的温度参数 t 控制分配锐度：t < 0.5 时接近均匀分配（无法集中参数到高误差通道），t > 1.5 时过度集中（忽略低误差通道中的重要系数）。论文推荐范围 t ∈ [0.5, 1.0]，默认 t=1 实验表现最优（GSM8k: t=1 得 41.47% vs t=0.25 得 40.11% vs t=2.0 得 40.04%）。AdaAlloc 保证 full-rank 的理论依据：满足 Coja-Oghlan et al. (2020) 的稀疏随机矩阵满秩条件（每行每列 ≥2 非零元）。在 P(r≥4) 的参数 budget 下，即使是输出维度最大的线性层也能保证每行 ≥2 参数。AdaAlloc 与 Refinement 步骤配合使用效果最佳：仅 AdaAlloc 不做 Refinement 时 avg error=7.06，加上 Refinement 后降至 3.86（Table 7）。

涉及论文标题：
- QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning
