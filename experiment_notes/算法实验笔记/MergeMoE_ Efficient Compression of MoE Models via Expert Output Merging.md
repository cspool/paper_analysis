## MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 MergeMoE，一种基于输出合并观点的 MoE 模型压缩算法。核心将 expert merging 重新解释为在 forward computation 中插入额外矩阵（A、B、T1、T2、T3）的优化问题：
    - 聚类阶段：基于 expert 权重矩阵 W_U 和 W_G 拼接结果的相似度进行聚类（选取 top-M 使用频率的 experts 作为聚类中心，其余按距离归类）。
    - 合并阶段：簇内使用相对使用频率作为加权权重（经理论证明最优），T2/T3 按 M-SMoE 方式设为加权平均矩阵（式4），T1 通过对采样输入做最小二乘法求解（式6：T1 = QP†）。
    - 最终输出 W'_D T1, T2 W'_G, T3 W'_U 作为合并 expert 的权重矩阵。
  - 实验比较 MergeMoE vs M-SMoE（主 baseline）、Average（简单平均）、ZipIt 三种合并方案；额外与同激活参数规模的 dense 模型（Qwen3-4B, Qwen1.5-1.8B/4B）对比。
  - 消融实验：(1) 不同压缩比例的影响（减少 experts 数量 vs 增加压缩层数）；(2) 输入样本数量对最小二乘法质量的影响（临界阈值 ~32 samples）；(3) 跨数据集泛化能力（单数据集采样 → 全 benchmark 评估）；(4) 压缩误差消融（w/o merging errors vs w/ merging errors）；(5) 合并时间开销对比（MergeMoE vs M-SMoE）；(6) IFEval 指令遵循 benchmark + knowledge distillation 验证。
  - 所有比较实验固定相同压缩层和压缩比确保公平，所有合并算法使用相同数量输入样本。

- 硬件平台是什么，配置是什么。
  - 合并执行：单张 NVIDIA H20 96GB 显存。
  - 评估执行：两张 NVIDIA H20 96GB。
  - 精度：BFloat16（合并阶段的压缩矩阵计算在 GPU 内存中进行）。
  - 合并算法在单 GPU 上运行，每层处理时间 <1 分钟。

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - **Qwen3-30B-A3B**：14B 参数，48 layers，128 routed experts，每 token 激活 8 experts，无共享 experts。压缩：layers 28-47, experts 128→64, 总参数 30B→25B，激活参数 ~3B。
    - **Qwen1.5-MoE-A2.7B**：14B 参数，24 layers，60 routed experts，每 token 激活 4 experts，有共享 experts。压缩：layers 10-23, experts 60→30, 总参数 14B→10B，激活参数 ~2.7B。
    - **DeepSeekMoE**：16B 参数，28 layers，64 routed experts，每 token 激活 6 experts，有共享 experts。压缩：layers 16-27, experts 64→28, 总参数 16B→12B。
  - 数据集/Benchmark（7 个 NLP 任务）：
    - MRPC（语义等价判断）
    - WinoGrande（指代消解）
    - SQuAD（抽取式问答）
    - Hellaswag（常识推理）
    - PIQA（物理交互推理）
    - ARC easy / ARC challenge（科学推理）
    - 额外：IFEval（指令遵循 benchmark）+ ShareGPT（知识蒸馏数据）
  - 评估框架：DCLM（DataComp-LM）执行下游任务评估。
  - 输入采样数据来源：各 benchmark 数据本身（self-sourced），或单一数据集跨任务评估。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未提供开源链接。经搜索未发现公开代码仓库。论文作者来自 Peking University 和 ByteDance。
  - 算法 pipeline 伪代码（单 MoE layer，将 N 个 experts 压缩为 M 个）：

```
输入: MoE layer 含 N 个 routed experts {E_1..E_N}, target M 个 experts
      calibration 输入样本 X̂ (来自 target task 数据)
输出: M 个合并后的 experts {E'_1..E'_M}

// ===== 步骤 1: 聚类 =====
// 计算每个 expert 的使用频率 f_i = 该 expert 被 top-K 选中的次数
// 选取 top-M 使用频率的 experts 作为聚类中心
for each non-center expert j:
    // 距离度量: 拼接矩阵 [W_U || W_G] 的相似度
    dist(j, center_k) = ||[W_Uj||W_Gj] - [W_Uk||W_Gk]||
    将 expert j 分配给距离最近的聚类中心

// 聚类结果 → 确定矩阵 A (式 2):
// A ∈ R^{M×N}, A_{ij}=1 iff 第 j 个 expert 归入第 i 个 cluster

// ===== 步骤 2: 确定合并权重 (Theorem 1, 使用频率最优) =====
for each cluster C_i:
    for each expert j in C_i:
        B_{ji} = f_j / Σ_{k∈C_i} f_k   // 簇内相对使用频率作为权重
// B ∈ R^{N×M} 的列 v_i 仅在 C_i 的索引位置非零

// ===== 步骤 3: 构造扩展参数的合并 expert =====
for each cluster i:
    // 构造中间扩展矩阵 (无维度缩减):
    W'_{Di} = [B_{1i}W_{D1}, B_{2i}W_{D2}, ..., B_{Ni}W_{DN}]  // 水平拼接
    W'_{Gi} = [W_{G1}; W_{G2}; ...; W_{GN}]                     // 垂直拼接
    W'_{Ui} = [W_{U1}; W_{U2}; ...; W_{UN}]                     // 垂直拼接

// ===== 步骤 4: 设置 T2, T3 (式 4, 加权平均) =====
// T2, T3 ∈ R^{E·N × E}  (E = 单个 expert 的 intermediate dim)
T2 = [B_{1i}I, B_{2i}I, ..., B_{Ni}I]  // block diagonal with weights
T3 = [B_{1i}I, B_{2i}I, ..., B_{Ni}I]  // 同上

// ===== 步骤 5: 最小二乘法计算 T1 (式 5-6) =====
// 在前向过程中用 torch hooks 获取中间激活
// 对采样输入 X̂ 做一次前向:
P = σ(T2 · W'_{Gi} · X̂) ⊙ (T3 · W'_{Ui} · X̂)   // 压缩路径的中间激活
Q = σ(W'_{Gi} · X̂) ⊙ (W'_{Ui} · X̂)             // 原始扩展路径的中间激活
// 最小二乘闭式解:
T1 = Q · P^†     // P^† 为 Moore-Penrose 伪逆
// T1 ∈ R^{E×E}, 将扩展维度压缩回单个 expert 维度

// ===== 步骤 6: 构造最终压缩 expert 权重 =====
W^final_Di = W'_{Di} · T1   // shape: (out_dim, E)
W^final_Gi = T2 · W'_{Gi}   // shape: (E, in_dim)
W^final_Ui = T3 · W'_{Ui}   // shape: (E, in_dim)

// ===== 步骤 7: 路由权重更新 =====
// 合并后路由权重 = A · 原始路由权重 (相当于原簇内 experts 路由权重求和)
merged_routing_weights = A · original_routing_weights
```

  - 关键实现细节：
    - 压缩按层从后往前执行（后层不影响前层激活），逐层获取中间激活 → 做最小二乘 → 释放内存。
    - BFloat16 精度最大化输入样本量，同时避免 GPU OOM。
    - 类似 M-SMoE，保留 N 个 expert 引用但指向 M 个实际 merged expert（矩阵 A 隐式编码）。
    - 对于含共享 experts 的 MoE 模型（DeepSeekMoE, Qwen1.5-MoE），仅压缩 routed experts，共享 experts 保持不变。
  - 批量大小与样本数配置：
    - Qwen3: ARC challenge/HellaSwag/PIQA/SQuAD 用 16 samples, WinoGrande/MRPC 用 40
    - Qwen1.5: PIQA/SQuAD 用 32 samples, 其余用 64
    - DeepSeekMoE: WinoGrande/MRPC 用 128, ARC easy/challenge/HellaSwag 用 64, 其余用 40
  - 合并时间：MergeMoE 比 M-SMoE 慢（因最小二乘法），但仍在 1 分钟内完成单任务合并（batch_size=128, Qwen1.5, WinoGrande）。
