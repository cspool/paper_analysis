## Hessian-aware Salient Data Selection (SDS, Hessian感知的显著数据选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hessian-aware Salient Data Selection（SDS）是 S²Q-VDiT 提出的面向 V-DMs 的校准数据选择策略。核心思想：在 PTQ 校准预算极端受限（仅几十样本）的情况下，随机采样导致量化性能方差极大（不同 seed 下 Imaging Quality 波动可达 ±1.76），需要一种基于数据"重要性"的筛选方法。SDS 从两个维度评估每个候选样本的重要性：(1) 扩散信息量（Diffusion Salience）C_diff = ||x_t - x_{t-1}||²/||x_t||²——相邻去噪步的隐变量变化越大，说明该 timestep 包含越多"新"的去噪信息；(2) 量化敏感度（Quantization Salience）C_quant = ||x_t^T x_t||_2——基于 Levenberg-Marquardt 近似的 Hessian 矩阵 X^T X 的 L2 范数，Hessian 特征值越大表示该样本对量化扰动越敏感。两个指标经 min-max 归一化到 [0,1] 后取乘积 C_sample = C̄_diff · C̄_quant 作为统一得分，乘积形式由算术-几何平均不等式保证只有当两个维度均高时才得高分，自然惩罚单维度强的样本。按 C_sample 降序选取 Top-N 构成校准集。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Hessian-aware Salient Data Selection (SDS)
# 输入: V-DM 模型 θ, 候选 prompt 集 P, 总去噪步数 T
# 输出: 校准数据集 D_calib (size N=40)

candidates = []
for prompt in P:  # 10 random prompts
    for t in range(1, T+1):
        x_t = model.get_latent(prompt, timestep=t)  # x_t ∈ R^{n×d}
        # (1) Diffusion Salience: 相邻步变化
        C_diff = ||x_t - x_{t-1}||² / ||x_t||²
        # (2) Quantization Salience: Hessian 近似
        C_quant = ||x_t^T @ x_t||_2  # Levenberg-Marquardt approx
        candidates.append((x_t, C_diff, C_quant))

# Min-max 归一化
C_diff_min, C_diff_max = min_max(C_diff for all)
C_quant_min, C_quant_max = min_max(C_quant for all)
for each (x_t, cd, cq) in candidates:
    cd_norm = (cd - C_diff_min) / (C_diff_max - C_diff_min)
    cq_norm = (cq - C_quant_min) / (C_quant_max - C_quant_min)
    score = cd_norm * cq_norm  # 联合得分

# 按 score 降序选 Top-N
D_calib = top_N_by_score(candidates, N=40)
```

Ablation 验证：SDS vs ATOP (All Timesteps from One Prompt) → Imaging Quality=52.95±0.69 vs 51.65±1.76；仅用 DS 或 QS 也能提升性能但联合使用(SDS)最佳；SDS 将随机种子方差从 ±1.76 降至 ±0.69。SDS 构造的校准集可集成到已有 PTQ 方法（如 PTQ4DiT + SDS 将 Aesthetic Quality 从 45.49 提升至 46.89）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SDS 的 Hessian 近似计算使用 Levenberg-Marquardt 近似 H^X = E[2 X^T X]，仅需一步矩阵乘法（X^T @ X）即可得到近似 Hessian，计算开销极小（CogVideoX-2B 仅增加 0.009 分钟、CogVideoX-5B 增加 0.015 分钟）。Attention map 在校准前用 FP 模型一次性前向计算并存储，校准时通过数据索引直接检索，不增加校准循环开销。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation

---
