## Residency-Aware Biasing (MoE Expert Selection / 驻留感知偏置路由)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Residency-Aware Biasing 是 MoE-ERAS 提出的第二种 residency-aware expert 选择算法，是对 thresholding 的改进。在 Softmax 之前对 off-chip expert 的 logits 施加基于激活频率的差异化惩罚：penalty = β(1 - freq(E_i))，其中 freq(E_i) 是从 profiling 阶段（500k tokens）收集的归一化激活频率。核心洞察：(1) 不同 expert 的"热度"差异显著——hot expert 频繁被激活，cold expert 很少被激活；(2) 加载一个冷门 off-chip expert 到 HBM 大概率很快被 LRU evict（两次 swap），比加载热门 off-chip expert 代价更大。Biasing 通过频率加权的惩罚体现这一差异——冷门 expert 惩罚大（几乎不会被选中加载），热门 expert 惩罚小（值得加载因为后续会复用）。

从算法pipeline角度拆解术语：
```
# 标准 MoE Gating
Logits = H_i @ W_exp               # [seq_len, num_experts]

# === Residency-Aware Biasing ===
# freq[e] ∈ [0, 1]: 归一化激活频率（从 profiling 收集）
# residency[e] = True 表示 expert e 在 HBM 中
for e in range(num_experts):
    if not residency[e]:  # expert 在 slow memory (CPU)
        Logits[:, e] -= beta * (1 - freq[e])

Weights = Softmax(Logits)           # 惩罚后重新归一化
Selected = SelectTopK(Weights, k=2)
```
关键与 thresholding 的区别：(1) 操作在 logits 层面（Softmax 前），而非 probabilities 层面；(2) 惩罚是差异化的（考虑 freq），而非统一 α；(3) 热门 off-chip expert（freq ≈ 0.8, penalty ≈ 0.2β）惩罚小，冷门 off-chip expert（freq ≈ 0.05, penalty ≈ 0.95β）惩罚大。

从算法pipeline角度拆解术语，给出具体例子：
以 Mixtral-8x7B layer i，profiling 得 freq = [0.20, 0.18, 0.05, 0.22, 0.08, 0.15, 0.10, 0.02]，当前 expert 0, 3, 7 在 CPU，β=1.0：
- Logits = [2.1, 1.9, 1.5, 2.0, 0.8, 0.5, 1.2, 0.3]
- Biasing 调整：
  - expert 0 (CPU, freq=0.20): Logits -= 1.0×(1-0.20) = Logits[0] -= 0.80 → 1.30
  - expert 3 (CPU, freq=0.22): Logits -= 1.0×(1-0.22) = Logits[3] -= 0.78 → 1.22
  - expert 7 (CPU, freq=0.02): Logits -= 1.0×(1-0.02) = Logits[7] -= 0.98 → -0.68
  - HBM experts (1,2,4,5,6): Logits 不变
- 调整后 Logits = [1.30, 1.9, 1.5, 1.22, 0.8, 0.5, 1.2, -0.68]
- Softmax 后 Top-2 可能为 expert 1 (HBM) + expert 2 (HBM)，避免传输。
- 对比 thresholding：热门 CPU expert 0 (freq=0.20) 惩罚较小，仍可能被选中（若其 logit 显著高于 HBM expert）；冷门 CPU expert 7 (freq=0.02) 几乎不可能被选中。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Profiling 依赖性：需在 serving 前运行 profiling（CNN DailyMail, 139k-500k tokens）收集每层每个 expert 的激活频率。论文用线性回归验证 expert activation predictability（仅用前 4 层预测后 28 层，accuracy > 50%，远超随机 12.5%）。
- 实现：在 serving 框架中维护 freq 查找表（从 profiling 结果加载）。每个 MoE layer 的 gating→TopK 之间插入 biasing 逻辑。
- 使用：β=1.0 时减少 8.0-9.7% 解码延迟。Quality trade-off 优于同 speedup 水平的 thresholding——因为 biasing 的差异化惩罚更智能。
- 局限性：(1) 依赖 profiling 数据质量，distribution shift 可能导致 freq 不准确（论文建议 periodic re-calibration）；(2) 当前仅在 Mixtral-8x7B 上实现。

涉及论文标题：
- MoE-ERAS: Expert Residency Aware Selection

---
