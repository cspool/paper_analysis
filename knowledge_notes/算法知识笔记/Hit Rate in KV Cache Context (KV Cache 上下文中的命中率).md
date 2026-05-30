## Hit Rate in KV Cache Context (KV Cache 上下文中的命中率)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hit Rate（命中率，记作 H）是 SnapKV 论文专门设计的一个量化指标，用于评估 observation window-based voting 机制在识别重要 attention features 方面的有效性。它衡量的是：通过 observation window 投票选出的"重要"attention features 中，有多大比例在后续生成阶段确实保持了高 attention weights。

形式化定义（Eq. 4-8）：给定注意力阈值 θ，将生成阶段 attention weights 超过 θ 的 prefix 位置标记为"实际重要的"（M_threshold_cur），将 observation window 投票选出的位置标记为"预测重要的"（M_vote_obs）。Hit Rate 是两者的交集大小与实际重要位置总数的比值：H = |M_threshold_cur ∩ M_vote_obs| / |M_threshold_cur|。H ∈ [0, 1]，越接近 1 表示投票机制越准确。

SnapKV 使用 hit rate 进行了两项鲁棒性分析：(a) Contextual Dependency——不同指令在相同文档上选出的重要特征差异较大（hit rate 下降），证明 KV 压缩需要 context-aware 策略；(b) Invariance to Instruction Positions——无论指令在 prompt 开头还是末尾，hit rate 均保持高位，证明 SnapKV 的 observation window 机制对指令位置鲁棒。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Hit Rate 计算流程（per head, per layer）
# 输入: A_cur ∈ R^{L_prefix}       当前生成 query 对 prefix keys 的 attention weights
#       M_vote_obs ∈ {0,1}^{L_prefix} observation window 投票选出的位置掩码
#       θ                           attention 阈值

def compute_hit_rate(A_cur, M_vote_obs, theta):
    # Step 1: 标记当前生成中"实际重要的"features
    M_threshold_cur = (A_cur > theta).float()  # {0,1}^{L_prefix}

    # Step 2: 计算命中(交集)
    O = M_threshold_cur * M_vote_obs  # 逐元素与, {0,1}^{L_prefix}

    # Step 3: 计算命中率
    H = O.sum() / (M_threshold_cur.sum() + eps)

    return H  # ∈ [0, 1]

# SnapKV 论文中用于鲁棒性分析的变体：
# H(M_vote_A, M_vote_B) — 两组不同投票结果的命中率
# 用于衡量不同 instruction-response pairs 在同一文档上的重要特征一致性
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Hit Rate 主要作为分析工具而非运行时指标——在运行时不需要计算 hit rate（因为生成阶段的实际 attention weights 此时未知）。其用途包括：(1) 验证观察窗口大小选择的合理性；(2) 比较不同 voting strategy 的预测质量；(3) 分析不同数据集、不同指令类型对注意力模式的影响。实现上仅在离线分析/消融实验中使用，不产生运行时开销。

涉及论文标题：
- SnapKV: LLM Knows What You are Looking for Before Generation
