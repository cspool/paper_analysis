## FFN Scaling Imbalance (d_ff/d_h Ratio)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FFN Scaling Imbalance 是 FlashMHF 论文识别的 naïve Multi-Head FFN 在大模型 scale 上失效的根本原因。问题本质：MH-FFN 继承 MHA 的设计惯例——d_h 固定（如 128）而 d_ff 随模型 scale 增长（因模型参数总量增长要求）。这导致 d_ff/d_h ratio 随模型变大而失衡：128M 模型 d_ff/d_h = 2048/128 = 16；370M 模型 = 2752/128 = 21.5；1.3B 模型 = 5760/128 = 45。根据 Kaplan et al. (2020) 的 scaling law，FFN 的 d_ff/d_model ratio 存在最优范围（经验值约 8/3），偏离此范围会导致 parameter efficiency 下降。在 MH-FFN 语境下，d_ff/d_h 的角色等同于标准 FFN 的 d_ff/d_model——每 head 的 internal capacity (d_ff) 与 input dimension (d_h) 之比若过大，单个 head 的参数利用效率低，部分 d_ff 维的 capacity 被浪费或未被有效利用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Scaling Imbalance 的量化分析（FlashMHF Table 4 数据）:

模型 Scale   d_model   d_ff    H    d_h    d_ff/d_h   问题严重度
─────────────────────────────────────────────────────────
128M         892       2048    6    128    16.0       中等失衡（MH-FFN 仍优于 baseline）
370M         1024      2752    8    128    21.5       显著失衡（MH-FFN = baseline）
1.3B         2048      5760    16   128    45.0       严重失衡（MH-FFN << baseline 预期）

# 此时标准 SwiGLU 的 d_ff/d_model ratio:
# 128M: 2048/892  ≈ 2.30  ≈ 8/3.5  (接近最优)
# 370M: 2752/1024 ≈ 2.69  ≈ 8/3    (接近最优)
# 1.3B: 5760/2048 ≈ 2.81  ≈ 8/3    (接近最优)

# FlashMHF 的解决方案——引入 E 个子网络:
# 每子网络 internal dim d_e ≈ 8/3·d_h = 8/3·128 ≈ 342
# 128M: E=8, d_ff_total = 8×342 = 2736, d_ff/d_h = d_e/d_h = 342/128 ≈ 2.67 ≈ 8/3 ✓
# 370M: E=7, d_ff_total = 7×342 = 2394, d_ff/d_h = d_e/d_h = 342/128 ≈ 2.67 ≈ 8/3 ✓
# 1.3B: E=15, d_ff_total = 15×342 = 5130, d_ff/d_h = d_e/d_h = 342/128 ≈ 2.67 ≈ 8/3 ✓

# 关键洞察: FlashMHF 通过 sub-network 分解将 ratio 锁定在 d_e/d_h ≈ 8/3,
#           而非 d_ff/d_h。每 sub-network 内 ratio 平衡，多个 sub-network 并联
#           提供足够的 total capacity 和 representational diversity。
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

该术语的实际使用：(1) 诊断工具——当设计新的 FFN 变体时，检查 d_ff/d_h（或等效等效扩展比例）是否在 2-4 范围内。若超出此范围，预期 parameter efficiency 下降；(2) 解决策略——FlashMHF 采用 parallel sub-network decomposition（dense MoE）将单一路径拆解为 E 个 balanced-ratio 的 sub-path，其他可能的策略包括增加 d_h（但减少 H 降低 diversity）、调整 d_ff 增长策略、或使用 non-uniform head sizes；(3) 跨模型 scale 的行为预测——此 ratio 可用于预测新 FFN 架构在不同模型 size 下的 scalability；(4) 实验验证——FlashMHF 的 128M vs 370M 消融实验提供了 direct evidence：MH-FFN 从 128M 的 gain 到 370M 的 failure，唯一的变化就是 d_ff/d_h 从 16→21（ratio 恶化），而加入 parallel sub-network 后恢复 gain。

涉及论文标题：
- Flash Multi-Head Feed-Forward Network
- Scaling Laws for Neural Language Models (Kaplan et al., 2020)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block-Level Attention Sparsity via Softmax Thresholding（BLASST）是一种训练无关的动态稀疏注意力方法。在FlashAttention的block-wise online softmax过程中，对每个KV block计算local maximum score m̃_i^{(j)}，并与running maximum m_i^{(j)}比较。当 m̃_i^{(j)} - m_i^{(j)} < ln(λ) 时（即block的局部最大值远小于已见最大值），跳过该block的后续计算。推导逻辑：(1) softmax中每个score的指数exp(s_ij)都会除以全局exp最大值做数值稳定，(2) 因此block中所有score的贡献被exp(m̃_i^{(j)} - m_i^{(j)}) < λ上界限制，(3) 当λ足够小时，block对最终输出的贡献可忽略。跳过三项操作：softmax指数计算（CUDA core MUFU.EX2）、attention-value矩阵乘法（tensor core MMA）、Value block的HBM加载（仅decode kernel）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

BLASST的算法pipeline（基于FlashAttention的tiled online softmax改造）：

```
# Input: Q∈R^{L×d}, K∈R^{L×d}, V∈R^{L×d}, threshold λ
# Tiling: Q→T_r blocks of B_r, K/V→T_c blocks of B_c

for i in 1..T_r:                    # 遍历query blocks
    m = -∞, O = 0, l = 0            # 初始化online softmax状态
    for j in 1..T_c:                # 遍历KV blocks
        S_ij = Q_i × K_j^T          # [B_r×B_c] QK^T, tensor core BMM1
        m_local = rowmax(S_ij)      # block local maximum
        m_new = max(m, m_local)     # 更新running maximum
        
        if m_local - m_new < ln(λ): # BLASST核心: 跳过检查
            continue                # 跳过softmax + PV乘法 + V加载
        else:
            P_tilde = exp(S_ij - m_new)           # softmax (MUFU.EX2)
            l = exp(m - m_new)*l + rowsum(P_tilde) # 更新归一化因子
            O = exp(m - m_new)*O + P_tilde × V_j   # PV matmul (BMM2)
            m = m_new
    O_i = O / l                      # final renormalization
return {O_i}
```

关键特性：(1) skip decision使用已在FlashAttention中计算好的统计量（local max, running max），零额外overhead；(2) exp(m_local - m_new) ≤ λ保证了被跳过block的输出误差有理论上界（Appendix B）；(3) 同一阈值λ适用于所有attention head和layer，自动适应不同head的稀疏度分布。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

BLASST已集成到TensorRT-LLM（https://github.com/NVIDIA/TensorRT-LLM）和FlashInfer中。使用时仅需在attention接口传入一个scalar threshold λ。λ通过校准自动确定：在校准数据集上sweep不同threshold，记录(λ·L, sparsity)数据点，拟合 λ·L = α·exp(β·s)。推理时给定目标sparsity S和context length L，直接用 λ = α·exp(β·S)/L。Sparsity-aware training变体在fine-tuning的forward pass中应用BLASST，backward中被跳过block自然不收梯度，迫使模型将重要信息集中到高attention score block。

涉及论文标题：
- BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding
