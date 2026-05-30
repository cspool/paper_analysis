## Globally-Modulated Local Gating (GMLG)

术语是什么？
Globally-Modulated Local Gating (GMLG) 是 MoDES 提出的 expert importance 评分机制。传统的 expert skipping 方法（如 NAEE、MC-MoE、DiEP）仅依赖当前层的 local routing probability $\pi_i^{(l)}$ 决定跳过哪些 expert，忽略了不同层 expert 对模型最终输出的全局贡献差异。GMLG 将离线校准得到的层级别全局重要性因子 $\alpha^{(l)}$ 与推理时的 local routing probability 相乘，得到综合考虑全局和局部贡献的 expert importance score：$s_i^{(l)} = \alpha^{(l)} \cdot \pi_i^{(l)}$。

从算法pipeline角度拆解术语：
GMLG 的校准与推理流程：

```
# === 离线：计算 alpha ===
calib_set C = {c_1, ..., c_N}  (e.g., GQA 中 1024 样本)
for each MoE layer l in [1..L]:
    # 前向传播原始模型 → 输出概率分布
    prob_j = model.forward(c_j)  for each c_j in C

    # 前向传播跳过第 l 层所有 expert 的修改模型
    prob_j_l = model.forward(c_j, skip_all_experts_at_layer_l)  for each c_j in C

    # 计算该层的 KL 散度均值作为全局重要性
    alpha[l] = (1/N) * sum_{j=1}^{N} KL(prob_j || prob_j_l)

# 实际使用前归一化：使得 0 < s_i^{(l)} < 1
alpha_tilde[l] = alpha[l] / sum_{l'=1}^{L} alpha[l']

# === 在线推理：计算 importance score ===
for token x at layer l:
    pi = softmax(router(x))
    for i in topk_indices(pi, k):
        s_i = alpha_tilde[l] * pi[i]    # GMLG importance score
```

核心 insight：浅层 expert 被跳过时，其误差会经后续 Transformer 层逐层放大（error explosion），因此浅层的 $\alpha^{(l)}$ 更大，对应的 $s_i^{(l)}$ 也更大，更难被阈值过滤掉。

术语一般如何实现？如何使用？
- 校准数据集对结果不敏感——GQA、COCO、VMMMU 等不同数据集的 $\alpha^{(l)}$ 趋势一致（浅层大、深层小），性能也接近。
- 校准计算量：对 N 个样本的 calibration set，需要 L（MoE 层数）次额外前向传播来计算每层被跳过时的 KL 散度。20-30B 参数模型在 8×H200 上校准耗时约 20 min ~ 4 hr。
- 归一化的 $\widetilde{\alpha}^{(l)}$ 作为标量预加载，推理时仅需一次乘法（$\widetilde{\alpha}^{(l)} \times \pi_i^{(l)}$），零额外推理开销。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping
