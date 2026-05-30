## Expert Skipping

术语是什么？
Expert Skipping（专家跳过）是一种针对 Mixture-of-Experts (MoE) 模型的推理加速技术。其核心思想是：MoE 的 top-k router 为每个 token 选择 k 个 expert，但并非所有被选中的 expert 都对当前 token 的输出有实质性贡献。Expert Skipping 通过在推理时动态识别并跳过（deactivate）冗余 expert，减少实际执行的 expert 数量，从而降低计算开销。与 training-aware 的 MoE 效率优化（如 load-balanced routing）不同，Expert Skipping 是 **training-free** 方法，直接应用于已训练好的 MoE 模型，无需重新训练或访问训练数据。

从算法pipeline角度拆解术语：
MoDES 中的 Expert Skipping 全流程（以单 token 经过第 l 层 MoE FFN 为例）：

```
# 离线阶段：校准（per model, 一次执行）
calib_set = 随机采样 1024 条数据 (GQA)
for each MoE layer l in [1..L]:
    prob_orig = model.forward(calib_set)           # 原始输出概率
    prob_skip_l = model.forward(calib_set,         # 跳过第 l 层所有 expert
                                skip_experts_at_layer=l)
    alpha[l] = mean(KL(prob_orig || prob_skip_l))  # Eq.(4): 层全局重要性
alpha_tilde = alpha / sum(alpha)                   # 跨层归一化

# 在线推理阶段：Dynamic Expert Skipping
for each token x:
    modality = "text" if is_text_token(x) else "vision"
    for each MoE layer l:
        r = router(x)                              # (M,) routing logits
        pi = softmax(r)                            # (M,) routing probs
        S = topk_indices(pi, k)                    # top-k expert indices
        for i in S:
            s_i = alpha_tilde[l] * pi[i]           # Eq.(3): importance score
            if s_i < threshold[modality]:          # Eq.(5): DMT
                skip Expert_i
        y = sum(pi[i] * Expert_i(x) for i in kept) # 仅保留的 expert 参与计算
```

关键设计：(1) 浅层 expert 的 $\alpha^{(l)}$ 更大 → 更难被跳过 → 保护关键层；(2) Vision token 阈值 $\tau_v$ > text token 阈值 $\tau_t$ → 更激进跳过 vision expert。

术语一般如何实现？如何使用？
- **离线校准**：使用小规模 calibration set (~1024 样本) 计算层的全局重要性因子 $\alpha^{(l)}$ + 搜索最优阈值，耗时 20 min ~ 4 hr (取决于模型大小和硬件)。
- **在线推理**：$\widetilde{\alpha}^{(l)}$ 和阈值 pair $(\tau_t, \tau_v)$ 预加载，每次 expert skipping 决策仅需对 top-k 个路由概率做 element-wise 乘法 + 比较，无额外推理开销。
- **适用场景**：已训练好的 MoE 模型（LLM 或 MLLM），尤其是 top-k > 2 的场景。无需重新训练、无需访问训练数据。
- **与其他技术结合**：可与模型量化（混合精度量化）正交叠加——MoDES 决定跳过哪些 expert，量化压缩保留 expert 的参数精度。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping
