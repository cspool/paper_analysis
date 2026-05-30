## Weight Decay in Muon (Muon中的权重衰减 / L2正则化在矩阵正交化优化器中的应用)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Decay（权重衰减）是 AdamW (Loshchilov & Hutter, 2019) 中提出的 decoupled weight decay 机制：W_t = W_{t-1} - η_t (∇L_effective + λ W_{t-1})，即直接将 λW 作为独立项加入更新而非通过 L2 正则化嵌入损失函数。在原始 Muon (Keller Jordan et al. 2024) 中未包含 weight decay。Liu et al. (2025) 发现在大规模训练中，vanilla Muon 的权重 RMS 和层输出 RMS 持续增长超出 bf16 表示范围（图 2：初期收敛快，但长期被 AdamW 超越），引入 AdamW 风格的 weight decay (λ=0.1) 后解决了此问题——Muon + weight decay 在 over-train 区间持续优于 AdamW。更新公式变为：W_t = W_{t-1} - η_t (0.2·O_t·√(max(A,B)) + λ W_{t-1})。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Weight Decay 在 Muon 中的更新流程：

```
# 对矩阵参数 W ∈ R^{A×B}：
O_t = Newton-Schulz(M_t)                        # 正交化 momentum

# 两项独立相加 (与 AdamW 风格一致)
gradient_update = 0.2 * O_t * sqrt(max(A, B))   # 正交化更新 (RMS ~0.2)
weight_decay_term = λ * W_{t-1}                  # λ = 0.1

W_t = W_{t-1} - lr * (gradient_update + weight_decay_term)
```

关键效果 (图 2)：
- 无 weight decay 的 vanilla Muon：初期收敛最快（红色曲线），但约 40B tokens 后被 Muon+weight decay（蓝色）超越，最终高于 AdamW（绿色）
- Muon + weight decay：全程优于 AdamW，在 100B tokens（~5× optimal）时仍保持优势
- Weight decay 有效抑制了大矩阵（如 MLP [H, 2.6H]）在长期训练中权重 RMS 的发散问题

论文还特别指出对 RMSNorm 的 gamma 参数施加 weight decay 对训练稳定性至关重要——防止每层输出 RMS 过高。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- 在 PyTorch 中，通过优化器的 `param_groups` 为不同参数组设置不同的 weight_decay 值，矩阵参数组与非矩阵参数组可共享相同的 λ
- λ 的典型值：0.1（论文所有训练阶段一致使用），与常见 AdamW 设置兼容
- 与 L2 正则化的区别：Weight Decay 直接作用在参数上（λW），而 L2 正则化作用于梯度（∂(λ||W||²)/∂W = 2λW，与自适应学习率交互后不等价）
- 对于 Muon 中的 weight decay，由于 Newton-Schulz 正交化已标准化了梯度方向，weight decay 是主要控制参数范数增长的机制
- 原始 Muon 仓库在论文发表后的 commit (e0ffefd) 中同步添加了 weight decay 支持

涉及论文标题：
- Muon is Scalable for LLM Training
