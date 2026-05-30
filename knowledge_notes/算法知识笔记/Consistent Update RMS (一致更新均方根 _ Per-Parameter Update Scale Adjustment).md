## Consistent Update RMS (一致更新均方根 / Per-Parameter Update Scale Adjustment)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Consistent Update RMS 是 Muon 扩展到大规模 LLM 训练时的关键技术之一（Liu et al. 2025）。其核心问题来自 Lemma 1：shape 为 [A, B] 的满秩矩阵经 Muon 更新后，理论更新 RMS = √(1/max(A,B))。这意味着不同 shape 的矩阵参数（如 attention QKV 的 [H, H] vs MLP 的 [H, 2.6H] vs 独立 KV head 的小矩阵）会有差异极大的更新尺度：(1) 大矩阵（max(A,B) 大，如 MLP up-projection）更新过小，限制模型容量；(2) 小矩阵（max(A,B) 小，如 GQA/MLA 中独立 KV head）更新过大，导致训练不稳定。解决方案：对每个矩阵参数按 √(max(A,B)) 缩放其 Muon 更新，再乘以 0.2 因子以匹配 AdamW 的经验更新 RMS 范围（0.2~0.4），使所有矩阵参数在不同 shape 下具有一致的更新尺度，且可直接复用 AdamW 调优的 lr 和 weight decay。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Consistent Update RMS 在 Muon 更新中的具体实现（Adjusted LR 方法）：

```
# 对每个矩阵参数 W，其 shape = [A, B]
O_t = Newton-Schulz(M_t)                        # 正交化后的更新方向
                                                 # O_t 的 RMS ≈ √(1/max(A,B)) (Lemma 1)

# Adjusted LR: 按 shape 缩放以取消 Lemma 1 的效应
scale = 0.2 * sqrt(max(A, B))                   # 使最终 update RMS ≈ 0.2
update = scale * O_t                             # 现在 update RMS ≈ 0.2，与 AdamW 一致

W_t = W_{t-1} - lr * (update + λ * W_{t-1})     # lr 和 λ 直接复用 AdamW 的 optimal 值
```

与其他方法的对比（消融实验，Table 1）：
1. Baseline: scale = 0.2 * √H（H=hidden size），对大矩阵 [H, 4H] 更新不足，validation loss 2.812
2. Update Norm: O_t / RMS(O_t) * 0.2，直接归一化更新 RMS 到 0.2，对所有矩阵一视同仁，但忽略了不同 shape 应有的不同行为，validation loss 2.789
3. Adjusted LR: √(max(A,B)) 缩放，既保持不同 shape 矩阵的自然差异，又使 RMS 与 AdamW 一致，validation loss 2.789（与 Update Norm 相当但计算开销更低）

Adjusted LR 被选为最终方案，因其在 MLP 权重上有效提升 RMS（相对 Baseline 翻倍），同时在 attention QKV 矩阵上保持与 Baseline 一致的 RMS（因 max(H,H)=H，√(max)/√H=1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- 在优化器的 `step()` 函数中，对每个矩阵参数组计算 `max(param.shape[0], param.shape[1])`，乘以 0.2 作为缩放因子
- 该缩放是纯标量操作，计算开销可忽略不计
- 需与 weight decay 配合使用，缩放仅作用于正交化更新 O_t，不作用于 weight decay 项 λW_{t-1}
- 设置 0.2 的理由来自消融实验（Table 8）：在 [0.05, 0.1, 0.2, 0.4, 0.8] 范围内，0.2 和 0.4 表现相当且明显优于其他值，0.2 被选择以与 AdamW 经验范围的下限对齐
- 原始 Muon (Keller Jordan) 的缩放方式为 √(max(1, A/B))，在矩阵 second dimension 相同时等价于本文方案

涉及论文标题：
- Muon is Scalable for LLM Training
