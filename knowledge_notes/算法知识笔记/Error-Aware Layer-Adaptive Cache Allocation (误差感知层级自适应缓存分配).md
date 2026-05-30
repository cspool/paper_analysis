## Error-Aware Layer-Adaptive Cache Allocation (误差感知层级自适应缓存分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Error-Aware Layer-Adaptive Cache Allocation 是 CompressKV 提出的层级 KV cache 预算分配策略。与依赖 attention 统计量（entropy/variance，如 PyramidKV/CAKE）不同，该方法直接量化 KV cache 压缩对每层 attention output 造成的重建误差，以此作为层级重要性的代理指标。核心思想：对压缩敏感的层（误差大）分配更多 cache budget，对压缩不敏感的层（误差小）分配更少 budget。

在离线阶段，模拟极端压缩场景（每层仅保留 m=32 tokens，约 0.3% 全量），计算每层压缩前后的 attention-block output 之间的 Frobenius norm 重建误差：

$$e^{(l)} = \sum_{t=1}^{T} \frac{\|\mathbf{O}_{\text{comp},t}^{(l)} - \mathbf{O}_{\text{full},t}^{(l)}\|_F}{\|\mathbf{O}_{\text{full},t}^{(l)}\|_F + \epsilon}$$

其中 O_full 使用完整 KV cache 的 attention output（含 output projection W_O），O_comp 使用压缩后 KV cache 的 attention output，ε=10^{-6} 防止除零。跨数据集 L1 归一化后平均，得到最终层级重要性分数 ẽ^{(l)}。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Error-Aware 分配的完整算法**：

```
# === 离线阶段：误差分数计算 ===
for each dataset d in LongBench:
    for each layer l:
        # 模拟压缩：每层仅保留 32 tokens
        K_comp^l, V_comp^l = retain_top(K_full^l, V_full^l, 32)
        for each decoding step t:
            O_full = Attention(Q_t, K_full^l, V_full^l) @ W_O^l
            O_comp = Attention(Q_t, K_comp^l, V_comp^l) @ W_O^l
            e_d^l += ||O_comp - O_full||_F / (||O_full||_F + 1e-6)
    e_hat_d^l = e_d^l / sum(e_d^k for all k)     # L1 norm within dataset

e_bar^l = mean(e_hat_d^l for all d in datasets)   # cross-dataset average
e_tilde^l = e_bar^l / sum(e_bar^k for all k)      # final importance

# === 在线阶段：预算分配 (Algorithm 1) ===
B_i = m  for all layers i                         # m = 32 minimum
R = B_total - sum(B_i)                            # remaining
B_i = clip(B_i + round(e_tilde_i * R), m, M)     # M = 3 * B_per_layer
delta = B_total - sum(B_i)
while delta != 0:
    if delta > 0:
        j = argmax(e_tilde_i for i where B_i < M)
        B_j += 1; delta -= 1
    else:
        j = argmin(e_tilde_i for i where B_i > m)
        B_j -= 1; delta += 1
return B
```

术语一般如何实现？如何使用？

离线误差计算在 LongBench 全部 16 个数据集上进行，取平均值确保不依赖特定 task。上下界 m=32 和 M=3×B_per-layer 通过实验调优。在 Mistral-7B 和 Llama-3.1-8B 上，不同模型的误差分布显著不同，验证了 error-aware 方法捕捉到了模型特定的层级差异。与 CAKE/PyramidKV 的关键差异：(a) 离线计算无在线开销；(b) 基于真实压缩误差而非 attention 统计量代理指标，跨模型泛化性更好。代码开源：https://github.com/TUDa-HWAI/CompressKV.git。实现包含 `longbench/get_avg.py` 用于跨数据集平均误差分数。

涉及论文标题：
- CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation

---
