## Row-Column Binarization (ARB-RC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ARB-RC 是 ARB-LLM 中针对 LLM 权重列间偏差设计的双轴缩放二值化方法。传统二值化仅用 row-wise 参数（α^r 和 μ），无法处理 LLM 中某些列值显著大于其他列的现象。ARB-RC 引入 column-wise scaling factor α^c，模型为 Ŵ = α^r·α^c·B（无 μ）。移除 μ 节省存储同时 α^c 保留列间偏差。初始值：α^r_i = mean(|W_i|), α^c_j = mean(|W_{·,j}/α^r|)。交替更新：α^r = diag(W(α^cB)^T)/diag((α^cB)(α^cB)^T)，α^c = diag(W^T(α^rB))/diag((α^rB)^T(α^rB))。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# init: α^r = mean(|W|, dim=1); α^c = mean(|W|/α^r, dim=0); B = sign(W)
for iter in 1..T:
    # update α^r: A = α^c.unsqueeze(0) * B  → α^r = diag(W @ A^T) / diag(A @ A^T)
    # update α^c: A = α^r.unsqueeze(1) * B  → α^c = diag(W^T @ A) / sum(A², dim=0)
    B = sign(W)   # ARB-RC 无 μ
# Ŵ[i,j] = α^r[i] · α^c[j] · B[i,j]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ARB-RC 优于 ARB-X：LLaMA-7B ppl 14.03 vs 21.81，且存储更少（2.63GB vs 2.93GB raw bitmap）。ARB-RC 不引入 calibration data（L₂ 优化导致参数耦合），仅用 L₁ 优化。+ CGB = ARB-LLM_RC 最终模型。

涉及论文标题：
- ARB-LLM Alternating Refined Binarizations for Large Language Models

---
