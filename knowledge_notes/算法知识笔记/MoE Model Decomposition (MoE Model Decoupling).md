## MoE Model Decomposition (MoE Model Decoupling)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoE Model Decomposition（模型解耦）是 MoE-DisCo 的核心操作——将完整 MoE Θ = (θ_shared, θ_1, ..., θ_E) 分解为 E 个独立 dense 子模型 Θ_k = (θ_shared^(k), θ_k)。每个子模型包含：(1) 完整共享 backbone（embedding、attention、LayerNorm），参数被复制 E 份；(2) 仅一个 expert，所有 MoE 层移除 gating，固定使用该 expert。分解后子模型退化为标准 dense Transformer，参数量远小于完整 MoE。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 Qwen1.5-MoE-2.7B（E=4）为例：

```
# 原始 MoE 每层：Input → Attn → LayerNorm → Gating → Top-K Experts → Output
# 子模型 k 每层：  Input → Attn → LayerNorm → Expert_k FFN → Output
#                                           ↑ gating 移除，固定 expert k

# 分解
原始: Θ = (θ_shared, θ_1, θ_2, θ_3, θ_4)
分解: Θ_1 = (θ_shared^(1), θ_1), ..., Θ_4 = (θ_shared^(4), θ_4)

# 重组（Reintegration）
θ_exp* = Concat(θ_1, ..., θ_4)             # expert 拼接
θ_shared* = Σ γ_k · θ_shared^(k)           # WP-SGD 加权平均
Θ = (θ_shared*, θ_exp*)                     # 完整 MoE
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
关键优势：(1) 子模型约 1/E expert 参数 + 1 份共享参数，可放入 RTX 4090 24GB；(2) 训练完全独立无需分布式框架（NCCL/GLOO）；(3) embarrassingly parallel——expert 数量增加时边际成本近常数。Qwen1.5-MoE-2.7B 上 S-phase 成本仅 $1.79-$4.37（4×RTX 4090），远低于 baseline $6.93-$29.91（A100）。

涉及论文标题：
- MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models
