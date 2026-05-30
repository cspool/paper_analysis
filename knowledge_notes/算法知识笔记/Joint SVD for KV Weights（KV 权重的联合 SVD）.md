## Joint SVD for KV Weights（KV 权重的联合 SVD）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Joint SVD for KV Weights 是 X-EcoMLA 中用于初始化 MLA 的 W^{DKV}、W^{UK}、W^{UV} 矩阵的 SVD 策略。与分别对 W^K 和 W^V 做 SVD 不同，Joint SVD 先将 W^K 和 W^V 沿列方向拼接为 [W^K, W^V] ∈ R^{d × 2·n_h·d_h}，再对拼接矩阵执行统一 SVD 分解，从而捕获 Key 和 Value 之间的相关性信息。

逻辑链：MHA/GQA 中 W^K 和 W^V 独立训练，但它们处理相同的 hidden state H，存在隐式的跨空间关联 → 拼接后做 SVD，U_kv 提取的是同时对 K 和 V 投影方向都重要的输入方向 → W^{DKV} = U_kv（共享 down-projection）→ Σ_kv V_kv^T 的前 n_h·d_h 列用于构造 W^{UK}，后 n_h·d_h 列用作 W^{UV} → 相比分别 SVD，Joint SVD 在相同 r_kv 下能更好地保留原始attention的KV联合信息。

从算法pipeline角度拆解术语：

```
# Joint SVD (X-EcoMLA)
W_KV = concat(W_K, W_V, dim=-1)      # [d, 2*n_h*d_h] —— 拼接
U, Σ, V^T = SVD(W_KV)                # 统一分解
W_DKV = U[:, :r_kv]                   # 共享 down-proj
W_UKV = Σ[:r_kv,:r_kv] @ V^T[:r_kv,:]  # [r_kv, 2*n_h*d_h]
W_UK = W_UKV[:, :n_h*d_h]   (截取前部) # key up-proj
W_UV = W_UKV[:, n_h*d_h:]   (截取后部) # value up-proj

# vs. Separate SVD (MHA2MLA 的变体)
U_k, Σ_k, V_k^T = SVD(W_K)           # 分别分解 W_K
U_v, Σ_v, V_v^T = SVD(W_V)           # 分别分解 W_V
# W_UK 仅来自 W_K 的信息，W_UV 仅来自 W_V 的信息
# 丢失了 K 和 V 之间的联合结构
```

术语一般如何实现？如何使用？

Joint SVD 的实现与普通 SVD 完全相同，仅需在调用 `torch.linalg.svd()` 前做一次 `torch.cat()`。对于大数据模型（d=4096, n_h=32, d_h=128），拼接后矩阵维度为 [4096, 8192]，经济型 SVD 的复杂度约 O(d × (2·n_h·d_h) × r_kv)，在 GPU 上计算时间 <1 秒/层。Joint SVD 特别适用于 GQA-based 模型（如 Llama 系列），因为 W_K 和 W_V 的维度天然较小。

涉及论文标题：
- X-EcoMLA__Upcycling_Pre-Trained_Attention_into_MLA_for_Efficient_and_Extreme_KV_Compression
