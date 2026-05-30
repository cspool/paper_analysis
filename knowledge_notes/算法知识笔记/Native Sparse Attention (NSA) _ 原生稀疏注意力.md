## Native Sparse Attention (NSA) / 原生稀疏注意力

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Native Sparse Attention (NSA, Yuan et al., 2025c) 是一种 learnable、hardware-aware 的稀疏注意力机制，替代 Transformers 中的标准 dense causal attention。NSA 不计算所有 query-key 对之间的 attention，而是对每个 query q_t 动态构建信息密集的 KV cache 子集。NSA 通过三个互补的支路实现稀疏化：(1) Token Compression Branch (CMP)：将连续 key blocks 通过 learnable MLP φ 聚合为粗粒度 block-level 表示，减少空间/时间冗余；(2) Token Selection Branch (SLC)：计算每个 KV block 的 importance score，选择 top-n 最重要的 blocks 保留细粒度信息；(3) Sliding Window Branch (SWA)：保留最近 w 个 KV pairs，确保局部上下文连续。三支路输出通过 learnable gate（两层 MLP + sigmoid）动态加权融合。NSA 的核心优势：数据依赖的稀疏性（data-dependent sparsity），即稀疏模式根据输入内容动态确定，而非使用固定的局部窗口/跨步 pattern。作为 hardware-aware 设计，NSA 的 block-level 分区与 GPU Tensor Cores 的 tile-based 计算对齐，在 H100 等硬件上可实现实际加速。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# NSA Attention Computation (per head, per timestep t)
# Input: q_t, K_{<=t} ∈ R^{t×d}, V_{<=t} ∈ R^{t×d}
# Hyperparams: block_size s, num_blocks n, window_size w

# Branch 1: Token Compression (CMP)
K_blocks = reshape(K[:t], [-1, s, d])
k_cmp = φ(mean(K_blocks, dim=1))         # MLP φ compresses block to vector
v_cmp = φ_v(mean(reshape(V[:t], [-1,s,d]), dim=1))
o_cmp = softmax(q_t @ k_cmp^T / sqrt(d)) @ v_cmp

# Branch 2: Token Selection (SLC)
p = importance_score(q_t, K_blocks)       # per-block importance
top_idx = topk(p, n)                      # top-n blocks
k_slc = gather(K_blocks, top_idx)
v_slc = gather(V_blocks, top_idx)
o_slc = softmax(q_t @ k_slc^T / sqrt(d)) @ v_slc

# Branch 3: Sliding Window (SWA)
k_swa = K[t-w:t]; v_swa = V[t-w:t]
o_swa = softmax(q_t @ k_swa^T / sqrt(d)) @ v_swa

# Dynamic Gating
g_cmp, g_slc, g_swa = sigmoid(MLP_gate(q_t))
o_t = g_cmp*o_cmp + g_slc*o_slc + g_swa*o_swa
```

Annotations: s=64 (block size), n=32 (selected blocks), w=256 (window) → total attention budget K_attn = 64×32 + 256 = 2304。在 L=128K context 下 γ = 2(K_attn)/(L-1) ≈ 3.6%。importance_score 通常为低秩近似。compression MLP φ: d→64→d (两层，SiLU activation)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NSA 原始实现在 FLA (Flash Linear Attention) 库（Yang & Zhang, 2024），基于 Triton 编写。VideoNSA 基于 FLA 并适配 SWIFT 训练框架（Zhao et al., 2024）。融合到 Video-LLM 时，vision tokens 使用 NSA（block size = 每帧 token 数），text tokens 保持 dense GQA。训练需联合优化 QKV + gate + compression MLP。开源：https://github.com/mdy666/Scalable-Flash-Native-Sparse-Attention (NSA 原始)；https://github.com/Espere-1119-Song/VideoNSA (VideoNSA)。

涉及论文标题：
- VideoNSA__Native_Sparse_Attention_Scales_Video_Understanding
