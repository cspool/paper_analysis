## Dynamic Attention Mask (DAM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Dynamic Attention Mask (DAM) 是一种免微调的动态稀疏注意力机制，为 Transformer 模型中每个 layer 和 attention head 的 attention map 分配自适应的稀疏 mask。DAM 通过两阶段流程运作：(1) **离线 mask 生成阶段**：冻结的预训练模型在 Pattern Capture Length (PCL) 范围内处理输入序列提取完整 attention map → Box-Cox 变换放大中小注意力值 → 全局归一化后以阈值 τ 二值化生成 "true mask" → 通过结构模式匹配（对角线 + 垂直模式）识别 pattern → 对超 PCL 长度外推生成 "extended mask"；(2) **推理应用阶段**：将 mask 在 softmax 前以 Hadamard product 应用于 attention score，mask 位置设为 -∞。FLOPs 从 O(L²) 降至 O(sL)（s 为每 query 平均保留 key 数，s ≪ L）。与 MoA 的预定义静态 mask 不同，DAM 从真实 attention 分布中捕获每层每头的异构模式，无需 fine-tuning。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// === Stage 1: Offline Mask Generation ===
// Input: calibration dataset, frozen LLaMA model, PCL L
// Output: per-layer per-head binary masks M_{ℓ,h}

for each sequence in dataset:
    L_eff = min(len(sequence), L)
    attn_maps = frozen_model(sequence[:L_eff])  // extract full attention

// Accumulate mean attention across dataset
for each layer ℓ, head h, positions (i,j):
    A_mean = A_accumulated / (C_count + ε)  // ε = 10^{-8}

// Box-Cox feature amplification (λ=0.5)
X = max(A_mean, ε)
B = (X^{0.5} - 1) / 0.5  // Box-Cox with λ=0.5
A_tilde = B - min_all(B)   // global normalization

// True mask generation (τ=0.3)
M_true[i,j] = 1 if A_tilde[i,j] >= τ else 0

// Structural pattern matching (μ=0.8)
Pattern pool P = {P_diag,r: j=i-r} ∪ {P_vert,c: j=c, i≥c}
for each P_k in P:
    γ_k = sum(M_true * P_k) / sum(P_k)
    if γ_k >= μ: add P_k to matched

// Build extended mask for length S > L
M_ext[i,j] = 1 if any matched P_k has P_k_ext[i,j] == 1 else 0

// === Stage 2: Online Inference ===
attention_scores = Q @ K^T / sqrt(d_k)
attention_scores = attention_scores ⊙ M    // Hadamard product
attention_scores[M == 0] = -∞              // discard masked
O = softmax(attention_scores) @ V
```

术语一般如何实现？如何使用？

基于 PyTorch + HuggingFace Transformers 实现。Stage 1 离线运行一次在 Multi-News 上，生成 {layer, head} → mask mapping 文件；Stage 2 加载 mask 替代标准 attention mask。PCL=512 由 A100 40GB 的显存约束决定。mask 与 tile-based GPU 执行兼容且可与 FlashAttention 融合。开源：https://github.com/HanzhiZhang-Ulrica/DAM。

涉及论文标题：
- DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration
