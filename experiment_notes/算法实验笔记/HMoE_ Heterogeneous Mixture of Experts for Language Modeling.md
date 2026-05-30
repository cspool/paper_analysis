## HMoE: Heterogeneous Mixture of Experts for Language Modeling

- 属于算法pipeline的实现是什么？实验比较什么？
  - HMoE 提出异构 Mixture of Experts 预训练语言模型，核心实现：(1) 为 MoE 层中的不同 expert 分配不同的 FFN hidden dimension（即不同大小的 expert），以引入专家异质性。每个 expert FFN 沿用 LLaMA 的 SiLU-gated 设计：e_i(x) = W_{o,i} · (SiLU(W_{g,i} · x) ⊙ (W_{p,i} · x))，其中 W_{g,i} ∈ R^{h_input × h_ffn,i}, W_{p,i} ∈ R^{h_input × h_ffn,i}, W_{o,i} ∈ R^{h_ffn,i × h_input}，通过改变 h_ffn,i 控制各 expert 大小；(2) 提出 P-Penalty loss (Parameter Penalty) L_P-Penalty = N · Σ M_i · P̂_i，其中 M_i = (1/T) Σ 1{e_i ∈ E^t} × h_ffn,i，将 expert 大小纳入损失，鼓励激活更小的 expert，防止 router 过度偏好大 expert；(3) 配合 Top-P routing 时额外使用 router entropy loss L_entropy = N · Σ P_i · log(P_i) 抑制激活 expert 数量增长；(4) 探索三种 expert 大小分布策略：Geometric（几何级数如 {1,2,4,8,16,32,64,128}）、Arithmetic（等差级数如 {9,11,13,15,17,19,21,23}）、Hybrid（混合如 {1,1,1,1,2,2,4,4}）。
  - 实验比较：(1) HMoE vs Homogeneous MoE vs Dense：0.4B 和 3B 总参数量级，在等 FLOPs 预算下对比 Top-K (k=2) 和 Top-P (p=0.6) 路由；(2) isoFLOP 分析：不同训练 FLOPs 下 HMoE vs Homogeneous MoE 的最优激活参数量和 loss 曲线；(3) Ablation：P-Penalty loss vs load balancing loss、三种 expert 分布策略 (geometric/arithmetic/hybrid)、不同 expert 大小方差（最大/最小 expert size ratio）的影响；(4) Expert 分析：expert 间相似度 (Wasserstein distance)、协同度 (KL divergence)、不同难度 token 的 expert 激活模式、层间激活参数分布。

- 硬件平台是什么，配置是什么。
  - 训练硬件：NVIDIA A800 (80GB 显存) 或 H800 (80GB 显存) GPU。
  - 训练加速：使用 DeepSpeed Zero2 策略进行分布式训练，配合 gradient checkpointing 节省 GPU 显存。
  - 高效训练支持：Megablocks (Gale et al. 2022) 实现 block-sparse 矩阵乘法 kernel 处理不规则形状 expert 的批量计算；ES-MoE (Kim et al. 2024) 引入 expert-wise offloading 和动态 expert 放置策略（论文在 Related Work/Efficient Training 中引述）。

- 模型是什么。数据集和bench分别是什么。
  - 模型架构：基于 LLaMA Transformer decoder-only 架构。Dense-0.4B: 12 layers, FFN hidden=12288, 12 heads×64 dim。Dense-1B: 12 layers, FFN hidden=32768, 16 heads×64 dim。MoE/HMoE-0.4B: 12 layers, 8 experts/layer, 总 expert hidden=12288。MoE/HMoE-3B: 12 layers, 8 experts/layer, 总 expert hidden=32768。Attention 层规格与对应 Dense 一致。LLaMA2 tokenizer, vocab=32000。
  - HMoE expert 大小分布（主实验）：arithmetic 策略，相对大小 {9, 11, 13, 15, 17, 19, 21, 23}（归一化后 expert 实际 hidden dim 分别为 2304, 2816, 3328, 3840, 4352, 4864, 5376, 5888 for 3B model）。
  - 训练数据集：RedPajama（开源），包含 Common Crawl, C4, GitHub, Wikipedia, Books (the Pile), arXiv, StackExchange。
  - 评估 benchmark（6 个）：PIQA（物理常识）、hellaswag（句子补全常识推理）、BoolQ（是非问答）、ARC-Easy（科学推理）、winogrande（代词消歧）、SIQA（社交常识推理）。评估协议：基于相同训练 FLOPs 预算（7×10^19 和 2.6×10^20）而非训练 token 数，因为不同方法激活参数量不同。使用 lm-evaluation-harness (Gao et al. 2021) 框架。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文声明 "Codes will be released upon acceptance"，当前未公开代码。
  - HMoE 算法 pipeline 伪代码（基于论文公式和 LLaMA FFN 设计）：

```python
# === HMoE Decoder Layer 前向传播 ===

# 输入: x [B, S, h_input], 当前层 hidden states
# N: 专家数量
# h_ffn_list: 各专家 hidden dim 列表，如 [2304, 2816, 3328, 3840, 4352, 4864, 5376, 5888]

# 1. Router (gating)
P = softmax(x @ W_r)  # W_r: [h_input, N], P: [B, S, N]

# 2. Top-K 或 Top-P 路由选择
if routing == "topk":
    top_k_vals, top_k_idx = topk(P, k=2, dim=-1)  # [B, S, 2]
    top_k_vals = top_k_vals / top_k_vals.sum(dim=-1, keepdim=True)  # 归一化
    # 被选中专家的 gate 值 = 归一化后的 top_k_vals，未选中 = 0

elif routing == "topp":
    P_sorted, sort_idx = sort(P, descending=True, dim=-1)
    # 如果 P_sorted[0] > p: 选 1 个
    # 否则累加直到 cumsum >= p
    t = argmin_k(cumsum(P_sorted) >= p)  # t 个专家
    selected_idx = sort_idx[:, :, :t]
    gate_vals = P.gather(dim=-1, index=selected_idx)
    gate_vals = gate_vals / gate_vals.sum(dim=-1, keepdim=True)

# 3. Expert Computation (异构)
output = zeros([B, S, h_input])
for i in range(N):
    mask_i = (expert i is selected for this token)  # [B, S]
    if mask_i.sum() == 0: continue
    x_i = x[mask_i]  # [n_i, h_input]
    gate_i = gate_vals[mask_i][对应于 expert i 的 gate 值]  # [n_i]

    # LLaMA-style FFN with expert-specific hidden dim h_ffn,i
    # W_g,i: [h_input, h_ffn,i], W_p,i: [h_input, h_ffn,i], W_o,i: [h_ffn,i, h_input]
    h_i = SiLU(x_i @ W_g,i) * (x_i @ W_p,i)  # [n_i, h_ffn,i]
    expert_out_i = h_i @ W_o,i  # [n_i, h_input]
    output[mask_i] += gate_i.unsqueeze(-1) * expert_out_i

# 4. Auxiliary Losses
# P-Penalty Loss (替代传统 load balancing loss):
L_pp = N * sum_i(M_i * P_hat_i)
# 其中 M_i = (1/T) * sum_t(1{e_i activated for token t} * h_ffn,i)
# P_hat_i = (1/T) * sum_t(P_i,t)
# h_ffn,i 是 expert i 的 hidden dim，大 expert 贡献更大 penalty

# Router Entropy Loss (仅 Top-P):
L_entropy = N * sum_i(P_i * log(P_i))  # 防止激活过多 expert

# 最终训练 loss:
L_total = L_lm + α * L_pp + β * L_entropy  # α=0.1, β=3e-2 (Top-P only)
```

张量计算示意（expert i, h_ffn,i=2304 vs expert j, h_ffn,j=5888）:
- Small expert i: input [n_i, 4096] → W_g [4096, 2304] → gate [n_i, 2304]; W_p [4096, 2304] → up [n_i, 2304]; SiLU(gate) ⊙ up → hidden [n_i, 2304]; W_o [2304, 4096] → output [n_i, 4096]。参数量 ≈ 3 × 4096 × 2304
- Large expert j: input [n_j, 4096] → W_g [4096, 5888] → gate [n_j, 5888]; 参数量 ≈ 3 × 4096 × 5888

异构 expert 使用 Megablocks block-sparse kernel 进行批量计算，避免不同形状 expert 的 padding 开销。
