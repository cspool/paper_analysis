## Load Balance Loss (Auxiliary Loss for MoE / MoE 负载均衡辅助损失)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Load Balance Loss 是 MoE 训练中防止 expert 负载不均衡的辅助损失函数。MoE router 倾向反复选择少数 expert → 其他 expert 欠训练、热门 expert 成为计算瓶颈。MixLoRA 采用 Switch Transformers (Fedus et al. 2022) 公式：L_aux = a·N·Σ F_i·P_i，F_i=batch 中被路由到 expert i 的 token 比例，P_i=router 分配给 expert i 的平均概率，a=scaling coefficient（推荐 1e-3）。N 倍乘使 loss 在 expert 数量变化时恒定。总 loss: L = L_CE + L_aux。

从算法pipeline角度拆解术语：
```
for i in {1..N}:
    F_i = (1/T)·Σ_{x∈B} 𝟙{argmax_k R(x)_k = i}    // token 分配比例
    P_i = (1/T)·Σ_{x∈B} R(x)_i                       // router 概率均值
L_aux = a · N · Σ_{i=1}^{N} F_i · P_i
// a=1e-3 最优; N=8 时 L_aux ≈ 8e-3 · Σ F_i·P_i
```
Ablation：a=1e-3 时 MixLoRA 最佳 accuracy；a=0 (禁用) 降 ~2.5%；a=1e-1 (过大) 降 ~1.5%。验证效果：启用后 expert load std dev 低至 0.0223 (MixLoRA) / 0.0328 (MixDoRA)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 训练 loop 中额外 loss 项与 CE loss 直接相加。需获取 router 完整概率分布（不仅是 top-K 结果）。
- 几乎所有 MoE 训练必备（GShard → Mixtral → DeepSeek-V3）。
- MixLoRA 发现 MixDoRA 对 a 更不敏感：禁用 load balance loss 时 MixDoRA 仅降 ~1% vs MixLoRA 降 ~2.5%。

涉及论文标题：
- MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts
- Mixture of Diverse Size Experts
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

**局部路由一致性与负载均衡的 trade-off** (来自 "Not All Models Suit Expert Offloading", ICLR 2026)：论文揭示了 Load Balance 与 Local Routing Consistency 之间的重要区分——(1) Local Load Balance（单个 query 内 expert 激活的均匀程度）：与局部路由一致性存在强 trade-off，高一致性模型路由更集中（expert activation SD 更大），TOY 模型 NoLB（无 load balance loss）SRP 最高 56.42 但 LB SD=13.21 极高，OverLB（loss coefficient=0.1）SRP 最低 36.42 但 LB SD=1.79 极低。(2) Global Load Balance（跨不同 query 的整体 expert 利用率）：可与高局部路由一致性共存——Qwen3 (SRP 54.14, LB SD 3.19) 和 GRIN-MoE (SRP 50.39, LB SD 3.89) 同时具有高 SRP 和适中的全局负载均衡。机制：domain-specialized experts 在匹配其专长领域的上下文中持续激活（高局部一致性），在不同领域被不同 expert 集处理（全局均衡）。论文建议：若 target 场景涉及 expert offloading（如边缘设备部署），可适度牺牲局部负载均衡以换取更高的局部路由一致性。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Diverse Size Experts（MoDSE）是 Mixture of Diverse Size Experts 论文提出的 MoE FFN 层新结构。与传统 MoE 中所有 N 个 expert 拥有完全相同的 hidden dimension h 不同，MoDSE 在同一 MoE layer 内设置不同 hidden dimension $\hat{h}_i$ 的 expert：大专家（$\hat{h}_i > h$）具有更强预测能力，处理高难度 token 预测；小专家（$\hat{h}_i < h$）计算量更小，处理低难度 token 预测。Experts 按对分组 $(i_k^1, i_k^2)$，每对满足 $\hat{h}_{i_k^1} + \hat{h}_{i_k^2} = 2h$，保证总参数量与 baseline（所有 expert 尺寸相同）一致。设计动机：预训练语料中 token 预测难度差异巨大——同一短语内 token 极易预测，跨领域推理 token 极难预测——same-size expert 无法区分对待不同难度 token，diverse-size expert 让不同能力 expert 各司其职。

从算法pipeline角度拆解术语：
```
# MoDSE FFN Layer with Diverse Size Experts (N=8, K=2)
# Expert pairs with diverse hidden dims:
#   pair_0: E_{4.5}(h=6912), E_{0.5}(h=768)  -> 6912+768=7680=2×3840
#   pair_1: E_{4.0}(h=6144), E_{1.0}(h=1536) -> 6144+1536=7680
#   pair_2: E_{3.0}(h=4608), E_{2.0}(h=3072) -> 4608+3072=7680
#   pair_3: E_{2.5}(h=3840), E_{2.5}(h=3840) -> 3840+3840=7680
# baseline h=3840, dim=1536

# Input: x [B, S, dim]
# Step 1: Standard gating (same as Switch Transformer)
logits = x @ W_g                          # [B, S, N]
noise = RMSNorm(Softplus(x @ W_n))        # [B, S, N]
H = logits + noise
probs = Softmax(KeepTopK(H, k=2))         # [B, S, N]

# Step 2: Diverse-size expert computation
# Each expert E_i has different hidden dim h_i
#   E_i: w1_i [dim, h_i] -> SiLU -> w2_i [h_i, dim]
#   Parameter count for expert i: 2 * dim * h_i
#   Total params: 2 * dim * Σ_i h_i = 2 * dim * N * h (same as baseline)

output = zeros([B, S, dim])
for each expert i in {0..N-1}:
    mask_i = (expert i in top-2 for each token)
    if mask_i.any():
        tokens_i = x[mask_i]              # [n_i, dim]
        h_i = tokens_i @ w1_i             # [n_i, h_i] -- h_i varies!
        a_i = SiLU(h_i)                   # [n_i, h_i]
        out_i = a_i @ w2_i                # [n_i, dim]
        output[mask_i] += probs[mask_i, i] * out_i

# Step 3: Load balance loss (Switch Transformer style)
f_i = fraction of tokens routed to expert i
P_i = mean router probability for expert i
L_aux = α * N * Σ_i f_i * P_i
```

术语一般如何实现？如何使用？
- 实现：在 PyTorch 中，每个 expert 用不同尺寸的 `nn.Linear(dim, h_i)` 和 `nn.Linear(h_i, dim)`，forward 时根据 router 输出的 expert index 分发 token 到对应尺寸的 expert。需使用诸如 `torch.index_select` 或 scatter/gather 操作处理不同 expert 的不同 batch size。
- 配置：实验中使用 4 对 expert，尺寸比例分别为 (4.5,0.5), (4.0,1.0), (3.0,2.0), (2.5,2.5) 相对于 input dim。更大模型维持相同比例但绝对值更大（如 700M×8 中最大 expert h=9216）。
- 负载均衡：由于 expert 尺寸不同，天然导致计算负载不均衡→需配合 Expert-Pair Allocation 策略将每对 expert 放置在同一 GPU。
- 论文验证：700M×8 MoDSE 在 9 个 benchmark 上全面超越 same-size baseline（如 MMLU 29.9 vs 26.5, SIQA 60.9 vs 42.9），训练 loss 曲线更低，且推理耗时几乎与 baseline 相同（MMLU 3min27s vs 3min26s）。
- 局限性：实验仅在小规模 MoE（300M×8, 700M×8）上验证，大规模 MoE 的可扩展性未知。训练数据为非开源中英双语 100B tokens。

涉及论文标题：
- Mixture of Diverse Size Experts

---
