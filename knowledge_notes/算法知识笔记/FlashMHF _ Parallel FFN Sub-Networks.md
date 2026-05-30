## FlashMHF / Parallel FFN Sub-Networks

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlashMHF（Flash Multi-Head Feed-Forward）是 MH-FFN 的实用化改进，其核心创新是 Parallel FFN Sub-Networks 设计——用于解决 naïve MH-FFN 的 Scaling Imbalance 问题。传统 MH-FFN 的 d_ff/d_h ratio 随模型 scale 膨胀（128M: 16, 370M: 21, 1.3B: 45），因为 d_ff 增长而 d_h 固定。FlashMHF 的解决方案：(1) 将每 head 的 d_ff 维计算分解为 E 个 parallel sub-network，每 sub-network 的 internal dimension d_e ≈ 8/3·d_h（维持 SwiGLU 的最优 expansion ratio），总 d_ff = E·d_e；(2) 引入 learned gating：每 head h 有 gating matrix W^h ∈ R^{d_h×E}，计算 per-token sub-network weights R^h = sigmoid(Q_h·W^h) / Σ sigmoid（soft normalization），然后用 R^h 加权聚合所有 sub-network 的输出；(3) 最终 concat 所有 head 输出并做 W_out 投影。这个设计本质上类似 dense MoE——每 token 的所有 E 个 "expert"（sub-network）都参与计算（无 sparse top-k routing），以微小 gating 开销换取平衡的 internal ratio 和丰富的 representational diversity。相比标准 SwiGLU（单路径 "greedy search"），FlashMHF 的 H×E 个 parallel pathway 可类比为 implicit thinking 的 "beam search"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```python
# FlashMHF Forward Pass (以 370M: d_model=1024, H=8, d_h=128, E=7, d_e≈342):
def flashmhf_forward(X):
    # 参数:
    # W_in ∈ R^{d_model × d_model}
    # For h=1..H, e=1..E: K_e^h, U_e^h, V_e^h ∈ R^{d_e × d_h}  (每个342×128)
    # For h=1..H:          W^h ∈ R^{d_h × E}                   (每头128×7)
    # W_out ∈ R^{d_model × d_model}
    
    # === Step 1: Head-wise split ===
    Q = split_H(X @ W_in)           # [L,1024]→[L,8,128]
    
    # === Step 2: Per-head gating + sub-network aggregation ===
    S = []  # 每head的输出
    for h in range(H):              # H=8
        Q_h = Q[:, h, :]            # [L, 128]
        
        # 2a: Gating — 学习每token对E个子网络的权重
        P = Q_h @ W[h]             # [L,128] × [128,7] → [L,7]   (logits)
        R = sigmoid(P) / (sigmoid(P).sum(dim=1, keepdim=True) + 1e-8)  # [L,7]
        # R[:,e] ∈ (0,1), Σ_e R[:,e] ≈ 1
        
        # 2b: Sub-network computation & weighted aggregation
        S_h = zeros([L, d_h])       # [L, 128]
        for e in range(E):          # E=7
            K_e, U_e, V_e = params_K[h][e], params_U[h][e], params_V[h][e]
            # 每个 [d_e, d_h] = [342, 128]
            
            # FFÑ sub-computation (SwiGLU-style key-value):
            gate = SiLU(Q_h @ K_e.T)       # [L,128]×[128,342]→[L,342]
            up   = Q_h @ U_e.T             # [L,128]×[128,342]→[L,342]
            out_e = (gate * up) @ V_e      # [L,342]×[342,128]→[L,128]
            
            S_h += R[:, e:e+1] * out_e     # gated aggregation
        
        S.append(S_h)                # [L, 128]
    
    # === Step 3: Concat & output ===
    O = concat_H(S) @ W_out         # [L,1024]×[1024,1024]→[L,1024]
    return O, Q, R

# 关键设计参数推导:
# 标准 SwiGLU: d_ff ≈ 8/3·d_model = 8/3·1024 ≈ 2731 (round to 2752)
# FlashMHF:   d_e ≈ 8/3·d_h = 8/3·128 ≈ 341 (round to 342 in multiples of 64)
#             E = floor(d_ff / d_e) = 2752/342 ≈ 8, 论文用 7
#             d_ff_actual = E × d_e = 7 × 342 = 2394 (需调整 layers 保持总参数 ≈ baseline)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashMHF 的实现：
1. **参数组织**：K/U/V 参数组织为 [H, E, d_e, d_h] 的 4D tensor 以利于 kernel 访问。Gating weights W^h ∈ R^{d_h×E} 为可训练参数，通过 sigmoid + normalization 得到 per-token per-sub-network 权重。相比 MoE 的 softmax-gated sparse routing，FlashMHF 的 sigmoid gate 避免了 top-k selection 的 load imbalance 和 token dropping 问题，且所有 sub-network 都参与计算（dense activation）。
2. **训练**：标准 PyTorch training loop，FlashMHF module 替换标准 SwiGLU FFN module。使用 AdamW optimizer，training hyperparameters 与 baseline 完全一致。128M/370M: 60B tokens (Pile), 1.3B: 100B tokens。单 GPU 训练（pretraining_tp=1）。
3. **推理优化**：SRAMFFN kernel（Triton/ThunderKittens 实现）用于高效 fused 计算——将 Step 2b 的 inner loop 和 Step 2a 全部融合为单个 I/O-aware kernel，避免中间 gate/up ∈ R^{L×d_e} 写入 HBM。
4. **配置灵活性**：d_h 可调（64/128/256），通过改变 H 和 E 适配不同模型 scale。380M 实验显示 d_h=128 为 sweet spot——d_h=64 每 head 容量不足（representational bottleneck），d_h=256 减少 head 数降低 diversity。

涉及论文标题：
- Flash Multi-Head Feed-Forward Network
