## Multi-Head Feed-Forward Network (MH-FFN)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multi-Head Feed-Forward Network (MH-FFN) 是将 multi-head attention 的思想直接应用于 FFN 的朴素设计（FlashMHF 论文定义）。核心操作：(1) 输入 X ∈ R^{L×d_model} 先经 W_in 线性投影再通过 split_H 沿 d_model 维度切分为 H 个 head query，每 head 维度 d_h = d_model/H；(2) 每 head h 独立执行 key-value 形式的 FFN：FFÑ(Q_h; K^h, U^h, V^h) = (SiLU(Q_h·K^{hT}) ⊙ (Q_h·U^{hT}))·V^h，其中 K^h, U^h, V^h ∈ R^{d_ff×d_h} 是每 head 的私有参数；(3) 所有 head 的输出 concat 后经 W_out 投影回 d_model。这个设计直接从 MHA 的 split/parallel/compute/concat 范式迁移而来，但遇到两个关键挑战：(1) Memory Pressure——H 个 head 各自 materialize 中间激活 ∈ R^{L×d_ff}，总内存 O((L·H + d_model)·d_ff)，随 H 线性增长；(2) Scaling Imbalance——模型 scale up 时 d_ff 增长（因模型总参数增长需求）但 d_h 固定（如 128，继承自 MHA 设计），d_ff/d_h ratio 从 128M 的 16 膨胀到 1.3B 的 45，远超过最优范围。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```python
# Naïve Multi-Head FFN (MH-FFN) forward pass:
# 参数: W_in ∈ R^{d_model×d_model}, W_out ∈ R^{d_model×d_model}
#       For h=1..H: K^h, U^h ∈ R^{d_ff×d_h}, V^h ∈ R^{d_ff×d_h}
# 输入: X ∈ R^{L×d_model}

def naive_mh_ffn(X, W_in, W_out, per_head_params):
    H = len(per_head_params)  # number of heads
    d_h = d_model // H         # per-head dimension
    
    # Step 1: Project and split into heads
    Q = split_H(X @ W_in)     # [L, d_model] → [L, H, d_h]
    
    # Step 2: Per-head independent FFN computation
    S = []
    for h in range(H):
        K_h, U_h, V_h = per_head_params[h]  # each: [d_ff, d_h]
        Q_h = Q[:, h, :]                     # [L, d_h]
        
        # head-wise SwiGLU-style key-value FFN:
        gate = SiLU(Q_h @ K_h.T)   # [L, d_h] × [d_h, d_ff] → [L, d_ff]
        up   = Q_h @ U_h.T         # [L, d_h] × [d_h, d_ff] → [L, d_ff]
        out_h = (gate * up) @ V_h  # [L, d_ff] × [d_ff, d_h] → [L, d_h]
        
        S.append(out_h)            # H × [L, d_h] → 总激活 H·L·d_ff
    
    # Step 3: Concatenate heads and output projection
    O = concat_H(S) @ W_out        # [L, d_model] × [d_model, d_model]
    return O

# 问题演示（370M scale: d_model=1024, H=8, d_h=128, d_ff=2752, L=4096）:
# 每 head 激活: [4096, 2752] ≈ 22.5MB (bf16)
# 总 head 激活: H × 22.5 = 180MB (bf16) —— 相比标准 SwiGLU 的 22.5MB 为 8×
# d_ff/d_h = 2752/128 = 21.5 → 显著超过 optimal range ~8/3
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MH-FFN 的实用化受限于两个核心问题：(1) 内存消耗随 H 线性膨胀，即使在小模型 (128M) 下激活内存也是标准 FFN 的 H 倍，成为训练和推理的 bottleneck；(2) FLOPs 相同但 scaling imbalance 导致性能退化——FlashMHF 实验证实 MH-FFN 在 128M 优于 baseline 但在 370M 已失效。已有近似工作：(1) MH-MoE (Wu et al., 2024) 探索了多 FFN head + MoE sparse routing，但所有 head 共享 expert parameters，在 dense 模式下比 FlashMHF 需要 H 倍 FLOPs（公平对比不可行），且 memory 同样随 H 线性增长；(2) Tokenformer 将 FFN 替换为 Token-Parameter Attention，使用 learnable token 而非 fixed weight matrix 作为 key-value store，可视为一种隐式的 multi-head FFN 实现。

涉及论文标题：
- Flash Multi-Head Feed-Forward Network
- Multi-Head Mixture-of-Experts (Wu et al., 2024)
