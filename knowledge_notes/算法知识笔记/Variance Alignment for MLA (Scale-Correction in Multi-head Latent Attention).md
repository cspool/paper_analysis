## Variance Alignment for MLA (Scale-Correction in Multi-head Latent Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Variance Alignment for MLA 是 LongCat-Flash 针对 MLA 低秩分解中的方差不对齐问题提出的修复。问题根源：MLA 的 query 压缩维度 $d_q$、KV 压缩维度 $d_{kv}$ 和模型维度 $d_{\text{model}}$ 通常在 scaling 时独立变化。在初始化时，query 分量 $q_t^C$ 和 $q_t^R$ 的方差分别 $\propto d_q$，key 分量 $k_t^C$ 的方差 $\propto d_{kv}$，而 rotary key 分量 $k_t^R$ 的方差 $\propto d_{\text{model}}$。维度间的方差不匹配导致注意力分数在初始化时不稳定（某些维度的分量主导 attention score），small scale 下表现良好的 MLA 配置在 scaling up 时性能退化。

解决方案：在低秩路径分量上应用 scale-correction 因子 $\alpha_q = \sqrt{\frac{d_{\text{model}}}{d_q}}$ 和 $\alpha_{kv} = \sqrt{\frac{d_{\text{model}}}{d_{kv}}}$，将它们缩放后的最终方差对齐到 $d_{\text{model}}$ 参考尺度。

修正后的 MLA 公式：$$c_t^Q = \alpha_q W^{DQ} h_t, \quad c_t^{KV} = \alpha_{kv} W^{DKV} h_t$$ 使得初始化时 $q_t^C, q_t^R, k_t^C, k_t^R$ 的方差均匀贡献到 attention score，确保模型 scaling 时的稳定性和可预测性。LongCat-Flash 实验（Figure 5a）显示在 1B activated MoE 上 scale-correction 带来更低 validation loss。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LongCat-Flash 的 MLA 完整 forward pass：

```
输入: h_t [batch, seq_len, d_model=6144]

# Hyper-params: d_q=1536, d_kv=512, n_h=64, per-head dim=128 on the up-projected side

# Stage 1: Latent Compression (with scale-correction)
alpha_q = sqrt(d_model / d_q)      # sqrt(6144/1536) = 2.0
alpha_kv = sqrt(d_model / d_kv)    # sqrt(6144/512) ≈ 3.464

c_Q  = alpha_q  * W_DQ  @ h_t    # [batch, seq, d_q=1536], query latent
c_KV = alpha_kv * W_DKV @ h_t    # [batch, seq, d_kv=512], KV latent

# Stage 2: Up-Projection
q_C = W_UQ @ c_Q                 # [batch, seq, n_h * dim_per_head=128] for compressed part
q_R = W_QR @ c_Q                 # [batch, seq, n_h * d_rope] for RoPE part
k_C = W_UK @ c_KV                # [batch, seq, n_h * dim_per_head=128] (non-RoPE part)
v   = W_UV @ c_KV                # [batch, seq, n_h * dim_per_head=128]
k_R = W_KR @ h_t                 # [batch, seq, d_rope] shared across heads

# Stage 3: RoPE
q_R_rope = RoPE(q_R)
k_R_rope = RoPE(k_R)  # broadcast to all heads

# Stage 4: Concatenation
q = concat([q_C, q_R_rope], dim=-1)
k = concat([k_C, k_R_rope.expand(-1, n_h, -1)], dim=-1)

# Stage 5: Attention
o = Attention(q, k, v)

# Stage 6: Output Projection
u = W_O @ concat([o_1, ..., o_{n_h}], dim=-1)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Scale-correction 的实现极简：在模型初始化/forward 时将低秩投影权重乘以上述 scaling factor（或直接在 forward 中对 c_Q, c_KV 乘 factor）。α 值仅依赖于架构选择的 d_q, d_kv, d_model 三个超参数，无需额外训练或调参。LongCat-Flash 配置：d_model=6144, d_q=1536, d_kv=512 → α_q=2.0, α_kv≈3.464。

与 DeepSeek-V2/V3 的 MLA 对比：DeepSeek 原始 MLA 未使用 scale-correction，可能在特定 d_q/d_kv 比例下表现良好但缩放时性能退化。Scale-correction 提供了保证任意维度配置下注意力机制都能稳定运行的通用解决方案。

涉及论文标题：
- LongCat-Flash Technical Report
