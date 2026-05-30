## Prefix Linear Attention (PLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PLA（Prefix Linear Attention）是JRT-RNN论文提出的encoder-decoder线性注意力变体。将输入序列分为前缀encoder区域（前M个token，非因果处理）和decoder区域（后N-M个token，causal处理）。Encoder使用独立投影(k_e, v_e)，decoder使用独立投影(k_d, v_d)，两套投影不共享（区别于Prefix-LM的单套投影）。核心公式：y_i = φ(q_i)(Σ_{j=1}^{i}k_d[j]^T v_d[j] + Σ_{j=1}^{M}k_e[j]^T v_e[j]) / φ(q_i)(Σ_{j=1}^{i}k_d[j] + Σ_{j=1}^{M}k_e[j])。Decode阶段O(1) per token（与标准linear attention相同）——prefix的贡献在prefill时预计算为固定KV-state s_M = Σ_{j=1}^{M}(k_e[j]^T v_e[j] + k_d[j]^T v_d[j])。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Input: u ∈ R^{N×d}, prefix length M, feature map φ (e.g. Taylor 2nd-order)

// Encoder projections (non-causal, tokens 1..M)
k_e = φ(W_{ke} · u_{1:M}),  v_e = W_{ve} · u_{1:M}

// Decoder projections (causal, tokens 1..N)
k_d = φ(W_{kd} · u),  v_d = W_{vd} · u,  q_d = φ(W_{qd} · u)

// Prefill: compute encoder KV-state (non-causal sum)
KV_enc = Σ_{j=1}^{M} k_e[j]^T v_e[j]       // ∈ R^{d×d̃}
K_enc  = Σ_{j=1}^{M} k_e[j]                 // ∈ R^{d̃}

// Decoder prefill: cumsum from encoder-init state
KV_dec[i] = KV_enc + Σ_{j=1}^{i} k_d[j]^T v_d[j]
K_dec[i]  = K_enc  + Σ_{j=1}^{i} k_d[j]
y_i = (q_d[i] · KV_dec[i]) / (q_d[i] · K_dec[i])

// Decoding (i > M, O(1)):
s_i = s_{i-1} + k_d[i]^T v_d[i],  z_i = z_{i-1} + k_d[i]
y_i = (q_d[i] · s_i) / (q_d[i] · z_i)
```
训练时追加MLM loss: L = (w1·L_NTP + w2·L_MLM)/(w1+w2)，encoder区域随机mask比例P的token。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于Based架构实现（交替gated conv+sliding window+PLA层），feature map用2阶Taylor近似。JRT-RNN CUDA kernel扩展ThunderKittens Based kernel：先fnbased(k_e,v_e)计算encoder KV-state存入寄存器，再fnbased(q_d,k_d,v_d)从该状态续算decoder。Pre-fill比FA2快19.2×（N=32768, H100）。开源：https://github.com/HazyResearch/prefix-linear-attention。适用于需要recall-intensive ICL但保持O(1)推理内存的循环LM场景。

涉及论文标题：
- Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

---
