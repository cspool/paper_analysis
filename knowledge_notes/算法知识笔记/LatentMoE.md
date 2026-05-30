## LatentMoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

LatentMoE 是 NVIDIA 提出的一种新 MoE 架构，核心思想是将 routed experts 的输入从原始 hidden dimension d 压缩到低维 latent space ℓ，利用节省的 memory bandwidth 和 communication 成本来扩展 expert 数量 N 和 top-K，在 iso-inference-cost 下提升模型精度。这是首次从 hardware-software co-design 角度系统性重新思考 MoE 架构设计的工作。

LatentMoE 包含两个变体：(1) ℓ-MoE_eff：压缩 hidden dim (d→ℓ)、扩展 N→αN、K 不变，在保持精度的同时降低 inference cost；(2) ℓ-MoE_acc（推荐）：同时扩展 N→αN 和 K→αK，在 iso-inference-cost 下提升精度。α = d/ℓ 为压缩比，论文通过 sweep 验证 α=4 为 safe compression ratio。

LatentMoE 被 NVIDIA Nemotron-3 Super/Ultra 模型采用，已通过 TensorRT-LLM v1.2.0+ 部署支持。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LatentMoE ℓ-MoE_acc 的算法 pipeline：

```
# LatentMoE ℓ-MoE_acc MoE Layer Forward
# 配置: d=4096, ℓ=1024, α=4, N=128 → N'=512, K=6 → K'=24, m=2688

Input: x ∈ R^{B×S×d}

# 1. Router（原始空间 d）
gate_logits = x @ W_r'.T              # W_r' ∈ R^{N'×d}
gate_probs = softmax(gate_logits)      # [B,S,N']
topk_vals, topk_ids = topk(gate_probs, K'=24)

# 2. Shared Down-Projection（所有专家共享）
z = W_↓ @ x                            # W_↓ ∈ R^{ℓ×d}, z ∈ R^{ℓ×B×S}

# 3. All-to-All Dispatch（在 latent space ℓ 中）
# 通信量 per token: ℓ (1024), 共 K'×ℓ = 24×1024 = 24576 bytes
# vs Standard MoE: K×d = 6×4096 = 24576 bytes (相同)

# 4. Expert FFN（在 latent space ℓ 中，per expert）
for each expert e (among 512 routed experts):
    z_e = tokens routed to e            # [n_e, ℓ]
    # W_gate ∈ R^{m×ℓ}, W_FC1 ∈ R^{m×ℓ}, W_FC2 ∈ R^{ℓ×m}
    h_gate = activation(z_e @ W_gate.T)  # [n_e, m]
    h_up   = z_e @ W_FC1.T              # [n_e, m]
    h      = h_gate * h_up              # [n_e, m] (SwiGLU/Squared-ReLU)
    e_out  = h @ W_FC2.T                # [n_e, ℓ]  FC2 down-projection

# 5. All-to-All Combine + Up-Projection
z_combined = aggregate across experts    # [B×S, ℓ]
routed_out = W_↑ @ z_combined            # W_↑ ∈ R^{d×ℓ}, [d, B×S]

# 6. Shared Experts（原始空间 d，S=2）
shared_out = Σ E_j(x; d) for j=1..S

# 7. Final output
output = routed_out + shared_out
```

关键对比：Standard MoE 中每个 expert 的权重总大小为 3×d×m（FC1 gate + FC1 up + FC2 down），LatentMoE 中每个 expert 的权重总大小为 3×ℓ×m，减少 α=4× per expert。但由于 K'=αK=24 vs K=6，总激活参数 K'×3×ℓ×m = K×3×d×m（在 ℓ-MoE_acc 中相同）。

Expert 组合空间：C(512,24) >> C(128,6)，指数级增长 token 级别的 expert combination diversity。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LatentMoE 通过以下方式实现：
- 共享的 W_↓ 和 W_↑ 对所有 routed experts 复用，仅增加 modest 的计算开销（~9% at trillion scale）
- Expert 权重在 latent space 中，每个 expert 的 weight loading memory BW 降低 α×
- All-to-All 通信在 latent space 中进行，per-token message size 降低 α×（但 K'=αK 补偿总通信量）
- TensorRT-LLM v1.2.0+ 原生支持 LatentMoE 部署
- 训练使用 DeepSeek-v2-lite 超参和架构，配合 aux-loss-free load balancing
- 可通过分离 CUDA streams for routed/shared experts 和专门的小 inner-dimension GEMM kernels（CUTLASS）进一步优化

涉及论文标题：
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts
