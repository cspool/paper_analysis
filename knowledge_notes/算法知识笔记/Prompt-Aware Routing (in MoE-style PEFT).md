## Prompt-Aware Routing (in MoE-style PEFT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Prompt-Aware Routing 是 MiLoRA 提出的 LoRA expert 路由机制。与传统 MoE-style LoRA（如 MOELoRA、LoRAMoE）的 token-wise routing 不同，prompt-aware routing 在输入 prompt 首次通过 LLM backbone 时仅计算一次路由决策（在生成第一个新 token 之前），后续所有 auto-regressive token 生成步骤均复用该决策。Router 输入为 prompt hidden states H^l ∈ R^{n_p×d}，经 Self-Attention Pooler → Rational Activation → Linear Router + Softmax → Top-k，每层选出 1 个 LoRA expert（从 Q/K/V/O/G/U/D 7 个模块中）。Router 调用次数从 token-wise 的 O(L×T×7) 降至 O(L×7)，生成阶段无 router 开销。

从算法pipeline角度拆解术语：
```
阶段一: Prompt编码（仅一次，在第一个new token前）
输入: H^l ∈ R^{n_p × d}  (layer l 的输入hidden states)

# Pooling
W_sa ∈ R^{d×1}
U = H^l @ W_sa → [n_p × 1]
A = Softmax(U, dim=0)
h^l = A^T @ H^l → [1 × d]

# Rational Activation（可学习，每层独立）
g^l = Ra(h^l)  # m=6, n=5, 初始化为GeLU近似

# Router
logits = g^l @ W_r^l   # [1×d]@[d×7]→[1×7]
probs = Softmax(logits)
expert_idx = TopK(probs, k=3)  # 选top-1 expert

阶段二: Token生成（复用阶段一路由决策）
for token in 1..T:
    if module == expert_idx:
        output = x@W_m + x@W_m^A@W_m^B + b_m  # LoRA修正
    else:
        output = x@W_m + b_m  # 原始backbone
```
效率：L=32, T=256时，router调用从57344降至224。实测 beam=1 tps 43.7 (vs MOELoRA 35.9, +21.7%)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 HuggingFace Transformers forward 中增加 router 计算。Routing decisions 缓存为 layer-level 属性（如 `self.activated_expert_idx`），供后续 token 复用。
- Pooler: 默认 self-attention pooling（最优），备选 last-token/average/max pooling。
- Load balancing: L_lb = N_mod · Σ f_i·p̂_i，λ_lb=1e-2。
- 场景: multi-tenant LLM serving（每 tenant 独立 LoRA adapter），消除 per-token routing 开销。

涉及论文标题：
- MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning

---
