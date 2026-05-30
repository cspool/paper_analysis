## Top-K Router (Top-K 路由器 for MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-K Router 是 MoE 为每 token 从 N 个 expert 中选择 K 个最适合 expert 的路由机制。Router 为线性层 W_r ∈ R^{d×N}，将 token hidden state h ∈ R^d 映射为 N 个 expert logits → Softmax → Top-K 选 K 个 expert。MixLoRA 采用 Top-2 Router（K=2），从 8 个 LoRA-based expert 中为每 token 选择 2 个 expert，按路由概率加权输出。

从算法pipeline角度拆解术语：
```
输入: h_i [1, d], W_r [d, N]  (N=8, K=2)
logits = W_r · h_i^T                           // [1, N]
probs = Softmax(logits)                         // [1, N]
r' = KeepTop-2(probs)                           // re-normalize top-2 positions
output = Σ_{k: r'_k>0} r'_k · E_k(h_i)         // 加权求和
```
Router 在每个 MoE layer 有独立参数 W_r^ℓ。MixLoRA 中 router 仅应用于 FFN MoE（非 attention），区别于 MOELoRA/MOLA 将 router 应用于 attention+FFN。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：`nn.Linear(d_model, num_experts)` + Softmax + `torch.topk()`。
- 训练时需 load balance loss 防止 collapse。MixLoRA 验证 Top-2 在单任务和多任务下均有效。

涉及论文标题：
- MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts
- Mixture of Diverse Size Experts (MoDSE 的 Top-K Router 逻辑与标准 MoE 相同，但 router 将 token 分配给不同尺寸的 expert，困难 token 倾向选择大专家)
- MoLA: MoE LoRA with Layer-wise Expert Allocation (per-layer independent Top-2 router; each layer j has its own W_r^j ∈ R^{d_q × N_j} with layer-specific N_j experts; router output S_i^{jt}(x) = TopK(Softmax(W_r^{jt}x), K)_i / Σ TopK(...)_i)

---
