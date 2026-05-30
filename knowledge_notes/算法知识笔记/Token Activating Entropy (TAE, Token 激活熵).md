## Token Activating Entropy (TAE, Token 激活熵)

术语解释
Token Activating Entropy (TAE) 是 BuddyMoE 的第一个 safety gate 指标，量化 token 对 expert 替代的敏感度：TAE_ℓ(x) = -Σ_{i∈S} p̃_ℓ(i|x)·log(p̃_ℓ(i|x)) / log(k) ∈ [0,1]。低 TAE=peaked routing（敏感，禁止替换），高 TAE=diffuse routing（容忍，允许替换）。当 TAE ≤ τ 时禁止替换。

术语是什么？
TAE 复用 router 的 top-k softmax 输出，在 renormalized top-k 概率上计算归一化信息熵。归一化因子 log(k) 使值域为 [0,1]。三个实现细节：(1) renormalize over top-k only 避免尾部 artifact；(2) optional temperature smoothing p̃(x;T)=softmax(z(x)/T) with T∈[0.8,1.2] 稳定跨层 TAE；(3) percentile calibration——τ 按 per-layer TAE 分布的 p-th percentile (p∈[10,20]) 选择，跨模型/领域鲁棒。可选与 probability margin (p̃_max - p̃_2nd ≥ γ) 组合增强安全性。

从算法pipeline角度拆解术语：
```
topk_probs = Router(x).softmax()[topk_indices]
p_tilde = topk_probs / sum(topk_probs)
TAE = -sum(p_tilde * log(p_tilde)) / log(k)
if TAE <= tau: forbid replacement for this token
```

术语一般如何实现？如何使用？
- 计算开销 O(k) 可忽略（复用已有 router 输出）
- τ 选择影响 accuracy-throughput trade-off：较高 τ 更保守，较低 τ 更激进
- 论文 τ=0.95 获最佳平衡（c=0.75: Acc=0.695, t/s=36.75, +7.4% vs original）
- TAE 是三个 safety gate 中最先执行的（overhead 最低）

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference

---
