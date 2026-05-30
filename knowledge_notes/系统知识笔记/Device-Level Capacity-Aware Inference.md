## Device-Level Capacity-Aware Inference

术语解释
将 expert 容量约束从 per-expert 粒度（N_i ≤ γN̄）放宽到 per-device 粒度（ΣN_i ≤ n_l·γN̄），允许同设备内 expert 间容量共享，减少因单个 expert 超限导致的过度 token 丢弃。是 Capacity-Aware Inference 的进阶变体。

术语是什么？
Expert-Level 约束严格限制每个 expert 的 token 数，即使同设备其他 expert 有大量剩余容量，单个超限 expert 的 token 仍被丢弃。Device-Level 约束只限制设备总 token 数——只要 ΣN_i ≤ n_l·γN̄，即使某 expert 超限也不丢弃 token。在 EP 下每 GPU 托管多个 expert 时（如 OLMoE 8E/GPU），Device-Level 更灵活，丢弃率更低。

从系统架构角度拆解术语：
以 Qwen3-MoE (8E/GPU EP), γ=1.0 Device-Level vs γ=1.5 Expert-Level 对比：
```
Expert-Level (γ=1.5):  N_i ≤ 1.5N̄, ∀i
  expert_3 收到 1.8N̄ → 丢弃 0.3N̄ token (即使其他 7 expert 总容量充足)
  → 过度丢弃

Device-Level (γ=1.0):  ΣN_i ≤ 8·1.0N̄ = 8N̄
  expert_3 收到 1.8N̄, 总计 7.2N̄ < 8N̄ → 全部保留
  → 灵活容量共享
```
效果：Qwen3-MoE Device-Level γ=1.0 speedup 1.31× (vs Expert-Level γ=1.5 speedup 1.23×)，MoE 层 speedup 1.51× vs 1.40×，性能 Avg 74.8 vs 73.9（Table 3）。

术语一般如何实现？如何使用？
实现时将 per-expert topk 改为 device 级 aggregate：统计 device 总 token 数 → 若 ≤ n_l·γN̄ 不丢弃 → 若超过则在 device 内跨所有 expert 按 gating score 全局排序丢弃最低分 token。适用于 EP 下每 GPU 托管多个 expert 的场景。

涉及论文标题：
- Capacity-Aware Inference Mitigating the Straggler Effect in Mixture of Experts

---
