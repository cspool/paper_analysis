## Dynamic Expert Pruning in MoE-LLMs (Token-Aware)

术语是什么？
Dynamic Expert Pruning（动态专家剪枝）是在 MoE-LLM 推理时对每个 token 动态决定实际激活的 expert 数量的技术。与静态剪枝（永久移除某些 expert）不同，动态剪枝的决策是 per-token 的。MC-MoE 的 ODP（Online Dynamic Pruning）含两个关键组件：(1) Weight-guided pruning：当 Top-2 中次要 expert 的 routing score 远小于主要 expert（w₁/w₀ < μ，μ 为 calibration 数据中位数），跳过该次要 expert；(2) Token protection：基于 token importance Iⱼ = ‖tⱼ‖₁ · mean_attention_score 保护 top 2% 重要 token 的所有 expert 不被剪枝，防止 attention decay 级联效应。平均减少约 15% 激活参数，准确率损失 < 1%。

从算法pipeline角度拆解术语：
```
// 每个 MoE layer 推理时动态执行
I_j = ||t_j||_1 · (Σ_{i≥j} A_{j,i}) / (L - j)  // token importance
is_protected = (I_j in top 2%)

{w_0, w_1} = Top-2{G(t)}
if is_protected:
    y = w_0·E_0(t) + w_1·E_1(t)     // 保护: 完整 top-2
elif w_1/w_0 < μ:                     // μ = 校准集 w₁/w₀ 中位数
    y = w_0·E_0(t)                    // 剪枝: 降为 top-1
else:
    y = w_0·E_0(t) + w_1·E_1(t)     // 正常 top-2
```

术语一般如何实现？如何使用？
- 实现依赖：(a) 校准数据确定 pruning threshold μ；(b) 推理时在线计算 token importance（计算开销 O(n²+mn) FLOPs，远小于 expert 推理的 O(n·m·m₁) FLOPs）
- 适用场景：MoE-LLM 延迟敏感推理（如 real-time chatbot）
- 与静态剪枝互补：静态剪枝减少存储，动态剪枝减少计算
- 创新点：token-aware protection 机制，仅保护 2% token 即消除 attention decay

涉及论文标题：
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

---
