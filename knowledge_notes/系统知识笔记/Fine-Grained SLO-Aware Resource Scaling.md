## Fine-Grained SLO-Aware Resource Scaling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fine-Grained SLO-Aware Resource Scaling 是 JANUS 提出的解耦式 MoE 推理资源自动缩放机制。与 monolithic 系统的 coarse-grained scaling（最小缩放单位 = 完整模型副本）不同，JANUS 将 scaling 定义为一个二维优化问题：联合选择 attention 实例数 n_a 和 MoE 实例数 n_e，在满足 token-level TPOT SLO 的前提下最小化总 GPU 数 n_a + n_e。

核心公式：
$$
\min_{n_a, n_e, B^*} \quad n_a + n_e \quad \text{s.t.} \quad \text{TPOT}(B^*, n_a, n_e, S_{\text{ctx}}) \leq \text{SLO}
$$

其中 TPOT 通过 layer-wise latency model (Eq. 1) 建模，包括 attention latency (Roofline model)、MoE latency (β·a_max + c_e)、communication latency。Steady-state batch size B* 通过 Little's Law 求解: B* = λ · TPOT(B*)。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Scaling 决策流程（Algorithm 2, 15min 间隔）：

```
1. 收集近期 activation statistics (per-layer expert activation trace)
2. 构建 Monte Carlo â_max lookup table:
   For each candidate (n_e, B):
     从 trace 采样 B tokens → 应用 AEBS 策略 → 记录 â_max^(ℓ)(n_e, B)
3. Enumerate (n_a, n_e) search space {1..n_max} × {⌈E/C⌉..n_max}:
   a. 通过 binary search 求解 B* = λ·TPOT(B*, n_a, n_e, S_ctx)
      residual f(B) = B - λ·TPOT(B)
      边界: f(1) ≥ 0 → B*=1 (太轻载), f(B_max) < 0 → discard (无法满足)
   b. 查 â_max lookup table 计算 T_moe
   c. 检查 SLO (TPOT ≤ target) + memory feasibility
   d. 记录 min(n_a + n_e) 配置
4. Apply incrementally: 添加/移除 attention & MoE instances
5. 更新 expert placement (Algorithm 3: activation-aware co-activation minimization)
```

配置示例 (DeepSeek-V2, different SLO/workload):
- 轻载 (B=64): 1A6E (7 GPUs), ~99 tok/s/GPU
- 中载 (B=256, relaxed SLO): 2A6E (8 GPUs), ~240 tok/s/GPU
- 重载 (B=512, tight SLO): 4A6E (10 GPUs)

对比 Coarse-Grained Scaling:
- SGLang monolithic: 仅能 snap 到 8/16/32/64 GPU 整数副本
- MegaScale-Infer: 限制为 balance attention/MoE 执行时间的配置子集
- xDeepServe: 以 4 GPU 为单位 scale

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 依赖一次性 offline profiling 获取 hardware-dependent coefficients (α, β, c_a, c_kv, c_e)
- Monte Carlo lookup table 周期性重建 (15min) 以跟踪 workload changes
- 搜索空间 bounded by cluster size (n_max ≤ total GPUs)
- 24h production trace 评估: 节省 GPU-hour 39% vs SGLang, 16% vs MegaScale-Infer

涉及论文标题：
- JANUS: Disaggregating Attention and Experts for Scalable MoE Inference
