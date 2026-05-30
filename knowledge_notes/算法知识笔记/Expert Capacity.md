## Expert Capacity

术语解释
Expert Capacity 是限制每个专家每批次最大处理 token 数的阈值，防止 MoE 训练中某些专家被过度使用导致计算热点和内存溢出。由 GShard (2020) 引入。

术语是什么？
capacity = (tokens_per_batch / N_experts) × capacity_factor (CF, 典型值 1.0-2.0)。超过 capacity 的 tokens 被丢弃或随机路由至备选专家。BPR 按 gate score 高→低分配优先级。

OpenMoE 发现的关键现象：(1) "Drop-towards-the-End": 序列后部 tokens 更易被丢弃 (2) "Context-independent Specialization": 专家按 token ID 专业化 (3) "Early Routing Learning": 路由模式在预训练早期固定。

术语一般如何实现？如何使用？
- CF 越大→dropped tokens 越少→计算量越大
- Expert-Choice Gating 不需 capacity（天然均衡）
- 推理时通常不启用 capacity（batch 更小）

从算法pipeline角度拆解术语：
SYMI 重新定义了 Expert Capacity 在 adaptive replication 下的行为。在传统静态系统中：
capacity(e_i) = capacity_factor × tokens_per_batch / E = slot_capacity × r

在 SYMI 的 adaptive replication 下：
capacity_SYMI(e_i) = slot_capacity × r_i

其中 r_i 随 iteration 动态变化（r_i ∝ popularity_i）。当 replication 精确匹配 popularity 时，capacity_factor 变得无关——热门 expert 因更多 replica 而自动获得更大总 capacity，冷门 expert 减少 replica 但不影响其处理能力。这使得 SYMI 在所有 auxiliary loss coefficient 下均保持约 10% token drops（vs DeepSpeed 的 40%+）。

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)
- Accelerating MoE Model Inference with Expert Sharding
- Capacity-Aware Inference Mitigating the Straggler Effect in Mixture of Experts

**MoEShard 对 Capacity Factor 的批判**：
MoEShard 指出 CF 方法的根本缺陷：(1) 超限 token 被丢弃直接损害模型精度；(2) CF 增大虽减少丢 token 但导致显存问题（实验中将 DeepSpeed CF 固定为 min(|E|, 50)，再增大即 OOM）；(3) DeepSpeed 的 CF 限制使得当 expert 数超过 50 时开始丢 token，此时 MoEShard 的 dropless 优势更明显（256 expert 时 MoEShard 仍保持 2.39× 加速）。MoEShard 通过 expert tensor sharding 从根本上避免对 CF 的依赖——所有 token 全程保留、无需 capacity 限制。

**Capacity-Aware Token Drop：推理时的 Expert Capacity 应用**：
Capacity-Aware Inference 首次将 Expert Capacity 系统性地应用于**推理阶段**（而非仅训练阶段），解决 Expert Parallelism 下的 Straggler Effect。核心设计：

1. **Capacity 定义**：C = γN̄ = γ(tk/n)，其中 γ 为容量因子（典型值 1.0-2.0），t = batch_size × seq_len 为总 token 数，k 为 top-k 值，n 为 expert 总数。

2. **Score-based Dropping**：使用 softmax 后的 gating score 作为 token 重要性度量，对超载 expert 丢弃 score 最低的 token。论文验证 Score 优于 Order、Reverse Order、Random（Table 1: Score Avg 61.1 vs Random 53.1 at γ=1.0）。

3. **效率-精度权衡**：γ=1.5 时 OLMoE 获得 30% speedup 仅损失 0.9% 性能（64.0→63.1）；Mixtral-8×7B γ=1.5 时 Token Drop 获 1.87× 加速。

4. **Dropped Token 比例**：DT = Σ ReLU(N_i - γN̄) / Σ N_i。丢弃 12% token 可获 85% 加速（Mixtral）。

5. **与 Expert Parallelism 的交互**：每 GPU 托管 expert 越少（如 Mixtral 1-2E/GPU）效果越显著，因单个 straggler expert load 占比大。托管 8E/GPU 时加速减弱。

---
