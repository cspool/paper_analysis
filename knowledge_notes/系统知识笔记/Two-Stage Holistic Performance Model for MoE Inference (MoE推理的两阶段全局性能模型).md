## Two-Stage Holistic Performance Model for MoE Inference (MoE推理的两阶段全局性能模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Two-Stage Holistic Performance Model 是 MoE-Lens 的核心理论贡献。Stage 1（§5.1-5.4）从 fundamental system components 推导理论上界：CPU memory capacity → Equation 2（饱和 GPU 的 token 数）→ PME（Equation 3）→ $T_{max}$（Equation 4）→ CPU resource requirements（Equations 5-6）→ prefill-decode overlapping benefit（Equation 7）；Stage 2（§5.5）引入真实执行因素：bounded batch size K、paged KV cache（block size b）、prefill/decode overlapping 调度 → Equations 8-14 预测真实 throughput（94% accuracy）。Stage 2 当 K → ∞ 且 b → 1 时收敛到 Stage 1。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Model 在 MoE-Lens 系统设计中的三个作用：(1) Pre-deployment: Stage 1 判断给定 HW + model 的 theoretical bound；(2) Configuration: Stage 2 为给定 workload 预测 throughput，选择最优 KV cache size；(3) Online guidance: Stage 2 输出的 q（per-iteration prefill sequences）和 $n_{real}$（GPU saturation threshold）直接输入 Resource-Aware Scheduler 作为调度约束。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Python analytical model（非 simulation），输入 HW/model/workload 参数，输出 predicted throughput 和 GPU utilization。
- 相比 MoE-Lightning HRM 的改进：HRM 仅建模 arithmetic intensity vs IO bandwidth（忽略 CPU memory capacity、workload characteristics、prefill/decode overlapping、paged KV cache fragmentation），导致 MoE-Lightning GPU utilization 仅 16.5%；MoE-Lens two-stage model 覆盖这些维度 → GPU utilization ~90%。

涉及论文标题：
- MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints
