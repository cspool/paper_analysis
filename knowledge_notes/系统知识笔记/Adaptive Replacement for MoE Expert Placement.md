## Adaptive Replacement for MoE Expert Placement

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Replacement (AR) 是 FineMoE 中与 per-micro-batch token scheduling 互补的长期 placement 调整机制。Token scheduling 处理 transient fine-grained 不均衡，AR 处理 long-term coarse-grained 不均衡。流程：(1) 训练初期用 symmetric placement（Cayley graphs）；(2) 后台监控 expert load；(3) 每 ~50 iterations 用 moving averages 预测未来 load → Equation 3 评估当前 placement 性能；(4) 性能下降则生成新 asymmetric placement（greedy replica count + Monte Carlo sampling），reinitialize 模型状态。

从系统架构角度拆解术语：
AR 与 expert scheduling 的关键区别：
- 设计目标：Expert scheduling 中 placement 调整是唯一手段 → FineMoE 中 token scheduling 为主、AR 为辅。
- 算法基础：Expert scheduling 假设 replica loads 均匀 → FineMoE replica loads 由 LP 决定（可不均匀），需 graph theory（Equation 3）指导。
- Equation 3: m = max_{G_max} (1/|G_max| · Σ_{e:EDP^e ⊆ G_max} load_e)，最优 placement 使 max induced subgraph density 最小。

术语一般如何实现？如何使用？
- Placement Manager（GPU 0 Python）负责整个 AR 流程。
- 迁移开销 ~数百 ms，通过调间隔频率控制（50 iter → <1% overhead）。
- 适用高 skewness（s>1）或 training 初期 load 剧烈波动场景。

涉及论文标题：
- FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

---
