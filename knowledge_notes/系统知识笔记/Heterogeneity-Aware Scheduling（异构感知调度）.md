## Heterogeneity-Aware Scheduling（异构感知调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Heterogeneity-Aware Scheduling 是把负载映射到最优/可用硬件代际的调度策略：计算密集相位（prefill、大模型）跑新 GPU，内存/带宽受限相位（decode、小模型）跑旧 GPU，从而在异构 fleet 中最大化性能-成本比、延缓刷新。本论文（Rearchitecting the Datacenter Lifecycle for AI）把它列为 operation 阶段 8 类软件优化之一（表 VIII：Heterogeneity-Aware Scheduling [55],[68]，"Map workloads to optimal/available hardware generation"，TCO 影响：Defer refresh costs），并给出 cross-stage 语义（A. Existing Cross-Stage Optimizations）：operate → IT provisioning——异构感知调度把旧 GPU 重新派给适合其能力的负载，把"硬件升级"变成"负载再分布"而非"提前退役"。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在本论文的驱动逻辑：论文实测不同模型-硬件组合的 workload 级效率差异（图 8-10，vLLM 测 TTFT/TBT 按 H200 归一化）——Llama3-70B 在 A100 上 goodput/Watt 比 H100 低约 3×10^10，而 1B/3B/8B 小模型 A100 每美元效率反而高于 H100（8–23%），V100 对小模型也能达到 H100 的 95%；sparse（Qwen3-235B-A22B）与 state-space（Mamba-2.8B）模型在旧 GPU 上退化更小（V100 上 Mamba 仅慢 3.6× vs Llama3 慢 7.7×）。异构感知调度据此把大模型 prefill 放新代、小模型与 decode 放旧代，使旧 GPU 持续产出价值 → 刷新策略可跳过 B100/B200、延长 H100/H200 寿命（图 13 平滑混合机队）。这是"workload 画像驱动调度 + 调度反馈驱动硬件采购"的闭环。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现：SchedTune（异构 GPU 集群 DL 调度）、HeterMoE（异构 GPU MoE 训练，attention 新卡/expert 旧卡）、Splitwise 相关异构相位部署；本论文在 TCO 模拟中把异构调度抽象为"负载-代际匹配矩阵"，与 roofline 性能模型、SLO goodput、蒙特卡洛刷新搜索联合求解，而非提供在线调度器实现。开源框架 AI Lifecycle Compass（https://github.com/Azure/AI-Lifecycle-Compass）的模拟器与策略层支持加载自定义异构调度假设并量化其对 TCO 的影响。

涉及论文标题：
- Rearchitecting the Datacenter Lifecycle for AI
