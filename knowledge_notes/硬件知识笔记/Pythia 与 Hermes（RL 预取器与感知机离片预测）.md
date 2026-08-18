## Pythia 与 Hermes（RL 预取器与感知机离片预测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Pythia（Bera 等，MICRO 2021，CMU SAFARI）是基于在线强化学习的可定制硬件预取框架：用多种程序上下文（PC/偏移等）与系统级反馈（命中/带宽）学习"何时、预取多少、多长距离"的策略，而非人工启发式。Hermes（Bera 等，MICRO 2022 Best Paper）用感知机（POPET，Perceptron-based Off-chip Predictor，约 84% 准确率 vs 先前 47%）预测哪些 load 会离片（miss 整个缓存层次），提前从主存取数据以消除片上缓存延迟的关键路径。二者是 Moirai 选择的"SOTA 泛化式预取器"组合 baseline（L2C RL 预取 + L1D 感知机离片预测），合计 29.5KB 存储——比 Moirai 大 97%。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 组合运转（Moirai 的对比场景）：L1D 侧 Hermes 感知机对每个请求预测是否离片 → 预测离片则并发从主存取（同时走 cache 层次）→ 命中率提升；L2C 侧 Pythia RL agent 观测 decimated miss 流、按状态（上下文+系统反馈）选预取动作。问题：它们只能看到被 L1D 过滤的流（缺完整时序上下文）、注入距离远（预取太晚）、合计 29.5KB 存储（L1D 预算内不可落地）。Moirai 多核 7.8% vs Pythia+Hermes 8.3%——Moirai 用 loss-based 节流达到类似多核抗争用效果但零带宽 agent 成本。
- 补充：Pythia 帮助友好负载 ~16% 平均 speedup，但对预取不利负载可降 ~11.6%；Hermes 在不利负载 +1.4%——二者朴素组合反而 -11.2%，需要智能协调（Athena/StaticBest），凸显跨层协同的复杂性，而 Moirai 在 L1D 源头自包含解决。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Pythia 用强化学习（状态=程序上下文，动作=预取配置，奖励=性能/带宽反馈），ChampSim 实现并开源（github.com/CMU-SAFARI/Pythia）；Hermes 用感知机（轻量线性学习器）预测 off-chip，开源（github.com/CMU-SAFARI/Hermes），依赖 Bloom Filter 库（Moirai artifact 的 libbf/ 即为其提供）。使用：作为"泛化式/智能预取"的 SOTA 代表评估 Moirai（29.5KB vs 780 Bytes）；Pythia+Hermes 是多核设计（Moirai 用 20 个 4 核混合对比）。局限：存储/延迟使其与 L1D 预算冲突，只能驻留 L2/LLC。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework
