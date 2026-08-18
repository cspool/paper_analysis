## 核间融合与流水线（Inter-kernel Fusion & Pipelining，空间分区流水线）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是面向加速器多 kernel 串行执行的调度概念：把协议计算图按 cut 点切分成子图，调度器用简单 roofline 模型估计各 kernel 行为后贪心合并可并行子图，把 PE 阵列空间分区，让多个 kernel 同时片上执行成流水线，并按吞吐匹配调整各 kernel 的 PE 比例。目标：消除 kernel 间中间数据落片外（fusion）与 kernel 间空闲等待（pipelining），缓解访存瓶颈，提高 PE 利用率。GenZA 中属于静态调度器的输出（Section VI-F），一次性离线调度，成本在多次证明实例间摊销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 调度流程（Groth16 例子）：
```
1) 把 PIOP 计算图按 cut 点切分（三类 cut：数据转置 / 全局归约 / Fiat-Shamir 串行点）
2) 对每个子图，贪心合并 kernel 为空间流水线（若 roofline 估计提升性能）
3) 调各 kernel 的 PE 比例匹配吞吐；子图间数据直接前递（fusion 消除中间写回）
例子：子-iNTT 尾部与后续子-coset-NTT 头部直接融合，中间数据留片上
```
- Annotations：效果（Table XI，对比简单 LRU 缓存）：Groth16 流量 247.9→237.5 GB（MSM 计算密集、收益小）；HyperPlonk 196.7→117.0 GB（1.7×）；Plonky2 1220.0→128.3 GB（fusion 单独 8.1×、pipelining 再 1.2×，共 9.5×），但 Plonky2 离线调度需 517.8 s（复杂计算图，一次性成本）。cut 点类型：数据转置（多项式阶段间/2D-NTT）、全局归约（MSM 标量点积、sumcheck 向量和）、Fiat-Shamir 哈希挑战（强制串行点）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：GenZA 静态调度器（分析成本模型+贪心合并+roofline 估计）产出 PE 分配与流水线方案，运行时每 PE 收模式配置（kernel 类型、bitwidth、场模、数据流模式）；tiling 相同的 producer/consumer 融合，树式 workload 分片、串行归约用 segmented-parallel。使用：访存密集、多串行 kernel 的协议（Plonky2）收益最大；与 kernel 级映射优化（折叠 NTT、动态 MSM window）正交叠加。类比：GPU 库的 kernel fusion（cuDNN/cuBLAS 手调 kernel 模板），GenZA 把融合决策自动化到调度器。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
