## Near-NAND Computing（近 NAND 计算：die 控制器侧有限算力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Near-NAND computing 指在 NAND die 的控制器/外围放少量计算单元做部分计算，减少 SSD→NPU 的原始数据传输。代表：Cambricon-LLM（MICRO'24，arXiv:2409.15654）——NPU + 专用 flash chiplet 经 die-to-die 链路直连（绕过 UFS 4.0 带宽限制），flash 内带 on-die 计算单元与超轻量 on-die ECC 保护权重 outlier，70B LLM 3.44 tokens/s（较基线 22×+）；Lincoln（HPCA'25）——消费级设备上用 LPDDR 接口的 compute-enabled flash（3D hybrid bonding 提内部带宽 + 近 flash 计算 + 投机解码），prefill 13.23×、generation 254.1×。DIAMoND 语境：near-NAND 是 in-NAND 的前一代 baseline——每 die 仅 1.6 GOPS（Cambricon-LLM）/38 GOPS（Lincoln），而解码单 token 需 100~300 GOPS 线性运算、边缘吞吐目标 4~12 tokens/s，故 near-NAND 算力远不够，多数计算仍回 NPU，带宽瓶颈依旧。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
near-NAND 的执行流程（DIAMoND 的对比对象）：SSD 读页（tR~20µs，每 plane 每页一次）→ die 内计算单元做有限计算（partial 算子）→ 结果经通道送 NPU 完成主计算。限制链：读延迟 ~20µs 且单页/plane 串行 → 要达到足够并行度需海量 die（Cambricon-LLM 需 256+ die）→ 而 NAND 功率大头是 establish 功率（WL 电容预充），与 die 数成正比 → 总功耗高；DIAMoND 的对策是 in-NAND 一次激活平面内多页（多 OU 并行），die 数只需 16，总 establish 功率反而更低（单 die 功率更高但 die 数少 16×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：die 控制器内集成小型 MAC/累加逻辑或近 flash 计算层（Lincoln 用 hybrid bonding 实现 near-flash compute），配合 on-die ECC/容错。使用方式：边缘/消费级 LLM 推理的存储侧加速——把 memory-bound 的 decode 阶段计算尽量留在 flash 侧、把权重搬运留在封装内；DIAMoND 将其作为直接对比基线（速度 4.97~11.72×、能效 2.39~11.36×）。同族延伸：KVNAND（arXiv:2512.03608）把 KV cache 也放 flash（DRAM-free）。评估工具：SSDsim 类周期级模拟器。

涉及论文标题：
- DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture
