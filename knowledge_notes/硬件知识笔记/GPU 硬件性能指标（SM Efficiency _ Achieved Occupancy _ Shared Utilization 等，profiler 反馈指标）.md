## GPU 硬件性能指标（SM Efficiency / Achieved Occupancy / Shared Utilization 等，profiler 反馈指标）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QiMeng-Tensify（ISCA'26）在 Architecture-aware Prior Adaptation 中收集的一组硬件执行指标（Table V）：SM Efficiency（活跃 SM 周期占比）、Shared Utilization（共享内存使用率）、Achieved Occupancy（每 SM 实际 warp 数，归一化）、Instructions per Warp（每 warp 平均执行指令数）、Tensor Precision FU Utilization（Tensor Core 按精度的利用率）、Warp Execution Efficiency（每 warp 平均活跃线程/最大线程）。这些是 NVIDIA profiler（Nsight/NCU）能提供的 kernel 执行统计量，用于离线阶段把"程序结构 + 性能结果"关联成可解释的优化根因，从而蒸馏出自然语言启发式并训练 cost model。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在架构感知先验适配中的运转流程：搜索 GatedMLP 等代表子图 → 真机执行候选 kernel → profiler 采集上述指标 → 连同"输入 TensorIR、采用的调度策略、奖励 R"一起作为反馈 F 喂给 meta prompt → LLM 用这些指标归纳性能根因（例如"共享内存利用率低 + 指令多 → 应 compute_at 融合减少中间 buffer"、"寄存器逼近上限 → 抑制 vectorize/unroll 保 occupancy"）→ 更新 prompt 中的 learned principles。也就是说硬件指标是"架构特征"的可观测代理，使 LLM 先验从"通用编译常识"变成"针对当前 GPU 微架构的经验规则"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：GPU profiler 在 kernel 执行期间采样（SM 活跃周期、occupancy 计数、shared memory 用量、指令计数、tensor core 利用率、warp 级活跃线程），统计到 kernel 粒度；论文离线用 7 个代表子图（DoubleMatmul、Conv2d+Bias+Relu、Matmul、LSTM、Matmul+Relu、MLP、Softmax）收集 12,500 次"程序-指标-延迟"测量。使用方式：两类消费——(1) 喂 LLM 蒸馏硬件感知启发式（prompt 自进化）、(2) 训练 XGBoost cost model 作为在线预测器；指标分布还直接揭示架构差异（A100 上 memory-centric 规则采样多 vs H100 上 compute-centric 规则采样多）。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
