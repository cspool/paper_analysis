## Adaptive Batching 与 Lane Partitioning（自适应批处理与 lane 划分）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Adaptive batching 是 FlashTFHE 让"batch 大小"从设计时固定变为运行时/编译时可调的机制：时间域架构的 ciphertext 并行本质是软件控制的 batch 大小，编译器按程序实际 ciphertext-level 并行度调整 batch（并行丰富时 batch 增至 48 摊薄 launch 开销、最大化 BSK 复用；并行稀缺时收缩 batch 消除空闲轮次，并把回收的 key-switching lane 带宽重新分配给外部乘积流水线）。Lane partitioning 是其硬件支撑：LPU 的 8 个 lane 可独立寻址、独立时钟门控，且可按 worst-case 参数集满配 8-lane 保证 key-switching 不成为瓶颈，编译时用 lane masking 禁用不用的 lane。
- 动机（论文 profiled）：真实 multi-bit workload 中大量 batch 并行度有限——Decision Tree/KNN 的串行决策路径使某些阶段只有 8–46 个并行 ciphertext，静态（满 batch）batching 让 BRU 流水线空转、利用率跌破 50%。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 两种触发情形（Figure 12）：Case 1——某 batch 几乎全是 bootstrapping、无跨 batch 可交错的 key-switching（依赖关系禁止重叠），此时缩小 batch 提高 BRU 利用率、降低单 batch 延迟；Case 2——key-switching 实际只需少于 8 条 lane（多数 workload 如此），lane masking 关闭闲置 lane、缩小 batch、把省下的带宽重定向给 PBS 流水线。效果：KNN 平均利用率 32.7%→56.2%、端到端延迟降低最多 41.7%；Figure 18 显示 DNN/XGBoost 类并行丰富 workload 静态与自适应几乎无差（利用率近 100%），低并行度 workload 获益最大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：编译器在展开 MLIR FHELinAlg 后做依赖分析，识别可并行 ciphertext 数与可交错 KS，生成 batch 大小与 lane mask 配置；硬件端 LPU lane 独立时钟门控、BRU 按 batch 收尾。使用：面向"并行度不均匀"的 TFHE 程序（决策树、KNN、if-else 分支程序）；对 DNN/批量 LUT 类高并行度程序收益有限。论文消融（Figure 20）显示 adaptive batching 相对静态 baseline 改善 delay/EDP/EDAP。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
