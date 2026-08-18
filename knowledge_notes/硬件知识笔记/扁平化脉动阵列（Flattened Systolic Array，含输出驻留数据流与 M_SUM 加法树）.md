## 扁平化脉动阵列（Flattened Systolic Array，含输出驻留数据流与 M_SUM 加法树）

术语解释
把传统 N×N 方形脉动阵列改为"一行小方阵（sub-arr）串接"的宽扁形状，以匹配长上下文 LLM 推理的 fat GEMM 与 per-head attention GEMM。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PLENA 的扁平阵列配置为 (BLEN,MLEN)：每个 sub-arr 做 (BLEN,BLEN) 输出 tile 的 (BLEN,MLEN)×(MLEN,BLEN) GEMM，BLEN≪MLEN（BLEN 常设成≈batch、MLEN 对齐隐藏维方向）。同乘法器数下，8×512 扁平阵列的可达 FLOPs 显著高于 64×64 方阵（图 2：144 GB HBM、512 GB/s 配置）。数据流为输出驻留（output-stationary）：部分和驻留 PE，操作数沿大归约维 K 流式推进，阵列全流水消除相邻 GEMM tile 间的气泡。每个 sub-arr 只积累结果的一部分，完整 (BLEN,BLEN) 输出需跨 sub-arr 求和——由结果加法树完成（M_SUM 指令触发，沿大 K 维一次 GEMM 只需一次 cross-array 求和）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
数据通路（图 6）：每周期从 Matrix SRAM（顶部）与 Vector SRAM（左侧）各取一个 MLEN 宽输入 → 缓冲重排 → 按 MLEN/BLEN 切成 BLEN 宽子向量分送对应 sub-arr 的顶/左边界 → 每 PE 做 MX 格式乘 + INT 累加，结果向下/右邻 PE 传递 → K 维归约完成后 M_SUM 加法树汇总同行 sub-arr 的部分和 → 转目标激活精度写回 Vector SRAM。FlashAttention 场景：阵列被切分为多个小 flattened core，每 core 执行 (BLEN,HLEN)×(HLEN,BLEN) 的 per-head GEMM，MLEN//HLEN 个头并行（head 预加载）。元素与 E8M0 scale 分流流送，PE 原生消费 MX 输入。
面积/能效（7 nm 合成，Table XI/图 14）：4×1024 阵列 comp area 0.237 mm²、TOPs/mm² 34.49；agentic 负载 A.A FLOPs/mm² 12.81，对比 64×64 方阵的 MicroscopiQ 1.08 / OliVe 0.40 / FIGNA 6.71；扁平阵列面积功耗略高（图 14），但利用率提升使 FFN/attention 有效能耗显著更低（OSWorld-L 负载）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：sub-arr 为 PE 网格，PE 做 MX 乘 + INT 累加；BLEN/MLEN 由 DSE 与 workload 共同决定（约束 MLEN mod BLEN=0、MLEN≥HLEN≥BLEN）；跨阵列部分和用加法树一次归约避免逐 tile 气泡。使用：面向 fat GEMM 与 per-head attention 的 LLM 加速器阵列组织——把长 K 做流水维、短 M 做 BLEN，取代方形阵列的对称 tile 假设；FFN 与 FlashAttention 复用同一阵列（形态切换）。Scale-Sim 支持矩形/扁平阵列仿真（纵横比与利用率非线性相关），SARA 探索可重构形状，PLENA 是按 autoregressive Transformer 特征（prefill/decode 两阶段、fat GEMM 与头级 GEMM 共存）workload 驱动的固定扁平设计；对比 SystolicAttention（把 FlashAttention 固化进阵列），PLENA 用头级分解 + 预取重叠的灵活方案。

涉及论文标题：
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
