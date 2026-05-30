## Pattern Compute Unit (PCU / 模式计算单元)

术语是什么？
Pattern Compute Unit (PCU) 是 SN40L RDU 中的可重构计算单元，每个 RDU 含 1040 个 PCU。PCU 可配置为两种模式：(1) Output stationary systolic array — 用于 GEMM 等矩阵乘法，输入从左到右、从上到下通过 broadcast buffer 流入 systolic array，累加结果从左到右经 tail unit 排出；(2) Pipelined SIMD core — 用于 element-wise 操作和 reduction，每个 SIMD stage 支持 FP32/BF16/INT32 的算术、逻辑和位运算。PCU 的 datapath 包含 header（消费输入数据流并驱动 body）、body（可配置 systolic/SIMD）、tail（transcendental functions、随机数生成、stochastic rounding、格式转换）。PCU 内含 cross-lane reduction network（蓝三角）和 lane-wise reduction。Counters 跟踪 loop iteration，产生 done events 用于控制流。

从硬件架构角度拆解：
PCU 在 single kernel 中的协作模式：
- Data Parallelism: 输入/输出被 partition 为多个独立 data stream，分别由不同 PCU 处理
- Tensor Parallelism: fork 为 data parallel streams → 各 stream 计算 → join 结果
- Pipeline Parallelism: 多个 PCU chained 为 pipeline stages，中间结果通过 PMU stage buffers 传递
- 对于 Figure 4 的 GEMM→Mul→Transpose→GEMM：Gemm0 映射到多个 PCU 配为 systolic → Mul 在另一组 PCU 以 SIMD 执行 → Gemm1 在更多 PCU 以 systolic 执行

术语一般如何实现？如何使用？
PCU 由编译器自动配置：编译器根据 operator 类型选择 PCU 模式（systolic vs SIMD），分配 PCU 数量以匹配各 stage 的算力需求（Gemm0/Gemm1 占用更多 PCU），配置 data/tensor/pipeline parallelism 模式。用户编写标准 PyTorch 模型代码即可，无需手动编程 PCU。

涉及论文标题：
- SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts
