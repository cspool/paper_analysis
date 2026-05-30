## Communication-Computation Overlap in MoE

术语解释
MoE 通信-计算重叠是在分布式 MoE 执行中将 expert 间的 all-to-all 通信与 expert FFN 的 GEMM 计算在时间上交叠执行的技术。由于 MoE 通信占端到端执行时间可达 47%（H800 Megatron-LM），有效重叠是 MoE 系统优化的核心方向。

术语是什么？
现有重叠方案按粒度分为三级：
1. **Coarse-Grained (Pipeline Degree = 2)**：FasterMoE 将 expert 计算拆分为 2 个 chunk 交替执行。初始和最后的通信阶段无计算可重叠，产生 bubble。仅 hide 29.2% 通信。
2. **Medium-Grained (Kernel-Level Scheduling)**：Tutel 通过自适应并行 + 2D 分层 all-to-all + 多 chunk pipeline。hide 68.6% 通信，但 expert 多时 CPU scheduling overhead 增大。
3. **Fine-Grained (Token/Tile Level)**：Comet 通过 shared tensor dependency resolving + fused kernel thread block specialization，将重叠粒度下沉到 token/tile 级。hide 86.5% 通信，消除 CPU scheduling overhead。

从系统架构角度拆解术语：

以 Comet 的一个 MoE layer 端到端重叠为例：
1. **Layer0 (Communication→Computation Pipeline)**：通信 TB（NVSHMEM 逐 token pull）与计算 TB（GroupGEMM tile-local 优先）在同一 fused kernel 内交织执行。Local token tile 零等待启动，remote token tile 在数据到达后立即消费。
2. **Layer1 (Computation→Communication Pipeline)**：计算 TB 按列 block 执行 column-wise GroupGEMM → 首列 block 完成后通信 TB 立即启动 top-K reduce + NVSHMEM write → 后续列 GEMM 与 reduce/通信重叠。

传统方案（FasterMoE/Tutel）在 kernel 边界做重叠（通信和计算分属不同 CUDA streams），Comet 在 kernel 内部做重叠（通信 TB 和计算 TB 共存于一 kernel，由 SM hardware scheduler 管理）。kernel 内重叠优势：(1) 消除多次 kernel launch 的 CPU↔GPU 往返，(2) 实现 token/列级精准重叠，(3) 避免 CUDA stream 同步开销。

术语一般如何实现？如何使用？
- 通信库：NCCL（coarse/medium-grained 方案）、NVSHMEM（fine-grained intra-kernel 方案）
- 实现模式：multi-CUDA-stream pipeline（FasterMoE/Tutel）、single fused kernel（Comet）
- 关键挑战：通信-计算粒度不匹配（token vs tile）、动态负载（M/EP/TP 变化）、I/O 对计算 kernel 效率的干扰
- 硬件要求：高带宽互联（NVLink/NVSwitch），fine-grained 方案对通信延迟更敏感
- 正交优化：expert packing（Lina，对齐通信和计算时间）、communication compression（ScheMoE，减少通信量）

涉及论文标题：
- Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts
- Accelerating Distributed MoE Training and Inference with Lina
