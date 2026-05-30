## A2E / E2A Communication

术语解释
A2E (Attention-to-Expert) 和 E2A (Expert-to-Attention) 是 DEP 中两个 GPU group 之间的双向通信阶段：A2E 将 token 从 AG 路由到 EG，E2A 将 expert 输出从 EG 返回 AG。

术语是什么？
在 DEP 的每个 MoE layer 处理中，A2E 负责将 AG 处理后的 token hidden states 根据 gating/routing 结果发送到持有对应 expert 的 EG 设备。通信量 z = E/eg × me × M 字节（me 为每个 expert 处理的 token 数，M 为 hidden size）。E2A 是反向通信，将 expert 的 FFN 输出从 EG 收集回 AG。由于 DEP 的双向通信拓扑（如 PCIe 或 NVLink 支持 full-duplex），A2E 和 E2A 的时间通常相等，即 t_a2e(me) = t_e2a(me)。

从系统架构角度拆解：
A2E/E2A 通信是 DEP 的性能瓶颈之一：每个 MoE layer 需要一次 A2E 和一次 E2A，无优化时大量时间消耗在 GPU 等待通信完成上。FinDEP 论文给出定量数据：Naive DEP 在 DeepSeek-V2 S=4096 下非重叠通信时间为 905.49ms（每 iteration）。优化方向：(1) 与计算重叠——PP-Pipe 在 micro-batch 间重叠，FinDEP 进一步在 fine-grained token 段间重叠；(2) 减小通信量——通过量化（如 4-bit/8-bit 通信）减少 z。

术语一般如何实现？如何使用？
A2E/E2A 使用 NCCL 的 P2P send/recv 操作实现：(1) AG 端对每个 EG rank 执行 ncclSend（A2E），传输 shape=[me, M] 的 token tensor；(2) EG 端执行 ncclRecv（A2E）接收；(3) EG 计算完成后执行 ncclSend（E2A）；(4) AG 端执行 ncclRecv（E2A）接收。通信时间建模为线性函数 t_c(z) = α_c + β_c·z，其中 α_c 为网络启动延迟，β_c 为带宽倒数（如 β_a2e ≈ 9.61×10^{-7} 对应 eg=7,ag=1 配置）。

涉及论文标题：
- Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism

---
