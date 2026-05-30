## Elastic Vertical Scaling with Zero-Downtime for MoE Inference (弹性零停机垂直缩放)

术语是什么？

Elastic Vertical Scaling with Zero-Downtime 是 ElasticMoE 提出的 MoE LLM 推理实例增量式垂直缩放框架，支持在不中断服务的前提下以 2 个 NPU 为粒度增减推理实例资源。区别于 Horizontal Scaling（启动完整推理实例副本，粒度粗至 32-320 NPU，延迟数十秒至分钟级）和 Naive Vertical Scaling（需要销毁旧实例再冷启动新实例，引入 downtime），ElasticMoE 通过三个机制实现零停机垂直缩放：(1) **HBM 管理与推理执行解耦**：HMM（HBM Management Module）以持久守护进程独立管理模型权重和 KV cache，权重仅从磁盘加载一次，推理实例通过 zero-copy IPC 引用句柄附加权重；(2) **Scale-while-serve 模型**：旧推理实例持续服务 in-flight 请求，新实例在后台准备（pre-initialized 或通过 HCCL P2P 传输权重），准备完成后 Coordinator 无缝切换流量；(3) **固定 TP 仅调 DP/EP**：保持 Tensor Parallelism 度不变使共享 NPU 上的 attention 权重和 KV cache 布局不变，可直接 zero-copy 复用，避免每步缩放的权重全量重分配。ElasticMoE 推理实例基于 vLLM，Coordinator/IMM/HMM 三级模块通过 ZMQ/UNIX domain socket IPC 通信。

从系统架构角度拆解术语：

Scale-up 操作流程（4→6 NPU, DP2→DP3, EP4→EP6）：
```
Step 1: SLO-Aware Load Estimator 检测 SLO < 90% → 触发 scale-up
Step 2: HMM 分析 current vs target config → 生成最小代价再分配计划
Step 3: Attention 权重: NPU 0-3 zero-copy 复用; NPU 4-5 HCCL P2P 传输
Step 4: KV Cache: NPU 0-3 zero-copy 复用; NPU 4-5 新分配
Step 5: Expert 权重: 全局 remap → p2p-copy 迁移 → vpage-remap 更新映射
Step 6: IMM pre-initialized 实例 zero-copy attach → ready
Step 7: Coordinator 停止旧实例新请求 → 等待 in-flight → 流量切换
```
Scale-down 是反向简化操作：仅需要 EP 重配置（expert 迁移到保留 NPU），attention 和 KV cache 在保留 NPU 上 zero-copy 复用。

术语一般如何实现？如何使用？

ElasticMoE 在 Huawei CloudMatrix384 (Ascend 910C NPU) 上实现：Coordinator (Python/ZMQ, TCP API), HMM (Python/Ray 控制面 + C++/CANN 数据面 + PyBind11), IMM (vLLM + ascend-vLLM)。支持 DeepSeekV2 Lite、Qwen3-30B-A3B、DeepSeek V3 等 MoE 模型。论文也给出了 CUDA 生态扩展方案：CUDA IPC (cudaIpcGetMemHandle/cudaIpcOpenMemHandle) 替代 Ascend IPC，CUDA virtual memory API (cuMemAddressReserve/cuMemMap) 替代 CANN vpage-remap，vLLM CUDA backend 替代 ascend-vLLM。关键性能：scale-up latency ~2.43s (DP3→DP4, DeepSeek V2 Lite)，为最佳 baseline 的 ~0.11×，0 downtime，peak memory 仅比 Cold Restart 高 2-3%。

涉及论文标题：
- ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models
