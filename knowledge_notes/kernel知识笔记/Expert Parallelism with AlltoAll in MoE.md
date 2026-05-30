## Expert Parallelism with AlltoAll in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Parallelism (EP) 是分布式 MoE 的核心并行策略：将 MoE layer 中多个 expert FFN 分布到不同 GPU 上，每 GPU 持有部分 experts。每个 token 通过 gating 动态路由到 top-k expert，需通过 AlltoAll collective 将 token 送到目标 expert 所在 GPU。AlltoAll 分两轮: (1) Dispatch AlltoAll——将 token 从原始序列顺序重排为按目标 expert 排列；(2) Combine AlltoAll——将 expert 输出从按 expert 排列恢复为原始序列顺序。AlltoAll 为同步 barrier-based collective——所有 GPU 必须同时参与，最慢 GPU 卡住全体 (straggler)。expert capacity C = (S×k×cf)/E 限制每 expert 最多处理 C 个 token，超出的 token 丢弃（capacity factor cf 控制，通常 1.0-1.25）。通信可占 MoE layer 运行时间的 68%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

传统 NCCL AlltoAll: `ncclGroupStart()` + 多对 `ncclSend`/`ncclRecv` + `ncclGroupEnd()`。各 GPU 向所有其他 GPU 发送不同数量 token。同步 barrier: 若 GPU 3 的 gate kernel 启动慢 (straggler) → 所有 GPU idle 等待。FlashMoE 替代方案: 消除 AlltoAll，Processor 通过 NVSHMEM put 直接写远端 symmetric tensor L (push-model): `nvshmem_putmem(remote_L[target][DISPATCH][INCOMING][tile], local_tile, size)` → signal flag → Subscriber 异步消费。Combine 同理: GEMM1 输出直接 put 到远端 combine buffer。无 collective barrier，每 GPU 独立 push。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

AlltoAll 常见实现: NCCL (GPU collective lib, NVLink/NVSwitch), DeepEP (混合 NCCL+NVSHMEM 优化), Tutel (自适应算法选择), FlashMoE (用 device-initiated one-sided DMA 替代)。Expert capacity: C = (S×k×cf)/E。Auxiliary loss 鼓励 balanced routing 以最小化 overflow。

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
