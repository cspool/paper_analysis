## NVSHMEM（GPU 对称内存 / PGAS one-sided 通信库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVSHMEM 是 NVIDIA 基于 OpenSHMEM 标准的 GPU 集群通信库，提供 Partitioned Global Address Space（PGAS）编程模型：所有 GPU 显存被抽象为统一全局地址空间（对称内存 symmetric memory——各 GPU 分配相同大小、相同虚拟地址的内存），GPU 通过 one-sided put/get/atomic 直接读写远端 GPU 对称内存，无需远端显式参与，也无需 CPU 中转；配合 GPUDirect（IBGDA）让 GPU 直接访问 RDMA 网络。相比 NCCL collective，NVSHMEM 支持 kernel 内 fine-grained、device-initiated 通信（如 nvshmem_putmem_nbi + 信号量信号）。Tetris（ISCA'26）用 NVSHMEM 实现 ring attention 的 KV 传输，降低 ring 通信开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Tetris ring attention 的 KV 传输（rank i → rank i+1）
nvshmem_malloc(&sym_kv, size)          # 初始化对称内存（每 rank 同址）
nvshmem_putmem_nbi(peer_sym_kv, local_kv, size, peer=i+1)   # one-sided put（非阻塞）
nvshmem_fence(); nvshmem_quiet()       # 完成语义
# 接收侧 nvshmem_signal_wait_until 轮询信号，无需 rank i+1 显式 recv
```
Annotations: putmem_nbi 为非阻塞 bulk put；对称地址使收发双方无需地址交换；信号/等待原语提供 completion 通知；one-sided 消除 two-sided MPI/NCCL 的 CPU 参与与同步配对。
在 Tetris 中：ring attention 每步把 K/V chunk 直接 put 到下一实例的对称 buffer 与下一轮 attention 计算重叠；相比 NCCL P2P，kernel 内 fine-grained 传输更易与 Flash Attention zigzag 计算融合/重叠。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NVIDIA NVSHMEM 库（https://developer.nvidia.com/nvshmem，OpenSHMEM 规范）；常见用法——初始化（nvshmem_init/nvshmem_malloc）→ kernel 内 put/get + 信号同步 → 完成（nvshmem_quiet/fence）。生态：DeepEP（DeepSeek MoE 通信库）构建于 NVSHMEM 之上、FlashMoE 用其做单 kernel 分布式 MoE、Comet/Mirage 等 fused kernel 用其做 kernel 内通信。使用场景：需要 device-initiated / fine-grained / 与计算融合的跨 GPU 通信（MoE dispatch-combine、ring attention、replicated expert 预取）。Web 证据：NVSHMEM 官方文档（docs.nvidia.com/nvshmem）与各 MoE 论文实现说明。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
