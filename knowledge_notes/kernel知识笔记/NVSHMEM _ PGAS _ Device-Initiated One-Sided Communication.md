## NVSHMEM / PGAS / Device-Initiated One-Sided Communication

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

NVSHMEM (NVIDIA Shared Memory) 是 NVIDIA 基于 OpenSHMEM 标准实现的 GPU 集群并行编程接口。它提供 Partitioned Global Address Space (PGAS) 编程模型——所有 GPU 的显存被抽象为一个全局地址空间，每个 GPU 可以直接通过 put/get/atomic 操作访问远端 GPU 显存，无需远端 CPU 参与。核心特征是 device-initiated communication：GPU kernel 内部（CUDA thread/warp/block）可直接调用 `nvshmem_put()`、`nvshmem_get()` 等 API 发起跨 GPU 数据传输，替代传统的 CPU 发起 NCCL collective（如 AlltoAll）。NVSHMEM 3.x 支持 NVLink（intra-node）、InfiniBand/RoCE RDMA（inter-node）、UCX transport，支持 Hopper TMA、threadgroup communication、collective kernel launch。One-sided semantics 指通信仅需发起方参与，target GPU 无需显式调用 recv——数据直接写入远端地址空间。Weak ordering 下需显式 fence/quiet 保证 ordering，signal/wait 原语用于 point-to-point 同步。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FlashMoE 使用 NVSHMEM 替代 NCCL AlltoAll 执行 token dispatch。Processor thread block 在 kernel 内直接调用 `nvshmem_putmem(dst_ptr, src_ptr, tile_size)` 将 token tile 从本地 GPU HBM 写入远端 GPU 的 Symmetric Tensor Layout L。完成后通过 `nvshmem_uint64_p(flag_ptr, 1, target_gpu)` 写 signal flag 通知远端 Subscriber。远端 Subscriber warp poll flag → memory fence → 从 L 读取 tile → 解码为 task。关键：传统 NCCL AlltoAll 是同步 pull-model collective（所有 GPU barrier 等待），NVSHMEM 使 FlashMoE 实现异步 push-model——每个 GPU 独立 push token 到目标，无需等待目标 GPU 也完成同一轮 push，消除 straggler delay。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

开源: https://github.com/NVIDIA/nvshmem (v3.4.5)。使用: (1) `nvshmem_init()` 初始化 + `nvshmem_team_split_strided()` 建立 GPU team；(2) `nvshmem_malloc()` 在各 GPU 分配等量 symmetric heap → 建立全局地址空间；(3) Kernel 内通过 `nvshmemx_*_block()` (block-level) 或 `nvshmemx_*_warp()` (warp-level) API 发起 put/get/atomic → `nvshmem_fence()`/`nvshmem_quiet()` ordering。Intra-node NVLink 带宽可达 395 GB/s (1GB put, vs NCCL P2P 336 GB/s)。FlashMoE 的 symmetric tensor layout L 利用 PGAS 全局地址空间: L 第一维索引 p = source GPU rank，保证不同 source GPU 写入不同目标位置 (write-write conflict-free)。

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
