## JANUS: Disaggregating Attention and Experts for Scalable MoE Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - JANUS 包含两个 kernel 调度/运行时计算层面的实现：
    1. **Activated-Expert-Balanced Scheduling (AEBS) GPU Kernel（Section 3.4, Algorithm 1）**：实现为 GPU kernel，在每 MoE 层解码时执行。调度流程：Gating 产生 top-K 逻辑 expert IDs → Kernel 扫描路由结果收集激活 expert 集合 (GPU 线程并行处理 tokens) → 单副本 expert 分配到唯一持有实例 → 多副本 expert 贪心选负载最低实例 (load[g] = 当前层已分配给实例 g 的 unique expert 数) → 将每个 token 的路由结果从逻辑 EID 重写为物理 RID → Dispatch token。每个 MoE instance 独立执行相同 kernel（synchronization-free），通过确定性算法 + 相同输入保证一致性，消除跨 GPU 协调开销。
    2. **NVSHMEM-based One-Sided Communication Kernel（Section 3.3, Implementation）**：使用 NVSHMEM putmem_signal/signal_wait 原语实现 GPU-initiated one-sided RDMA——发送端 GPU kernel 直接写入接收端 GPU memory → signal 通知完成。元数据 (layer index, token count) 打包进 signal value 避免单独传输。NVSHMEM 参数调优包括 IBGDA transport、request-batching threshold、per-peer RC queue count。与 NCCL intra-node collectives 配合实现 adaptive two-phase communication。
  - 实验比较：
    - AEBS vs EPLB（DeepSeek EP Load Balancer）：a_max 对比 (Fig. 13)、MoE-layer latency (Fig. 14)、scheduling overhead (Fig. 15)
    - Two-phase (2PC) vs One-phase (1PC) communication：TPOT 和 throughput 消融 (Fig. 12)
    - EGate (MoE 侧 gating) vs AGate (Attention 侧 gating)：throughput 消融 (Fig. 12)
    - AEBS overhead: <20μs (small batch) ~ <90μs (batch=4096)

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA H100 80GB (每节点 8×, 最多 4 节点 32 GPU)
  - Intra-node 互联：NVLink 900 GB/s
  - Inter-node 互联：400 Gbps InfiniBand NIC per GPU + GPUDirect RDMA
  - 通信库：NVSHMEM (GPU-initiated one-sided RDMA) + NCCL (intra-node collectives)
  - 软件：CUDA/C++ 自定义 kernel (~300 行) + Python (~4K 行)

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 SGLang 评估端到端性能（TPOT, TPG）。
  - 微基准脚本测量：
    - AEBS scheduling overhead：测量 AEBS kernel 在不同 batch size (64–4096) 和 MoE 实例数 (8–16) 下的执行时间 (μs)
    - a_max 对比：记录 AEBS vs EPLB 在不同 batch size 和 MoE instance 数下的最大激活 expert 数
    - Communication overhead：消融 1PC vs 2PC vs EGate vs AGate 各组件
    - MoE-layer latency：测量单 MoE 层的 wall-clock 延迟
  - 修改内容：
    - **AEBS GPU Kernel**：CUDA kernel 实现 Algorithm 1，每 MoE 层在每个 MoE instance 的 default stream 上 launch。输入：token routing results (GPU global memory)、replica mapping (GPU constant memory)、instance metadata (更新频率低)。输出：per-token physical replica IDs (GPU global memory)。
    - **NVSHMEM Communication**：替换 SGLang 原有 intra-instance 数据移动为 NVSHMEM putmem_signal/signal_wait。发送端 kernel: prepare payload → nvshmem_putmem_signal(dest_pe, dest_addr, src_addr, size, signal_addr, signal_value)。接收端: nvshmem_signal_wait(signal_addr, expected_value) → 读取 payload。
    - **Metadata Packing**：将 layer index + token count 通过位运算打包为 64-bit signal value，CPU 侧仅在首 MoE 层 unpack 并缓存。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确提供 JANUS 独立开源仓库。基于开源 SGLang + NVSHMEM + NCCL 实现。
  - **AEBS GPU Kernel 执行原理全过程**：

    ```
    ┌── Kernel Input (GPU Global Memory) ───────────────────────┐
    │ L(i,j): logical EID for token i, expert j  [T × k, int]   │
    │ R(e): number of replicas for expert e       [E, int]       │
    │ G(e): set of instances hosting replicas     [E, list]      │
    │ P(e,g): physical RID on instance g          [E × n_e, int] │
    │ T: batch size, k: top-k, n_e: MoE instances                │
    └────────────────────────────────────────────────────────────┘

    ┌── AEBS CUDA Kernel (每个 MoE instance 独立执行) ──────────┐
    │ // Step 1: Collect activated expert set (GPU parallel)     │
    │ E_set = {}                                                 │
    │ Parallel for (i,j) in [0..T-1] × [0..k-1]:                │
    │     atomicOr(E_set_bitmap, L(i,j))  // 位图标记激活 expert │
    │                                                            │
    │ // Step 2: Initialize (per-instance state)                 │
    │ actRep[e] = -1 for all e in E_set  // selected replica    │
    │ load[g] = 0 for g = 1..n_e          // distinct expert cnt │
    │                                                            │
    │ // Step 3: Assign single-replica experts                   │
    │ for e in E_set where R(e) == 1:                           │
    │     g = unique instance in G(e)                            │
    │     actRep[e] = P(e,g)                                     │
    │     atomicAdd(load[g], 1)                                  │
    │                                                            │
    │ // Step 4: Assign multi-replica experts (greedy)           │
    │ for e in E_set where R(e) > 1:                            │
    │     g* = argmin_{g in G(e)} load[g]  // 最少负载实例      │
    │     actRep[e] = P(e, g*)                                   │
    │     atomicAdd(load[g*], 1)                                 │
    │                                                            │
    │ // Step 5: Rewrite token routing (GPU parallel)            │
    │ Parallel for (i,j) in [0..T-1] × [0..k-1]:                │
    │     O(i,j) = actRep[L(i,j)]  // 逻辑EID → 物理RID        │
    │                                                            │
    │ // 关键: 所有 MoE instances 用相同输入独立运行相同 kernel  │
    │ // 确定性算法保证一致性 → 无跨GPU协调                      │
    └────────────────────────────────────────────────────────────┘

    ┌── Dispatch (根据 O(i,j) 分发 token) ──────────────────────┐
    │ for each token i:                                          │
    │     for j in 1..k:                                         │
    │         dest_instance = instance_of(O(i,j))                │
    │         send activation[i] to dest_instance                │
    │         (通过 NVSHMEM one-sided put)                       │
    └────────────────────────────────────────────────────────────┘
    ```

  - **AEBS Scheduling Overhead 测量原理**：
    1. 在每个 MoE layer 的 forward path 中插入 CUDA event (cudaEventRecord) 测量 AEBS kernel 的 wall-clock 时间
    2. 变化 batch size (64, 128, 256, 512, 1024, 2048, 4096) 和 MoE instance 数 (8, 16)
    3. 关键结果：small batch (64) 下 <20μs，large batch (4096) 下 <90μs，始终远小于 MoE computation (~hundreds of μs)
    4. AEBS overhead 随 batch size 增长后趋于 plateau (因大部分 expert 已被激活)

  - **NVSHMEM Adaptive Two-Phase Communication 时序**：
    ```
    Case-1 (少量目标, 直接传输):
    Attention GPU A: [Aggregate intra-node via NCCL AllGather]
                   → [nvshmem_putmem_signal → MoE GPU E0, E1, ..., Em]
                   → [nvshmem_signal_wait ← MoE done]
    
    Case-2 (大量目标/数据, 中继传输):
    Attention GPU A: [Aggregate intra-node via NCCL]
                   → [nvshmem_putmem_signal → designated MoE relay GPU R]
    MoE relay R:     [nvshmem_signal_wait ← received]
                   → [intra-node NVLink multicast → local MoE instances]
    ```
