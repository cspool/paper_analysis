## Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：提出 AMoE——一个兼容 vLLM 的异步 Expert Parallelism (AEP) 原型系统，从零实现 6K 行 Python + 4.8K 行 C++. 核心实现包括：
    - **µ-queuing（微队列）**：将 token 按 layer 粒度而非全局 batch 排队，每个 GPU 为每个 expert layer 维护独立的 µ-queue，使 GPU 可以自由选择任意 ready layer 执行。
    - **Defragging Scheduler（Algorithm 1）**：为每个 (block, expert) pair 计算 Score = LScore + Q[b][e]，其中 LScore = sum_{k=1}^{K} (TotalTokens_{b+k} / N_e) × δ^k，lookahead K 个 block 并以衰减因子 δ 加权，优先调度下游 token 密集的 layer 以合并碎片化 mini-batch。
    - **Attention-Expert 解耦架构**：attention 层使用 Data Parallelism (DP) 部署在部分 GPU，expert 层使用 Expert Parallelism (EP) 部署在其余 GPU，两类 GPU 异步通信不阻塞。
    - **两阶段异步通信（Figure 8）**：Phase 1 通过 ZeroMQ CPU 消息队列传递 metadata (tensor size, rank)；Phase 2 通过 NCCL P2P 直接 GPU-to-GPU 传输 tensor，CPU 不等待 NCCL 完成即处理下一个传输任务。
    - **Token 级依赖追踪**：每个 token 携带 metadata <RequestID, LayerID, Tensors[], prefill_length, topk_weights>，使异步重排序执行中仍可正确追踪请求归属和下一层目标。
    - **Coordinator-Runtime 架构**：Coordinator (CPU) 包含 API Server (tokenizer/detokenizer, 请求状态管理)、Load Balancer (按 GPU memory 分配 DP rank)、Cluster Manager (GPU 内存追踪，通信通道建立)；每个 GPU 一个 Runtime 实例负责该 GPU 上所有层的执行。
    - **CUDA Graphs 逐层优化**：为每个 layer 独立记录 G 个 CUDA Graph（不同 batch size），共计 L×G 个 graph，通过共享 input buffer 减轻 GPU memory 压力，但 expert 层不使用 graph（因 GEMM 主导时 kernel launch latency 可被第一个 GEMM 掩盖）。
    - **Execution 四阶段流水线**：Receptor (按 LayerID 分流入 µ-queue) → Scheduler (选最优 layer) → Executor (page table 管理 + kernel launch) → Dispatcher (按 expert/DP rank permute tokens 后发送)。
  - 实验比较：
    - (a) **Top-1 routing throughput-latency (Figure 9a-c)**：AMoE vs SGLang (EP)，Mixtral 8x7B，8× A100 80GB。在 Short/Medium/Reasonable 三种 workload 下，AMoE throughput 分别提升 2.7×/2.3×/2.0×；低负载下 AMoE ITL 略高（layer-wise scheduling overhead + attention disaggregation 延迟）。
    - (b) **Top-2 routing (Figure 9d-f)**：AMoE throughput 优势减小，因 Top-2 (12.5%→25% expert activation) 降低 load skew，且 token merge 引入部分同步点。
    - (c) **多节点可扩展性 (Figure 10)**：16 experts + 16 GPUs (2×AWS P4)，medium workload + Top-1。AMoE throughput 3× vs SGLang，从 8→16 GPU 实现 1.92× 线性扩展，SGLang 无扩展。
    - (d) **Scheduler 消融 (Figure 11/12)**：defragging vs MTFS (most-token-first) vs FLFS (first-layer-first) 在 80% 最大 throughput 下的 ITL 和 throughput；FLFS 存在新请求打断高层 block 导致 输出率低于输入率 的 live-lock 问题。
    - (e) **Execution breakdown (Figure 13)**：attention step 2.7ms (page table overhead 显著)，expert step 0.8ms (GEMM 计算主导)，scheduling stage 仅占总时间小部分（C++/CUDA 优化效果）。
- 硬件平台是什么，配置是什么。
  - 单节点 (Lambda)：8× NVIDIA A100-SXM4-80GB，NVSwitch 600 GB/s per GPU，CUDA 12.8，NCCL 2.25.1，2× AMD EPYC 64 cores，1800 GB RAM，Ubuntu 22.04。
  - 多节点 (AWS P4)：2× p4dn.24xlarge，每节点 8× A100-SXM4-40GB，NVSwitch 600 GB/s，4× 100 Gbps EFA，CUDA 12.4，cuDNN v9.1.0，NCCL 2.22.3，CPU 2× AMD EPYC 64 cores @ 1.5 GHz，988 GB RAM。
- 开源Serving框架是什么。修改了什么。
  - 框架：AMoE 从零构建，但 runtime 中 model executor 复用了 vLLM 的 paged attention 和 CUDA graph 等优化基础设施。Communicator, Receptor, Scheduler, Dispatcher 用 C++ 实现 + pybind11 暴露 Python 接口，以规避 Python GIL 并保证各组件并发运行。Scheduler 和 Executor 在主 Python 线程运行，Receptor 和 Dispatcher 在独立 POSIX 后端线程运行。
  - 关键修改（相对标准 EP serving）：
    1. **Scheduling**：从全模型同步 batch 调度 → 逐层异步 µ-queuing + defragging scheduler（Algorithm 1）
    2. **Communication**：从 barrier all-to-all → ZeroMQ (CPU metadata) + NCCL P2P (GPU tensor) 两阶段异步通信
    3. **Execution model**：从固定 batch 遍历所有 layer → GPU 按 Score 自主选 layer 执行，cold expert tokens 积累到足够 batch size 才执行
    4. **Architecture**：Attention-Expert disaggregation → 不同类型层部署到不同 GPU 组，独立扩展
    5. **Token tracking**：新增 metadata-based token dependency tracking 支持异步乱序执行
- 开源情况。论文声明将开源 AMoE（"We open-source our serving system, AMoE, for public use"），**但论文全文及 arXiv 页面均未给出具体 GitHub URL，当前无法确认开源仓库地址。**
  
  基于论文描述的 AMoE Serving 全流程（Figure 5-6 对应）：

  ```
  [请求到达]
      │
      ▼
  API Server (Coordinator/CPU)
    ├─ tokenizer: 将 request text → token embeddings
    ├─ Load Balancer: 按 GPU memory 选最空闲 attention DP rank
    └─ 为 token 附加 metadata → 发送至对应 GPU Runtime
      │
      ▼
  [GPU Runtime - Attention Worker]
      │
      ├─ Communicator (Phase 1: ZeroMQ CPU metadata exchange)
      │     └─ sender 告知 receiver tensor size + GPU rank
      ├─ Communicator (Phase 2: NCCL P2P GPU direct transfer)
      │     └─ ncclSend/ncclRecv on CUDA stream, CPU 不等待完成
      ▼
  Receptor (C++ POSIX thread)
    ├─ 按 token.LayerID 将 token 分入对应 (block#, expert#|attnDPrank) µ-queue
    └─ Top-K: token pool 等待 K 路输入全部到达 → merge → 入队
      │
      ▼
  Scheduler (Defragging Algorithm 1, main Python thread)
    ├─ 遍历所有 (block, expert) pairs
    ├─ 计算 Scores[b][e] = LScore(lookahead) + Q[b][e]
    └─ 选 argmax → drain 该 µ-queue 全部 tokens → 合成 batch
      │
      ▼
  Executor
    ├─ [Attention layer] Page Table: 为 new tokens 分配 KV cache slot
    ├─ Pre-processing: fused batched CPU→GPU metadata transfer (dedicated stream)
    ├─ Kernel: paged attention or expert GEMM (mixture of kernels)
    ├─ Post-processing: GPU→CPU routing info (expert indices + weights) 
    └─ CUDA Graph: attention 层用预录制 graph 加速小 batch；expert 层不用 (GEMM 掩盖 kernel launch)
      │
      ▼
  Dispatcher (C++ POSIX thread)
    ├─ Attention output → permute by expert ID → 分组发送到各 expert GPU
    ├─ Expert output → permute by attention DP rank → 发送回 attention GPU
    └─ 递增 LayerID → 循环到下一 block
      │
      ▼
  [最后一层 attention output → Sampler (在首层 attention GPU)]
      └─ sample next token → detokenizer (API Server) → 返回用户
  ```
