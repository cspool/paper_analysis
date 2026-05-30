## Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony

- baseline方法是什么？
  Baseline 是标准 Expert Parallelism (EP) MoE serving，以 SGLang [49] 为代表性系统。Baseline 的执行模式：
  - **全栈执行例子**：请求到达 → tokenizer → 整 batch 同步执行 decoding block 0 attention → **barrier all-to-all** → expert GPU 并行执行 expert layers → **barrier all-to-all** → block 1 attention → ... → block N → sampler → detokenizer。所有 GPU 在每个 barrier 等待最慢的 expert 完成（straggler effect）。
  - **Baseline 的缺陷**：
    1. **GPU stall on straggler experts**：hot expert 接收最多 tokens，计算时间显著长于 cold expert，其他 GPU 被迫空等。实验中 GPU stall 可达总时间的 70%（Figure 4）。
    2. **Cold expert 小 batch 低效执行**：cold expert 在每次 barrier 前只有少量 tokens，batch size 小 → HBM weight loading 时间主导 → GPU 计算单元利用率低。Figure 3 显示 batch < 128 时 throughput 远低于线性。
    3. **Barrier all-to-all 通信开销固定**：无论 load 如何偏斜，all-to-all 必须在全部分参与 GPU 间同步，无法通过增加 GPU 缓解。
    4. **无法利用动态负载变化**：expert load skew 随时间变化 [11,21,23,31]，固定 batch 执行无法自适应。
  - 缺失层次：编译框架（论文未涉及编译框架层）；kernel 调度（使用 NCCL + vLLM 现有 kernel，未提出新 kernel）；硬件架构/芯片设计（使用商用 A100 GPU）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Asynchronous Expert Parallelism (AEP)** 并通过 **AMoE** 系统实现，核心设计解决 baseline 各缺陷：
  
  - **全栈执行例子（AMoE）**：
    1. 请求到达 → tokenizer (API Server/CPU) → Load Balancer 选 attention DP rank
    2. Token 携带 metadata 发送到 attention GPU → Receptor 按 LayerID 入对应 block 的 µ-queue
    3. Defragging Scheduler 计算所有 layer 的 Score → 选最高分 layer 执行（无 barrier）
    4. Executor 对选中 layer：page table (attention) / GEMM (expert) → Dispatcher 按目标 GPU permute tokens
    5. Communicator: ZeroMQ (CPU) metadata 交换 → NCCL P2P (GPU) 异步传输（CPU 不等待 NCCL 完成）
    6. 循环 step 3-5，各 GPU 独立异步执行 → sampler → detokenizer
  
  - **解决 baseline 缺陷的对应设计**：
    1. **消除 straggler stall（对应缺陷 1）**：AEP 完全去除 all-to-all barrier。GPU 完成当前层后立即从 µ-queue 拉下一批 token 执行任意 ready layer，不存在"等待最慢 expert"。Hot expert GPU 持续处理积累的 tokens，cold expert GPU 可转而执行 token 充足的其他 layer。
    2. **自适应 batch size（对应缺陷 2）**：µ-queuing 允许 cold expert 的 tokens 在队列中积累，直到 batch size 足够大（接近高效的 batch=128 区域），由 defragging scheduler 的 Score 机制自动延迟低 token 计数的 layer 的调度。
    3. **异步 P2P 通信替代 all-to-all（对应缺陷 3）**：用 ZeroMQ + NCCL P2P 的异步点对点传输替代全局 barrier all-to-all，发送方 CPU 启动 NCCL 后立即继续处理下一个任务，接收方异步同步。通信自然与计算重叠。
    4. **动态自适应负载（对应缺陷 4）**：Defragging Scheduler 每步基于当前 µ-queue 状态动态决策最优 layer，lookahead 机制鼓励 token wave 向前传播并保持 compact，无需 profiling hot/cold experts 或预分配 expert replica。
  
  - **额外的系统级创新**：
    - **Attention-Expert 解耦**：attention 和 expert 层可独立扩展 GPU 数量（e.g., 4 attention GPUs + 4 expert GPUs），解决 KV cache 容量瓶颈限制并发请求数的问题。
    - **FLFS/MTFS 平衡调度**：在完全 defrag (FLFS) 和纯 throughput (MTFS) 之间折中——通过 lookahead decay δ 控制，避免 FLFS 的新请求 live-lock 和 MTFS 的 batch fragmentation。
    - **C++ 关键路径优化**：Receptor, Scheduler, Communicator, Dispatcher 用 C++ 实现（pybind11），避开 Python GIL，scheduling overhead 仅占执行时间的小部分（Figure 13）。
  - 缺失层次：编译框架（论文未涉及编译框架修改）；kernel 调度（使用 vLLM/NCCL 现有 kernel，未提出新 kernel 算子）；硬件架构/芯片设计（使用商用 A100 GPU）。
