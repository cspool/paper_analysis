## Asynchronous Expert Parallelism (AEP / 异步专家并行)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asynchronous Expert Parallelism (AEP) 是一种新的 MoE 推理服务范式，由 AMoE 系统（arXiv:2505.08944）提出。AEP 的核心思想是将 MoE 推理中的层执行（layer execution）从 barrier 式同步中解耦：每个 GPU 不在每个 decoding block 的 expert 层前后等待 all-to-all barrier 通信完成，而是动态地将 token 排队到每个 layer 级别的微队列（µ-queue），自适应地按需重新批处理（re-batch），持续执行任意 ready 的 layer。传统 EP 要求在 expert 层前后有 all-to-all barrier——所有 GPU 必须等待最慢的 expert GPU 完成计算并完成 group communication 后才能进入下一层。AEP 彻底消除了这种 barrier：每个 GPU 独立决策执行哪个 block 的哪个 expert 层，不等待其他 GPU。这种异步方法缓解了两个主要低效问题：(1) 等待最热 expert 时的 GPU 空闲时间；(2) 冷门 expert 上的小 batch 执行浪费 HBM 带宽。

从系统架构角度拆解术语，给出术语在系统架构中运转流程的具体例子。
AEP 的核心执行流程：
1. **Token 到达**：每个 GPU Runtime 通过 Communicator（ZeroMQ CPU metadata + NCCL P2P GPU tensor 两阶段异步传输）接收来自其他 GPU 的 token 数据。
2. **µ-queuing**：Receptor 按 token 的 LayerID 将 token 分离入对应 block+expert 的 µ-queue，而非放入全局 batch。
3. **Defragging Scheduling**：当 GPU 空闲时，Scheduler 遍历所有 (block, expert) pairs，计算 Score = LScore（lookahead 加权前方 token 密度） + Q[b][e]（当前队列 token 数），选择最高分 layer 执行。
4. **自适应 Batch**：Executor 从选中 µ-queue 中 drain 所有 tokens，合并为连续 batch 进行 GEMM（expert）或 paged attention（attention）计算。
5. **下一层转发**：Dispatcher 将输出 token 按下一层目标 expert 或 attention DP rank 分组，通过 Communicator 异步发送。

关键：Step 2-5 之间不存在全局 barrier。Hot expert 的 µ-queue 快速积累 tokens，Score 高 → 优先调度执行；Cold expert 的 Queue 积累慢，Score 低 → 自然被延迟，token 继续积累直到达到高效 batch size。

AEP 与传统 EP 的对比：
- EP：token batch → all-to-all barrier → expert compute → all-to-all barrier → 下一层 → GPU 在 barrier 空闲
- AEP：token → µ-queue → [GPU 独立选层执行] → 转发 → cold expert tokens 积累无需 stall

术语一般如何实现？如何使用？
AEP 通过 AMoE 系统实现：(1) Coordinator (CPU) + Runtime (每 GPU 一个) 架构；(2) Receptor (C++ POSIX thread) 负责 token 分流入队；(3) Defragging Scheduler (Algorithm 1) 选择最优 layer；(4) Executor 执行 kernel；(5) Dispatcher (C++ POSIX thread) 路由转发。实现使用 6K 行 Python + 4.8K 行 C++（pybind11），兼容 vLLM。AEP 最适用于 expert computation 为主的 decoding 工作负载，且 expert load skew 越大，AEP 相对于 EP 的优势越大。Top-K > 1 时因 token merge 引入部分同步，AEP 收益有所下降。

涉及论文标题：
- Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony
