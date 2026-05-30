## Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：QLLM——基于 HuggingFace TGI 构建的 priority-aware 抢占式调度推理系统。核心设计：(1) Per-Expert FIFO Queues：每个 expert 拥有独立队列，token 按需入队，打破传统 layer-wise 同步 barrier；(2) Priority-Aware Scheduler：Dispatcher 将请求按优先级（LS/BE）分别入队（LS_PrefillQueue, LS_DecodeQueue, BE_PrefillQueue, BE_DecodeQueue），Batch Engine 按 Algorithm 1 优先级排序组 batch；(3) Expert-Level Preemption：在任意 layer 内收到 LS 请求时，Scheduler 立即通知 Inference Engine 停止 BE batch 执行，执行 LS prefill+decode 后动态合并 LS/BE 到同一 batch；(4) Unified Sequence/Batch Abstraction：用 Facade Pattern 封装每个 Sequence 的独立 tensor（KV cache, hidden states, routing_weights），对外呈现为统一 batch tensor 接口，支持零拷贝的 individual sequence 状态更新而无需 split-merge；(5) Unified Dynamic KV Cache：解耦 sequence-level 和 batch-level cache 操作，避免大 KV tensor 的 split-merge 开销；(6) Closed-Loop Feedback Controller：Inference Engine 在每个 attention 和 router stage 后回调 Scheduler，Scheduler 根据 user-defined policy 动态调整执行流。
  - 实验比较：(1) TTFT——QLLM vs HF TGI baseline，在 ShareGPT 数据集、20% LS/80% BE 混合负载下，Poisson 到达率从低到高；(2) Throughput——job completion rate vs request arrival rate；(3) Turnaround time——LS 和 BE 两类请求的 turnaround time 对比。QLLM 降低 LS TTFT 平均 65.2×（最高 101.6×），SLO 设置为 3s（10× 单次 decode iteration），QLLM 在 7 req/s 以内满足 SLO 而 baseline 任何负载下均不满足。LS turnaround time 降低最高 12.8×，BE turnaround time 增加 1.38×（最高 2.04×）。Throughput 持平或略优。
- 硬件平台是什么，配置是什么。
  - GPU: 单卡 NVIDIA A100 80GB HBM。CPU: 双路 Intel Xeon Gold 6336Y。DRAM: 256 GB。互联: PCIe 4.0。系统环境: bare-metal。
- 开源Serving框架是什么。修改了什么。
  - 开源框架：HuggingFace TGI (Text Generation Inference)，https://github.com/huggingface/text-generation-inference。
  - 修改内容：
    1. **MoE Layer 重设计**：在原有 MoE block 中插入 per-expert FIFO queues。每个 expert 队列独立存储待处理 token 的 Sequence 引用，eliminate layer-wise synchronization barrier。Router 输出的 top-k expert 选择结果将 sequence 引用 push 进对应 expert 队列，expert 从其队列中 pop 处理。
    2. **Scheduler 模块**：新增 Dispatcher（按优先级分派 jobs 到四个队列）和 Batch Engine（Algorithm 1：优先 LS_Decode → LS_Prefill → 填充 BE → BE_Decode → BE_Prefill）。Scheduler 在 LS 到达时通过 closed-loop feedback 通知 Engine 在任意 layer 处 preempt BE batch。
    3. **Sequence/Batch 抽象层**：新增 Sequence 对象封装 per-token 全部状态（KV cache tensors, hidden states, attention mask, residuals, routing metadata）。Batch 对象用 Facade Pattern 对外呈现为单一拼接 tensor，对内维护 per-sequence 独立 tensor，支持 zero-copy individual update。
    4. **Unified Dynamic KV Cache**：新增 KV cache 管理模块，解耦 sequence-level 和 batch-level cache ops，避免传统系统中 preempt 时的大 tensor split-merge。
    5. **Inference Engine** 改为 closed-loop feedback controller：每层 attention/router 阶段后回调 Scheduler，支持 user-defined preemption policy（<50 行 Python 实现）。
  - 论文声明 QLLM 是建立在 HF TGI 之上的原型系统，计划未来开源（"Our ultimate plan is to release QLLM as an open-source project in future versions"）。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：论文尚未开源（计划未来开源）。基于论文描述复原的全过程如下。
  - 框架输入到硬件执行全过程（Mixtral 8×7B, 4-bit 量化, FP16, batch_size=32, single A100 80GB）：
    
    **阶段 0 — 系统初始化与模型加载**：
    1. 基于 HF TGI 源码，继承其模型加载逻辑，加载 Mixtral 8×7B 权重（4-bit 量化，FP16 compute，约 22.93 GB GPU memory）。
    2. 替换原生 MoE block 为 QLLM MoE block：在每个 expert 前插入 FIFO queue 数据结构。初始化 Scheduler（Dispatcher + Batch Engine）和四个优先级队列。
    3. 初始化 Unified Dynamic KV Cache 管理器：预先分配 GPU memory pool 用于 dynamic cache 增长。
    
    **阶段 1 — 请求到达与调度（BE 先到，LS 随后）**：
    4. 4 个 BE decode requests（ShareGPT prompts）到达。Dispatcher 将它们 enqueue 到 BE_DecodeQueue。
    5. Scheduler Batch Engine 调用 GetNextBatch()：LS_DecodeQueue 和 LS_PrefillQueue 均为空 → 检查 BE_DecodeQueue 有 4 个 jobs → 返回 batch=[BE1, BE2, BE3, BE4]。
    6. Scheduler 将 batch 提交给 Inference Engine，进入 MoE layer 1 的 attention 计算。
    
    **阶段 2 — Expert-Level Preemption（LS 到达触发）**：
    7. 在 MoE layer 1 的 attention 阶段完成后，一个 LS job 到达。Dispatcher 将其 enqueue 到 LS_PrefillQueue。
    8. Closed-loop feedback：Inference Engine 在 attention 完成后回调 Scheduler。Scheduler 检测 LS_PrefillQueue 非空 → 发送 preempt 信号给 Engine。
    9. Engine 在 layer 1 的 router 阶段后暂停 BE batch：当前 BE batch 的 partial computation 状态（hidden states, routing_weights, expert assignments）通过 Sequence 对象的独立 tensor 原地保存，不需要 split BE batch tensor。
    10. Engine 切换到 LS job：执行 prefill——在 GPU SM 上并行处理 LS prompt 的全部 input tokens，生成 KV cache entries 并产出第一个 output token。
    11. LS job 转入 decode phase，Engine 执行 LS decode iteration 生成后续 token。
    
    **阶段 3 — 动态合并与恢复（LS 与 BE 同 batch 执行）**：
    12. LS decode 完成后，Scheduler 通过 Batch Engine 将 LS job 加入当前运行 batch。由于 Batch 的 Facade Pattern 对外表现为单一 tensor，model 无感知 batch composition 变化。
    13. Engine 在后续 layers 中同时处理 LS decode + BE decode：LS job 的 token 走 router → top-k expert selection → push 进入对应 expert queue → expert 从 queue pop 并执行 feed-forward。
    14. BE jobs 从 preemption point 恢复执行：Sequence 对象中保存的 routing_weights 和 hidden_states 被重新加载，Unified Dynamic KV Cache 恢复 BE 的 cache 行，无需 recomputation。
    
    **阶段 4 — 后续 Layers 的 Per-Expert Queue 执行**：
    15. 每个 MoE layer：Router 为每个 token（LS/BE 混合）计算 gating logits → softmax → TopK=2 → 将 sequence 引用 push 进选中的 2 个 expert 队列。
    16. 各 expert 独立从其 FIFO queue 中 pop token 执行 FFN 计算。Per-expert queue 的 FIFO 顺序确保 LS token 优先（因为 LS sequence 引用先于 BE 入队前的 preemption 点入队，或者通过 policy 显式优先 enqueue LS）。
    17. Expert 输出写入 Sequence 对象的 hidden_states tensor。Batch Facade 收集所有 Sequence 的 output 拼接为下一 layer 的 input tensor。
    18. 重复 layer 1..32（Mixtral 8×7B 的 32 层），每层 attention → router → per-expert queue → expert FFN → combine。
    
    **阶段 5 — 输出与完成**：
    19. 最终 layer 输出经 LM head 投影到 vocabulary，生成 logits → softmax → sample → output token。
    20. LS job 完成所有 decode iterations → Dispatcher 将其移出队列，output tokens 返回客户端。BE jobs 继续执行直至各自完成。
