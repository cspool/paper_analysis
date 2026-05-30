## Accelerating Distributed MoE Training and Inference with Lina

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 Lina 的 **Communication Micro-Op Scheduler**：将 all-to-all 和 allreduce 通信张量分区（tensor partitioning）为固定大小的 micro-ops（如 30MB chunk），通过 priority queue 运行时调度，保证 all-to-all micro-op 优先获满带宽，allreduce micro-op 仅在无 all-to-all 待发送时发射。同时引入 **all-to-all pipelining**：将 all-to-all 也分区为 micro-ops，使 FFN 计算可在第一个 all-to-all micro-op 完成后立即启动（token 粒度计算），消除计算等待时间。配合 **Expert Packing**（每 device 打包多个 expert，2^n 递增）对齐 FFN 与 all-to-all micro-op 时间，最大化 pipeline efficiency。

  实验比较:
  - **设计消融**: Baseline → +Priority → +Tensor Partitioning → +Pipelining → +Fixed Scheduling，在 2/4/8/16-expert 配置下比较 step time speedup
  - **Partition Size 敏感性**: 从 10MB 到 100MB 在 16-expert 模型上比较 step time
  - **Expert Packing 效果**: 比较 w/o Packing vs w/ Packing 的 pipelining efficiency（Transformer-XL: 33%→86%, GPT-2: 36%→85%, BERT2GPT2: 34%→79%）
  - **Overhead**: tensor partition/concatenation overhead 平均 1.02% step time; micro-op 传输 overhead 平均 1.7% 额外时间

- 后端平台是什么，配置是什么。
  4 节点 × 4 NVIDIA A100 GPU (40GB HBM)，节点间 100Gbps InfiniBand。Training 使用与 expert 数量等量的 GPU (2/4/8/16)。NCCL 2.10 底层通信。

- 评估性能的软件/脚本是什么。修改了什么。
  **评估工具**: PyTorch Profiler 采集 CUDA kernel 执行时间和 GPU 活动；training metrics 在 10-step warm-up 后 averaged over 50 steps；inference 在 test set 上平均。
  
  **修改内容**:
  1. **Tensor Partitioning**: 使用 LibTorch `chunk` 和 `cat` API 沿 token 维度将 gradient/activation tensor 分割为固定大小（30MB default）的 micro-ops；避免跨 gradient 混合 chunk 以简化 concat
  2. **Priority Queue Scheduler**: 每 device 单线程维护 priority queue；all-to-all micro-op 优先级高于 allreduce；当 backward pass 进入 combine computation 阶段时暂停 allreduce micro-ops 发射（预示 all-to-all 即将到来）
  3. **Expert Packing Coordinator**: MoE 模型中嵌入单线程 controller，在 forward pass 记录 all-to-all 和 FFN micro-op 时间 → 每 10 steps 调整 packing → 需要时插入 one-time synchronous all-to-all 交换 expert params（下次 iteration 生效）→ multi-stream parallel execution（多 expert forward/backward）
  4. **All-to-all Pipelining**: 将 all-to-all dispatch 分区为 micro-op → 每个 micro-op 完成后立即启动对应 token subset 的 Expert FFN → combine 阶段同理

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  **开源情况**: 论文基于 DeepSpeed MoE (https://github.com/microsoft/DeepSpeed) 和 PyTorch 构建，约 7500 LoC C++/Python 修改，但论文未提供独立的 Lina 开源仓库链接。

  **Communication Micro-Op 调度全过程（以 backward pass 中一个 MoE layer 为例）**:
  1. **输入**: 计算 stream (Stream a) 完成 expert FFN 反向计算后，梯度分别进入 expert-parallel 通信 (Stream b, all-to-all) 和 data-parallel 通信 (Stream c, allreduce)
  2. **Tensor Partition**: 15MB gradient tensor → `tensor.chunk(chunk_size=30MB)` → 5 个 30MB micro-ops 入队
  3. **Priority Queue 调度逻辑**（单线程 per device）:
     ```
     while queue not empty:
       if queue has all-to-all micro-op:
         pop and launch all-to-all micro-op (NCCL all-to-all)
         wait for completion
       else if queue has allreduce micro-op AND combine_computation not yet started:
         pop and launch allreduce micro-op (NCCL allreduce)
       else:
         idle (等待下一 micro-op 入队或 combine 阶段结束)
     ```
  4. **All-to-all Pipelining**: all-to-all dispatch 分 3 个 micro-ops → micro-op 1 完成后 1/3 tokens 进入 FFN → micro-op 2 完成 +1/3 → micro-op 3 完成 +1/3
  5. **Expert Packing 决策**: 记录 FFN micro-op time vs all-to-all micro-op time → 若 FFN << all-to-all → packing_factor *= 2 → 下次 iteration 生效
  6. **输出**: 所有 allreduce micro-ops 完成后 optimizer step

  **Baseline 对比**:
  - Baseline 中 Stream b (all-to-all) 和 Stream c (allreduce) 分别独立发射完整的大张量通信原语 → NCCL 底层无协调地公平共享 InfiniBand 带宽 → median all-to-all slowdown 1.83x (worst 4.14x)
  - Lina: micro-ops 使 all-to-all 与 allreduce 不并发 → all-to-all 获得满带宽 → all-to-all time speedup 2.21x~2.39x; step time speedup 1.37x~1.73x vs DeepSpeed
