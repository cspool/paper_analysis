## ProMoE: Fast MoE-based LLM Serving using Proactive Caching

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：ProMoE 在 transformers 和 llama.cpp 两个主流 LLM 框架上实现 proactive caching 系统，通过 learned predictor + prefetcher 协调机制将 expert offloading 的数据传输移出推理关键路径。核心修改：(1) **Predictor 模块**——在 CPU 上运行二层的 MLP predictor（~2M 参数/层），基于前一层 hidden state 预测当前层将激活的 experts，以 layer-wise（或 stride）模式发出 prefetch 任务；(2) **Prefetcher 模块**——worker thread + 双优先级任务队列（LOW=speculative prefetch, HIGH=precise prefetch），通过 cudaMemcpyAsync 从 CPU memory 向 GPU memory 传输 expert chunk；(3) **Chunked Prefetching**——将每个 expert 参数按三个 linear layer 天然拆分为 3 个 chunk，使 worker thread 能以更细粒度调度，减少高优先级任务的等待延迟；(4) **Early Preemption**——在 gate function 完成后插入 hook 获取精确 expert 列表，清除同层 LOW 任务，将缺失 experts 作为 HIGH 任务入队；(5) **Reordered Inference**——在 gate 完成后根据 cache/prefetch 状态重排 expert 计算顺序：已缓存优先 → 正在 prefetch → 完全未开始，建立计算与 prefetch 的 pipeline；(6) **Cache 管理**——per-layer LRU cache，预分配连续 GPU memory 减少碎片。代码量 6,600 行 C++。
  - 实验比较：(1) Overall Performance（transformers 和 llama.cpp 两套 codebase）——ProMoE vs static cache vs LRU cache vs TO/UM/LO baselines，评估 5 个 MoE 模型（DS-1/DS-2/QW-1/QW-2/Mixt）的 TTFT 和 TPS/TPOT；(2) Ablation Study——逐步启用 prefetch/chunked-prefetch/early-preemption/reordered-inference 的加速贡献；(3) Impact of Cache Rate——cache rate 变化（10%-90%）对 TTFT 和 TPOT 的影响，含关键路径加载时间 breakdown；(4) Impact of Batch Size——batch size 1-4 对 prefill/decode throughput 的影响；(5) Impact of Model Size——BPW 从 4 到 16 变化对性能的影响。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA RTX 4090（24 GB GDDR6X）
  - CPU：Intel i9-14900K，128 GB host DRAM
  - 互联：PCIe 4.0（单向 32 GB/s，实测可达带宽 23.9 GB/s host-to-GPU）
  - 量化：FP16（DS-1/DS-2/QW-1）和 INT4（QW-2/Mixt），INT4 使用 GPTQ 量化

- 开源Serving框架是什么。修改了什么。
  - 框架：HuggingFace transformers（https://github.com/huggingface/transformers）和 llama.cpp（https://github.com/ggerganov/llama.cpp）
  - 修改内容（集成到两个框架）：
    1. **MoE 层 hooks**——在每层 gate function 结束后插入 hook 捕获 gating 输出，获取精确 expert 列表（用于 early preemption 和 reordered inference）
    2. **Expert 计算顺序重排**——修改 MoE block 中 expert FFN 的执行顺序，按 cache/prefetch 状态重排（cached-first → prefetching → not-started），建立计算-prefetch pipeline
    3. **Predictor 集成**——在每层前向开始时，将前一层的 hidden state clone 到 CPU，CPU 上执行 MLP predictor 预测当前层 experts，通过 PushPredictedExperts API 入队 LOW 优先级 prefetch 任务
    4. **Prefetcher worker thread**——在推理进程中启动独立线程，轮询双优先级任务队列，通过 cudaMemcpyAsync 执行 expert chunk 的 CPU→GPU 传输
    5. **Memory manager 接管**——ProMoE 接管 expert 参数的内存管理，预分配 per-layer LRU cache 连续 GPU memory 区域
    6. **Dependency mechanism**——实现依赖机制确保 prefetch 与 computation 的正确同步（expert.ready_chunk 计数器跟踪每个 expert 的已加载 chunk 数）
  - 未集成到 vLLM/TGI 的原因：当时支持 MoE 量化不完善，且这些框架 fused expert execution 假设所有 experts 计算前就绪（不适合 memory-constrained GPU 上的 prefetch 场景）

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：GitHub https://github.com/promoe-opensource/promoe
  - 框架输入到硬件执行全过程（ProMoE with transformers, DS-1 FP16, RTX 4090, single token decode）：
    ```
    初始化阶段：
    1. 非 expert 参数（attention, embedding, layernorm）常驻 GPU memory
    2. 全部 expert 参数在 CPU host memory，per-layer LRU cache 预分配 GPU memory
    3. 加载 offline 训练的 per-layer MLP predictor（~2M params/layer, 28 layers × 2M ≈ 56M params total）
    4. 启动 Prefetcher worker thread（CPU 线程，轮询双优先级任务队列）

    推理循环（第 l 层 MoE block）：
    5. Pre-attention norm 后 hidden state X ∈ R^{1×H} 被 clone 到 CPU
    6. CPU Predictor: 第 l-1 层 hidden state → MLP predictor_l → 预测第 l 层应 prefetch 的 experts
       → PushPredictedExperts(layer=l, experts=[e_pred_1,...,e_pred_k]) → LOW 优先级任务入队
    7. Prefetcher worker thread 开始异步传输预测 experts 的 chunk（CPU→GPU cudaMemcpyAsync）
    8. GPU 执行 self-attention（FlashAttention 或标准 attention）与 CPU prediction+prefetch 并行
    9. Gate function 完成 → hook 触发 → 获取精确 expert 列表 [e_1,...,e_k]
    10. Early Preemption: 清除第 l 层所有 LOW 任务 → Reordered: cached experts 排前 → PushPreciseExperts(layer=l, experts=reordered) → HIGH 任务入队
    11. Prefetcher worker thread 优先处理 HIGH 任务，以 chunk 粒度从 CPU 传输缺失 experts 到 GPU
    12. GPU 按重排顺序执行 expert FFN：cached experts 立即执行 → prefetching experts pipeline（计算与传输重叠） → 最后执行完全未开始 experts
    13. 每个 expert 计算时检查 ready_chunk 计数器，等待对应 chunk 就绪
    14. Expert 输出加权求和 → 进入第 l+1 层
    ```
  - 关键性能收益：ProMoE 将 key critical path 上的 expert 加载时间从 LRU 的 60.4% decode / 82.7% prefill 降至显著更低比例（QW-2 cache rate 增长时从 69.68% 降至 30.96%），通过 proactive prefetch 将大部分数据传输移出关键路径。
