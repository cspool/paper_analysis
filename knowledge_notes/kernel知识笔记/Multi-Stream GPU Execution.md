## Multi-Stream GPU Execution

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Stream GPU Execution（多流GPU执行）是将 GPU 计算任务（kernel launches）分配到多个 CUDA stream 上并行执行的技术。每个 CUDA stream 是一个 FIFO 命令队列，同一 stream 内的操作按序执行，不同 stream 间的操作可以在 GPU 上真正并发执行——前提是硬件资源（SM 数量、shared memory、register file）允许多个 kernel 同时驻留。

与 CPU 多线程的类比：一个 CUDA stream 类似一个线程——同一线程内的指令串行，多线程可并发。但 GPU 的并发约束更严格：kernel 的 thread blocks 必须竞争 SM 资源，只有资源足够时不同 kernel 的 blocks 才能同时执行。

在深度学习场景中，multi-stream 的价值在于：DL 模型的 DAG 中通常存在多条独立分支（如 NASNet cell、Inception module、multi-head attention），这些分支的算子间无数据依赖，可以并行执行。但默认的 PyTorch eager mode 将所有 kernel 提交到单一的 default stream，导致这些分支被串行化。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Nimble 的 multi-stream 执行流程（以 NASNet-A Normal Cell 为例）：

```
// NASNet-A Normal Cell 的 DAG 结构:
// input → 
//   branch_0: sep_conv_5x5_1 → sep_conv_5x5_2 ─┐
//   branch_1: sep_conv_3x3_1 → sep_conv_3x3_2 ─┤
//   branch_2: sep_conv_5x5_3 → sep_conv_5x5_4 ─┤
//   branch_3: sep_conv_3x3_3 → sep_conv_3x3_4 ─┤
//   branch_4: sep_conv_5x5_5 → sep_conv_5x5_6 ─┤
//   branch_5: avg_pool ─────────────────────────┤
//   → concat (所有 branch 汇聚)

// Stream Assignment (由 Nimble Graph Rewriter 自动完成):
// stream_0: sep_conv_5x5_1 → sep_conv_5x5_2 → concat_input_0
// stream_1: sep_conv_3x3_1 → sep_conv_3x3_2 → concat_input_1
// stream_2: sep_conv_5x5_3 → sep_conv_5x5_4 → concat_input_2
// stream_3: sep_conv_3x3_3 → sep_conv_3x3_4 → concat_input_3
// stream_4: sep_conv_5x5_5 → sep_conv_5x5_6 → concat_input_4
// stream_5: avg_pool                      → concat_input_5
// ──── CUDA event sync barrier ────
// stream_0: concat → batch_norm → relu → ...

// GPU Timeline (简化):
// Time ─────────────────────────────────────────────────────────→
// SM0: |sep5_1|sep5_2|       |concat|bn|relu|...
// SM1: |sep3_1|sep3_2|       |               ...
// SM2: |sep5_3|sep5_4|       |               ...
// SM3: |sep3_3|sep3_4|       |               ...
// SM4: |sep5_5|sep5_6|       |               ...
// SM5: |avg_pool |           |               ...
//                             ↑ event sync (所有分支完成)
```

Multi-stream 在 serving/deployment 场景的实现模式：
```
// 常见模式 1: Compute-Memcpy Overlap
stream_compute: kernel_A → kernel_B → kernel_C
stream_copy:    cudaMemcpyAsync(H→D) → cudaMemcpyAsync(D→H)
// kernel_B 执行时，stream_copy 可同时进行数据传输

// 常见模式 2: Prefill-Decode Concurrency (LLM serving)
stream_prefill: prefill_layer_0 → prefill_layer_1 → ... → prefill_layer_N
stream_decode:  decode_iter_0 → decode_iter_1 → ... → decode_iter_M
// 两个 stream 的 kernel 共享 SM 资源（spatial multiplexing）

// 常见模式 3: 分支并行 (Nimble 的核心用例)
// 多个独立计算分支分布到多 stream，汇聚点 sync
stream_0: branch_A_subgraph_0 → branch_A_subgraph_1 → sync_point
stream_1: branch_B_subgraph_0 → branch_B_subgraph_1 → sync_point
// → sync (cudaEventSynchronize/cudaStreamWaitEvent)
//   → merged_subgraph
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
1. **手动编程**：`cudaStreamCreate(&s)` → `kernel<<<grid, block, shmem, s>>>()` → `cudaStreamSynchronize(s)` → `cudaStreamDestroy(s)`。需要开发者手动分析数据依赖并插入 CUDA event 同步。
2. **框架自动并行**：Nimble 通过 Graph Rewriter 自动分析 DAG 并进行 stream assignment，用户无需编写 CUDA stream/event 代码。
3. **编译器自动生成**：如 HuntKTm 通过 LLVM pass 自动发现 kernel 间数据依赖并生成多 stream 代码。

关键同步原语：
- `cudaEvent_t`：跨 stream 同步点。Producer stream 在完成计算后 `cudaEventRecord(event, stream)`，Consumer stream 在消费数据前 `cudaStreamWaitEvent(stream, event)`。
- Nimble 的最小同步原则：仅在有数据依赖的跨 stream 算子间插入 event，利用 MEG 消除冗余传递依赖。

限制和注意事项：
- NULL stream (default stream) 具有隐式全局同步语义——使用 NULL stream 会串行化所有其他 stream → 必须使用 per-thread default stream 或显式创建 non-blocking streams
- 多 stream 的加速比受限于 DAG 的逻辑并发度（logical concurrency）——如果模型大部分算子串行（如 VGG 式的线性 chain），多 stream 无加速效果
- Stream 数量过多可能导致 context switching/scheduling overhead → Nimble 通过 stream assignment algorithm 确定最优 stream 数量

涉及论文标题：
- Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---
