## MoE Adaptive Pipelining（MoE 自适应流水线 / Multi-stream Token Partition）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Adaptive Pipelining 是 TUTEL 的运行时优化技术，通过将 MoE 层的 All-to-All 通信与 Expert FFN 计算在不同 CUDA stream 上重叠执行，最大化 GPU 利用率。与传统的 batch-splitting 或 pipeline parallelism（会放大 MoE dispatch 不均衡和破坏 Batch Prioritized Routing 正确性）不同，TUTEL 仅在 All-to-All 和 Expert 区间内沿 capacity 维度分区。每 iteration 通过查字典选择最优的流水线度 d∈{1,2,4,8} 和 All-to-All 算法 a∈{Linear,2DH}。

从kernel调度角度拆解：

Multi-stream Token Partition 的调度时序（以 d=2 为例）：

```python
# Adaptive Pipelining with degree d=2
def moe_layer_pipelined_forward(input, gate_output, d=2, algo='2DH'):
    # input: (E, C_g, D) — token dispatches per expert
    # Step 1: Partition along capacity dimension
    partitions = input.chunk(d, dim=1)  # 2 chunks of (E, C_g/2, D)
    
    streams = [torch.cuda.Stream() for _ in range(d)]
    all2all_outputs = [None] * d
    expert_outputs = [None] * d
    
    # Step 2: Pipeline dispatch across streams
    for i in range(d):
        with torch.cuda.stream(streams[i]):
            # A2A Dispatch (communication stream)
            all2all_outputs[i] = all_to_all(partitions[i], algo=algo)
            # → shape: (E_g, C/2, D) for Flexible A2A
    
    # Step 3: Wait for A2A dispatch, then Expert FFN
    for i in range(d):
        streams[i].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(streams[i]):
            expert_outputs[i] = expert_ffn(all2all_outputs[i])
            # Expert compute on GPU compute stream
            # Stream i+1's A2A overlaps with Stream i's Expert FFN
    
    # Step 4: A2A Combine
    combined = [None] * d
    for i in range(d):
        with torch.cuda.stream(streams[i]):
            combined[i] = all_to_all(expert_outputs[i], algo=algo)
    
    # Step 5: Barrier and merge
    for s in streams:
        s.synchronize()
    output = torch.cat(combined, dim=1)  # (E_g, C, D) or (E, C_g, D)
    return output
```

时序图（Gantt）：
```
Time →     |---- A2A_dispatch_0 ----|         |-- A2A_combine_0 --|
           |    |-- A2A_dispatch_1 ----|       |    |-- A2A_combine_1 --|
           |    |    |-- Expert_FFN_0 --|      |    |    |-- Expert_FFN_1 --|
Stream 0:  [====A2A_dispatch_0====][====Expert_FFN_0====][====A2A_combine_0====]
Stream 1:       [====A2A_dispatch_1====][====Expert_FFN_1====][====A2A_combine_1====]
Compute:        |<-- A2A and FFN overlapped -->|
```

关键设计：(1) 仅对 All-to-All + Expert 区间分区，不影响 gating 和 MoE 层外操作；(2) 不破坏 Batch Prioritized Routing 的语义正确性；(3) 自定义 inline reshape 操作无需额外数据拷贝；(4) 最优度 d 通过预构建字典 O(1) 查找，搜索空间 {1,2,4,8}（度 > 8 几乎不改进 overlap 但增大 A2A 开销）。

术语一般如何实现？如何使用？

基于 PyTorch CUDA Stream API 实现多流调度，All-to-All 操作使用定制的可接受分片输入的通信 kernel。TUTEL 用户通过 MoE 层 API 自动启用自适应流水线，无需手动管理 stream。预构建字典在训练开始前通过少量 profiling（每 key 最多 (log_{1.5}⌈W/E⌉ + 2) × 4 × 2 trials）完成。

涉及论文标题：
- Tutel Adaptive Mixture-of-Experts at Scale
