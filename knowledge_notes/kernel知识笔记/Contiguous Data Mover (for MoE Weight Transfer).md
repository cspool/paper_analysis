## Contiguous Data Mover (for MoE Weight Transfer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Contiguous Data Mover 是 MoE-Lens 提出的专用 CPU→GPU 权重传输模块，以独立线程运行（C++ PyTorch extension），负责在 CPU-GPU 混合 MoE 推理系统中高效搬运模型权重。其设计动机是解决 head-of-line blocking：如果将所有权重传输 API 调用嵌入执行流水线中，大批量 weight transfer 会阻塞延迟敏感的 compute transfer（如 PyTorch 操作、attention 同步数据），导致 GPU stall。Contiguous Data Mover 将 weight transfer 从执行流水线中解耦——执行引擎以 layer-wise granularity 推送传输请求，data mover 内部以 fine-grained 小 packet（100MB）分批执行传输，避免与其它 CPU-GPU transfer 竞争。100MB packet size 是 trade-off：足够大以最大化 PCIe bandwidth utilization，又足够小以避免 head-of-line blocking 和过长 latency。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Contiguous Data Mover 的调度伪代码：

```
// Data Mover 线程（独立于 compute 线程）
shared queue<TransferRequest> request_queue;  // layer-wise requests
const PACKET_SIZE = 100 * 1024 * 1024;  // 100MB

function data_mover_thread():
    while not done:
        // 1. 从执行引擎接收 layer-wise 传输请求
        Request req = request_queue.pop()
        
        // 2. 将 layer weights 分区为 100MB packets
        packets = partition(req.weight_data, PACKET_SIZE)
        
        // 3. 逐个 packet 异步传输
        for packet in packets:
            // cudaMemcpyAsync 到 GPU Weight Buffer
            event = cudaMemcpyAsync(
                dst: gpu_weight_buffer[req.layer_slot],
                src: cpu_pinned_weight[req.layer_id][packet.offset],
                size: packet.size,
                stream: mover_stream  // 独立 CUDA stream
            )
            record_event(event)
    
    // 4. 在每个 stage boundary 同步（不等待每个 phase）
    synchronize_stream(mover_stream)
```

关键调度特性：
- **独立 CUDA stream**：data mover 使用独立 stream，不与 compute stream 的 PyTorch operation 竞争。
- **100MB packet 粒度**：在 bandwidth utilization 和 interference minimization 之间折中。论文实证 100MB 为最优值。
- **Stage-boundary synchronization**：data mover 仅在 stage boundary（而非 phase boundary）同步——VSLPipe 中每个 stage 有 CPU phase + GPU phase，data mover 在整个 stage 期间异步运行，仅在 stage 结束时同步确保下一 stage 的 weights 就绪。
- **Weight Buffer 管理**：GPU 端 Weight Buffer 大小为 $2 \times$ per-layer weight size（通常仅为 model size 的 2-3%），实现双缓冲——一块用于当前 computation，另一块用于 prefetch 下一 layer/group 的 weights。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **实现**：C++ PyTorch extension，约数百行 C++ 代码。使用 CUDA Runtime API 的 `cudaMemcpyAsync` 和 `cudaStream_t`。
- **与执行引擎的交互**：执行引擎（VSLPipe）在每个 stage 开始时，通过 request_queue 推送下一 stage 所需的 layer weights 和 expert weights 的传输请求。Data mover 内部调度所有 packets，在 stage 结束前完成全部传输。
- **与 CPU Attention 的带宽竞争**：当 KV cache 较大（210GB）且 generation length 较大（256 tokens）时，CPU attention 需要扫描大量 KV cache blocks，与 data mover 竞争 CPU memory bandwidth，导致 weight transfer 时间从 ~5s 增加到 ~6s（论文 §8.2）。这是 MoE-Lens 当前的主要性能瓶颈之一。
- **Weight 存储**：所有权重存储在 pinned CPU memory 中，分为 layer-wise weights（attention projection matrices + normalization）和 expert weights（MoE layer 的 expert-specific 参数）。Transfer 以 layer-wise granularity 请求，内部分区为 packets。

涉及论文标题：
- MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints
