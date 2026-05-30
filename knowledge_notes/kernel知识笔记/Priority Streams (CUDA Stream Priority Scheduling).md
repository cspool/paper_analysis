## Priority Streams (CUDA Stream Priority Scheduling)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Priority Streams 是 NVIDIA CUDA 提供的 intra-process 并发机制，允许同一 OS 进程内的不同 CUDA stream 被赋予三级优先级（-2 最低 / -1 / 0 最高）。当多个 stream 都有 pending thread blocks 时，GPU thread block scheduler 会优先从高优先级 stream 取 block 进行调度。但关键限制是：**priority streams 不抢占已在 SM 上执行的 block**——高优先级 kernel 到达后必须等待低优先级 kernel 的已执行 blocks 自然完成（SM draining），仅能插队调度后续 blocks。因此 priority streams 适用于短 kernel 之间的调度优化，对 long-running kernel（>1ms）的效果受限于 compounded delay。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Priority Streams 的调度流程（并发 training + inference workload 场景）：

```
CPU Host:
  Process:
    cudaStreamCreateWithPriority(&stream_high, 0)    // inference, highest priority
    cudaStreamCreateWithPriority(&stream_low, -2)    // training, lowest priority
    // 交替发射 kernel 序列
    for each inference_request:
        cudaLaunchKernel(inf_kernel, stream_high)    // 高优先级
    for each training_batch:
        cudaLaunchKernel(train_kernel, stream_low)   // 低优先级

GPU Thread Block Scheduler:
  while True:
    // Step 1: 检查高优先级 stream 是否有 pending blocks
    if stream_high has unscheduled blocks:
        schedule one block from stream_high via most-room policy
    // Step 2: 高优先级为空才考虑低优先级
    elif stream_low has unscheduled blocks:
        schedule one block from stream_low via most-room policy
    // Step 3: 不等已执行 blocks（无法抢占）
    // -> 高优先级 kernel 到达时，低优先级已执行 blocks 继续运行直到完成
```

Compounded Delay 发生场景：
```
Time ------------------------------------------------->
Training:  [Block_T0][Block_T1][Block_T2]...[Block_Tn]  ← 低优先级，占据 SM
Inference:              [Kernel_I0 arrives]              ← 高优先级，需等待
                        |<--- compounded delay --->|
                        Blocks T0-Tn 需全部完成后
                        Kernel_I0 才能被调度
```

此论文在 NVIDIA GeForce RTX 3090 (Ampere, 82 SMs) 上实测：ResNet-50、VGG-19、DenseNet-201 等模型的 inference turnarround time 增加 1.75X-4X（vs baseline isolation），且 variance 增大——priority streams 比无优先级的 MPS 表现并无优势。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

CUDA API: `cudaStreamCreateWithPriority(&stream, flags, priority)`，priority 取值 -2/-1/0（更负=更低优先级）。优先级仅在同一 CUDA context 内有效——跨进程不适用。流间异步调度由 GPU hardware thread block scheduler 自动处理，无用户态 API 可干预调度决策。适用场景：(i) 单进程内 latency-sensitive（高优先级）+ throughput-oriented（低优先级）workload 混合；(ii) GPU-internal 并发 pipeline（如 prefill/decode 混合）。局限性：(a) 无法抢占已执行 blocks → long-running kernel 场景效果差；(b) warp scheduler 可能不配合优先级（用 greedy-then-oldest policy 可能 de-prioritize 高优先级 warp）；(c) blocks colocation 导致 L1/functional unit contention 削弱优先级效果。

涉及论文标题：
- Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads
