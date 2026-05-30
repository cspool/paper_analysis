## Payload-Efficient Communication / In-Place Padding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Payload-Efficient Communication 是 FlashMoE 消除 MoE dispatch 网络带宽浪费的技术。传统 AlltoAll 的对称性约束迫使零填充 token 参与通信和计算。FlashMoE 用 In-Place Padding（本地对齐 tile size bM=128）后用 NVSHMEM one-sided put 仅发送有效 token，消除 null padding 的网络传输和后续无效计算。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Traditional: AlltoAll(padded) → null tokens 占带宽!
// FlashMoE:
for each expert e on remote GPU j:
    if actual_tokens > 0:
        nvshmem_put(
            &L[j, DISPATCH, incoming, e, 0, :],   // remote
            &L[0, DISPATCH, outgoing, e, 0, :],   // local
            actual_tokens × H × sizeof(float),     // 仅有效数据
            peer = j
        )
    // In-place padding (对齐 bM=128) 仅在本地 buffer, 不传网络
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 前提是 one-sided (R)DMA——每 GPU pair 独立决定传输量，不依赖 collective symmetry
- Capacity upscaling（对齐 bM=128）确保 Processor coalesced read

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
