## CUDA Stream Async Expert Overwrite Loading (HarMoEny)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

CUDA Stream Async Expert Overwrite Loading 是 HarMoEny 的异步 expert 预取机制（Section 4.3），通过独立 CUDA stream 从 system memory（CPU RAM）异步传输 expert 权重到 GPU memory，直接覆写已完成的 expert 所占内存。核心洞察：expert 权重在推理中不变（read-only），无需先写回 system memory。Overwrite-based loading 比传统 "write-back + load" 快 5.5×（11ms vs 2ms on V100, 18MB Switch128 experts）。

从 kernel 调度角度拆解术语：

```
# HarMoEny 双 CUDA stream 模型 (Algorithm 1, Step 5)
# compute_stream: 执行 expert FFN GeMM
# load_stream: 异步 system→GPU expert weight transfer

gpu_expert_slot = [slot0, slot1]  # ping-pong 双 slot

for idx, expert_e in enumerate(assigned_experts):
    curr_slot = gpu_expert_slot[idx % 2]
    next_slot = gpu_expert_slot[(idx + 1) % 2]

    if expert_e not in GPU_memory:
        # load_stream: 异步预取下一 expert 权重
        with torch.cuda.stream(load_stream):
            # 直接覆写 next_slot (已完成的 expert)
            # 无需 write-back → 5.5× faster
            next_slot.copy_(system_mem[expert_e.offset], non_blocking=True)
            # Transfer: 18MB / 32GB/s ≈ 0.56ms theoretical, ~2ms actual (V100)

    with torch.cuda.stream(compute_stream):
        # 当前 expert 计算 (与 load_stream 的传输重叠)
        output += gate_weights[e] * expert_ffn(expert_e, tokens_e)
        # Expert FFN: 2× GeMM (W1[x] @ x → activation → W2[x] @ result)

    load_stream.wait_stream(compute_stream)  # 确保 slot 读写无冲突

torch.cuda.synchronize()
```

Annotations:
- **load_stream**: 独立 stream 执行 cudaMemcpyAsync (system→GPU)
- **compute_stream**: 主 stream 执行 expert FFN GeMM
- **双 slot ping-pong**: 仅需 2 个 expert slot（compute 用 1 个 + prefetch 用 1 个）
- **Overlap condition**: computation_time > transfer_time → 传输完全隐藏
- **5.5× origin**: 传统 offloading 需先 GPU→CPU write-back (9ms) + CPU→GPU load (2ms) = 11ms; overwrite 仅需 load (2ms)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HarMoEny 在 PyTorch 中使用 `torch.cuda.Stream()` 创建独立 load stream + `.copy_(non_blocking=True)` 异步传输。需至少 2 个 expert slot fit in GPU memory（大多数 MoE serving 已满足）。由 token threshold q 保证 computation > transfer（防止传输无法隐藏的场景）。HarMoEny 1115 行 PyTorch 代码中实现，开源：https://github.com/sacs-epfl/HarMoEny。

涉及论文标题：
- HarMoEny: Efficient Multi-GPU Inference of MoE Models
