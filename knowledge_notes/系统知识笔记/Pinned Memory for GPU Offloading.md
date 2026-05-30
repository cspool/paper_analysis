## Pinned Memory for GPU Offloading

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Pinned Memory（页锁定内存，也称 page-locked memory）是 CUDA 编程中的一种 host 内存分配类型。与普通可分页（pageable）host 内存不同，pinned memory 被操作系统锁定在物理内存中不可被换出（swap out），使 GPU 的 DMA 引擎可直接通过 PCIe 访问而无需 CPU 介入逐页拷贝。在 PyTorch 中通过 `tensor.pin_memory()` 创建。对于 offloading 场景，pinned memory 的 host-to-device 带宽可达 PCIe 理论带宽的 80-90%，而 pageable memory 仅约 50-60%（因需 staging buffer）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Mixtral offloading 中 pinned memory 的利用方式：

```
# 初始化: 为所有 expert 分配 pinned memory 缓冲区
for layer in 0..31:
    for expert in 0..7:
        # contiguous pinned allocation (一次 pin 而非逐 expert)
        host_expert_buf[layer][expert] = torch.empty(
            expert_size, dtype=torch.int8, pin_memory=True
        )
        # 量化后的 expert 权重直接写入 pinned buffer
        host_expert_buf[layer][expert].copy_(quantized_expert_weights)

# 推理时: 直接从 pinned memory 异步拷贝
def load_expert(layer, expert_id, gpu_buf):
    # cudaMemcpyAsync: CPU 不参与数据搬运
    # GPU DMA engine 通过 PCIe 直接从 pinned buffer 读取
    stream = get_io_stream()
    with torch.cuda.stream(stream):
        gpu_buf.copy_(host_expert_buf[layer][expert_id], non_blocking=True)
    return stream  # 调用方可等待或重叠其他计算

# Pageable vs Pinned 对比:
# Pageable: CPU 先拷贝到 staging buffer → DMA 从 staging buffer 读 (2次拷贝)
# Pinned:   DMA 直接从 pinned buffer 读 (1次拷贝, 无 CPU 中转)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- PyTorch API: `t.tensor.pin_memory()` 或 `tensor.to('cuda', non_blocking=True)` 时自动使用
- 限制：pinned memory 过多会抢占 OS 可分页物理内存，可能影响系统稳定性——论文通过仅 pin expert 部分（attention 常驻 GPU）平衡此问题
- 对于 16GB host RAM 机器（如 Colab free-tier），需仔细管理 pinned memory 总量以免 OOM
- 论文的 contiguous expert buffer 设计允许单次 `cudaMemcpyAsync` 传输完整 expert（避免多次小传输的 PCIe 事务开销）
- 与 `non_blocking=True` 配合时，CPU 继续执行而 GPU DMA 在后台搬运，是实现 overlapping 的关键前提

涉及论文标题：
- Fast Inference of Mixture-of-Experts Language Models with Offloading
