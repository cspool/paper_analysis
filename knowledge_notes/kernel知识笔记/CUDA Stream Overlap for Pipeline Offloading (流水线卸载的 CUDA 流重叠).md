## CUDA Stream Overlap for Pipeline Offloading (流水线卸载的 CUDA 流重叠)

术语解释
在使用 offloading 技术的流水线并行训练中，通过维护独立的 CUDA stream 分别执行计算（comp_stream）和数据移动（load_stream），利用 GPU 的并发执行能力使计算与 PCIe 数据搬移并行进行，以隐藏数据加载延迟。

术语是什么？
APTMoE 采用双流架构：
- **comp_stream**（计算流）：执行 MHA、gate operation、expert FFN 的 forward/backward 计算
- **load_stream**（加载流）：执行 host→device 的 cudaMemcpyAsync（加载下一层/下一个 stage 的参数）

两个 stream 通过 `torch.cuda.Event` 建立依赖关系：load_stream 加载完某个 block 后 record event，comp_stream 执行该 block 的计算前 wait 该 event。这确保了在优化 overlap 的同时不违反数据依赖。

从kernel调度角度拆解术语。
```
# GPU 时间线（以 2 个 micro-batch 的 forward 为例）
# load_stream:  [L_S1_MHA][L_S1_Gate][L_S1_E1][L_S2_MHA]...
# comp_stream:            [C_S1_MHA][C_S1_Gate][C_S1_E1]...
#                <------- overlap region ------->

# 伪代码
for micro_batch in micro_batches:
    # load_stream: 异步预取
    with torch.cuda.stream(load_stream):
        for block in preload_blocks:
            data = load_from_host(block)         # cudaMemcpyAsync H→D
            event = torch.cuda.Event()
            event.record(load_stream)
            ready_events[block] = event
    
    # comp_stream: 等待数据就绪后计算
    with torch.cuda.stream(comp_stream):
        for block in stage_blocks:
            if block in ready_events:
                comp_stream.wait_event(ready_events[block])
            execute_forward(block)               # MHA / gate / expert FFN

# 关键性能指标：
# - 加载时间完全隐藏率 = (overlap_time / total_load_time) × 100%
# - APTMoE 的三阶段加载使该比例显著高于 Mobius（Mobius 在 MoE 场景下加载阻塞计算）
```

重叠效率取决于 data-to-computation ratio：MoE 的数据量远大于 dense model（多个 expert 参数），因此重叠更困难。APTMoE 通过选择性加载（仅高热度 expert → GPU）降低 data 量，使重叠更可行。

术语一般如何实现？如何使用？
- PyTorch `torch.cuda.Stream()` 创建独立流
- `torch.cuda.current_stream()` 获取/设置当前流
- CPU 端使用 `psutil.Process().cpu_affinity()` 将进程绑定到指定 CPU 核心，避免 compute 和 I/O 线程竞争
- 注意：同一方向的 cudaMemcpy 操作即使在不同 stream 中也会在 GPU 端序列化（硬件限制）→ 因此 APTMoE 需要 demand-priority scheduling 来协调

涉及论文标题：
- APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes

---
