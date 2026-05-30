## Expert Buffering（专家缓冲 / GPU-CPU两级专家缓存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Buffering 是 Huang et al. (NeurIPS 2024) 在 "Toward Efficient Inference for Mixture of Experts" 中提出的 MoE 推理 GPU 显存优化机制。核心思想是利用 expert 激活的 **temporal locality**（时序局部性）：推理过程中仅将频繁被激活的"热"expert 参数缓存在 GPU 显存中，其余 expert 参数存放在 CPU 内存中。当冷 expert 被激活时，通过 PCIe 从 CPU 异步拷贝参数到 GPU（cudaMemcpyAsync），与 token 的 all-to-all 传输重叠。

与普通 LRU cache 不同，Expert Buffering 使用 **LIFO (Last In, First Out) eviction policy**，适配 MoE Transformer 中 experts 按 ID 顺序串行执行的特性：假设 GPU 缓存 2/4 experts，batch 激活 experts {1,2,3}，执行顺序为 1→2（evict 2）→3。LIFO 选择 evict expert 2（最近使用的）而保留 expert 1（复用距离更短）。缓存 miss rate 接近理论最优 Belady's MIN。

微架构：每个 GPU 独立管理自己的 expert cache（仅缓存分配到自己 GPU 的 experts）。CPU memory 中保留所有 experts 的完整参数作为后备。每次 MoE forward 前检查所需 experts 是否在 cache 中。

从系统架构角度拆解术语：

```
[Expert Buffering Request-Level Flow]

Pretime: 所有 expert 参数初始化于 CPU memory
         GPU cache (per GPU): slots for K experts (e.g., K=10/GPU)

Per-batch Forward:
  For each GPU i (hosting experts assigned by EP):
    │
    ├─ Step 1: Receive token assignments from gate
    │    active_experts = {e | e has tokens in this batch}
    │
    ├─ Step 2: Cache lookup + prefetch
    │    for e in active_experts:
    │      if e not in GPU_cache[i]:
    │        launch cudaMemcpyAsync(e_params, CPU→GPU, stream=copy_stream)
    │        // copy_stream 与通信 NCCL stream 并发
    │      else:
    │        // cache hit, 直接使用
    │
    ├─ Step 3: Token all-to-all (与 Step 2 重叠)
    │    tokens arrive on GPU i via NCCL all-to-all
    │
    ├─ Step 4: Expert Execution (sequential by expert ID)
    │    for e in active_experts (sorted by ID):
    │      await e_params ready (sync copy_stream if needed)
    │      out_e = FFN_e(tokens_e)
    │
    ├─ Step 5: Cache Eviction (LIFO)
    │    if GPU_cache[i].full():
    │      for e in GPU_cache[i]:
    │        if e not active in current batch:  # temporal locality
    │          candidates.append(e)
    │      evict = candidates[-1]               # LIFO: 最后使用的 evict
    │      copy evicted params back to CPU if modified (usually not needed)
    │      GPU_cache[i].insert(e_new, evict_slot)
    │
    └─ Step 6: Output all-to-all (send expert outputs back)
```

LIFO 正确性说明：
```
GPU hosts experts {1,2,3,4}, cache size K=2, batch activates {1,2,3}
执行顺序: FFN_1 → FFN_2 → FFN_3 (按 ID 递增)
LIFO eviction on expert 2 → cache 保留 {1, 3}
  下次 batch 若激活 {1,4}: expert 1 hit, expert 3 可被 evict
FIFO eviction on expert 1 → cache 保留 {2, 3}
  下次 batch 若激活 {1,4}: expert 1 miss（刚刚被 evict）, 更差
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现在 Fairseq MoE Transformer 基础上，Python + CUDA：(1) 每个 GPU 维护 `expert_cache: dict[int, ExpertParams]` 和 `cache_capacity: int`；(2) `cudaMemcpyAsync` 在独立 CUDA stream 上执行，与 NCCL all-to-all 的 default stream 并发；(3) LIFO eviction 通过维护访问时间戳实现（每次 expert 执行后更新 timestamp）。论文开源于 https://github.com/hyhuang00/moe_inference。

性能特征：(1) 减少 static GPU memory 1.47×（~2.25GB on V100 32GB）；(2) cache miss 仅在 cache < 5 experts/GPU 时显著；(3) PCIe 带宽（12GB/s）是瓶颈，新技术如 Grace Hopper 可缓解；(4) 与传统的 offloading（ZeRO-Offload 等）正交，可同时使用。

涉及论文标题：
- Toward Efficient Inference for Mixture of Experts
- Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference
