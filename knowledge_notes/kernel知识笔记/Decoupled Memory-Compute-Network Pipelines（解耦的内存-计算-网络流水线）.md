## Decoupled Memory-Compute-Network Pipelines（解耦的内存-计算-网络流水线）

术语是什么？通过联网搜索让回答具体和精准。

Decoupled Memory-Compute-Network Pipelines是RPU论文提出的一种微架构设计模式，将传统GPU中耦合的内存访问、计算和网络通信三条数据路径分离为独立、可异步前进的硬件pipeline。每pipeline有独立的DMA engine（Memory DMA、Compute DMA、Network DMA）和专用SRAM buffer。关键创新在于：pipeline之间通过Pipeline Arbiter在buffer entry粒度做data-driven同步（非global barrier），使得任一pipeline可以在其他pipeline stall时继续前进——例如memory pipeline在compute等待network activation时继续预取weights到on-chip buffer。这种设计解决了低batch LLM decode中因kernel launch、synchronization和small distributed matrices导致的memory bandwidth underutilization问题（H100在decode时仅达32.2% BW utilization）。

从kernel调度角度拆解：

Decoupled pipelines的kernel执行伪代码（以Llama3-8B, BS=1, 64-CU RPU为例）：

```
// === Decoupled Pipeline Execution for One Transformer Layer ===
// Each core autonomously executes its instruction stream
// Three pipelines operate concurrently with buffer-level synchronization

// --- Memory Pipeline (never stalls on compute/network) ---
memory_pipeline:
    for each weight_tile in layer.weights:
        // Prefetch weights from HBM-CO to memory buffer
        DMA_HBM_to_MemBuf(weight_addr, mem_buf_entry)
        mem_buf[mem_buf_entry].valid_count = consumercount  // set by Pipeline Arbiter
    for each kv_cache_tile:
        DMA_HBM_to_MemBuf(kv_addr, mem_buf_entry)
        mem_buf[mem_buf_entry].valid_count = consumercount

// --- Compute Pipeline ---
compute_pipeline:
    for each stripe in VMM:
        // Check-valid: stall if activation not yet in network buffer
        network_buf[act_entry].check_valid()
        activation = network_buf[act_entry].read()
        register_file.write(activation)  // 64 BF16

        for each tile_col in stripe:
            for each tile_row:
                // Check-valid: stall if weight tile not yet decoded
                mem_buf[weight_entry].check_valid()
                weight = mem_buf[weight_entry].read()
                decoded_weight = stream_decoder.decode(weight)  // on-the-fly dequant
                TMAC.compute(activation[col], decoded_weight)

            tree_sum_reduce(TMAC.accumulators[col])

        // Write output to local register
        local_reg.write(output)

// --- Network Pipeline ---
network_pipeline:
    // Receive activation fragments from upstream cores
    for each activation_fragment:
        DMA_Recv(upstream_core, net_buf_entry)
        net_buf[net_buf_entry].valid_count = 2  // consumed by: compute + forward
    // Forward activation fragments to downstream cores
    for each forward_fragment:
        net_buf[fwd_entry].check_valid()  // wait for compute to produce
        DMA_Send(downstream_core, fwd_entry)
```

解耦带来的关键执行行为（Fig.8 simulation trace）：
- **BS=1, wQKV阶段**: network latency-limited（activation broadcast across CUs），memory pipeline继续预取weights（提前~80KB ahead of compute）
- **BS=1, QK^T阶段**: 跨CU gather Q/K/V shards + distributed max/reduction → compute stalls → memory pipeline prefetches KV$ entries
- **BS=32, wUp/wGate阶段**: compute-bound（weight processing ~4× longer than memory read）→ memory pipeline prefetches deep ahead (~6MB/CU) → buffer absorbs phase imbalance
- **无decoupling对比**: 全局barrier同步会使用memory/compute/network互相等待，导致累计stall延迟增加至1.6×

术语一般如何实现？如何使用？

在RPU中，decoupled pipelines通过以下机制实现：1) 每个pipeline有独立DMA engine和专用address space；2) SRAM buffer entry粒度的valid counter（Pipeline Arbiter）；3) NUMA at all scales（每core独立NUMA domain，无shared memory，跨domain通信显式由software-programmable DMA管理）；4) RPU ISA指令embed Pipeline Arbiter flags（check-valid/valid-count set）。这种设计与GPU的host-driven offload + global barrier模式根本不同——GPU需要等所有thread blocks完成kernel后才释放barrier，而RPU的pipeline在数据ready时立即前进。

涉及论文标题：
- RPU - A Reasoning Processing Unit

