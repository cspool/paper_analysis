## PIM Capacity Wall（PIM容量墙）

术语是什么？通过联网搜索让回答具体和精准。
PIM Capacity Wall是AQPIM论文识别的PIM架构关键瓶颈：虽然PIM通过近存计算解决了memory bandwidth瓶颈，但LLM推理中的KV cache内存footprint（长context场景可达数百GB）频繁超过PIM的on-chip memory capacity。由于bank-level PIM implementation本身的density cost（bank旁添加PE、wiring、control logic等），PIM device的实际可用内存密度低于同等制程的纯DRAM。例如AttAcc!甚至需要40个HBM-PIM devices才能容纳short context的KV cache。Scaling PIM devices数量经济上不可行，而offloading via PCIe (90~98.5% of decoding latency) 和 sparse attention (scattered access与PIM data locality冲突) 等mitigation方法在PIM架构下均失效。该墙不仅是容量问题，更是algorithm-hardware mismatch：传统quantization添加ALU导致126% area overhead进一步压缩内存密度，形成"要压缩需加硬件，加硬件降低密度"的困境。

从硬件架构角度拆解术语：
PIM Capacity Wall的影响链：
```
Long-context LLM (128K+ tokens, KV cache ~100s GB)
        │
        ▼
┌───────────────────────────────────────────┐
│  HBM-PIM Device: 16GB per stack            │
│  - PE overhead: ~50% area → reduced density│
│  - 4 stacks → 64GB total KV capacity       │
│  - For 128K context, KV cache > 64GB       │
│  → OOM or offloading required              │
└───────────────────────────────────────────┘
        │
        ├── Solution A: Scale PIM devices (40 HBMs +)
        │   → Economically unviable, power/area explosion
        │
        ├── Solution B: PCIe offloading
        │   → GPU-CPU communication = 90~98.5% of decode latency
        │
        ├── Solution C: Add quantization ALU
        │   → FP16+INT32 cost 126% area vs FP16-only [Lee et al.]
        │   → Further reduces already scarce memory density
        │
        └── Solution D (AQPIM): In-memory PQ clustering
            → 80%+ compression, 0.43% area overhead, 3.4× speedup
```

术语一般如何实现？如何使用？
Capacity wall影响评估方法：(1) 计算模型KV cache per-layer size = 2 × N_layers × N_heads × d_head × N_tokens × 2bytes (BF16)；(2) 对比HBM-PIM effective capacity（扣除PE/control overhead后的可用存储）；(3) 判断是否需要compression/offloading。AQPIM的破解策略：(a) 算法层：online PQ compression (80%+ reduction) → KV cache footprint降至可fit HBM-PIM；(b) 架构层：仅添加0.43% intra-row indirection硬件（vs 126% for INT32 ALU），保持memory density；(c) 系统层：PIM内直接compute-on-compressed-data，消除offloading penalty。评估metrics: memory reduction ratio vs accuracy trade-off (LongBench, Fig.10)，KV cache memory footprint (Fig.13: gpu+cpu 120GB → aqpim ~0GB KV cache)。

涉及论文标题：
- AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

