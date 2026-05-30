## Lazy Weight Fetching for Trillion-Parameter Models（万亿参数模型的延迟权重获取）

术语是什么？
Lazy Weight Fetching 是 QMoE 在处理 1.6T 参数 SwitchTransformer-c2048 时开发的内存管理策略——模型原始权重（3.2TB bfloat16）甚至无法完全放入系统 RAM（通常 ≤ 2TB）。策略：(1) 权重原始存储于磁盘（SSD/HDD）；(2) 仅当某 expert 需要被压缩时才从磁盘加载其权重到内存（CPU RAM）；(3) 压缩完成后立即将压缩版本写回磁盘并释放原始权重所占内存；(4) 在整个压缩流水线中每个 expert 的原始权重恰好加载一次（"lazy" = 按需、不预加载、用完即弃）。结合 activation offloading 和 expert grouping，使单台配备 A6000 + hundreds GBs RAM + 数 TB 磁盘的服务器即可处理万亿参数模型压缩。

从系统架构角度拆解术语：
```
# 传统方法（不可行）:
全部模型权重 (3.2TB) → 加载到 CPU RAM (需 ≥3.2TB RAM)
  → 逐层加载到 GPU → 压缩 → 写回

# QMoE Lazy Weight Fetching:
磁盘存储: 原始 bfloat16 checkpoint (3.2TB, sharded)
CPU RAM: List Buffer (calibration activations) + current expert group weights
GPU: Current expert group weights + activations + compression workspace

压缩过程（per expert group）:
    1. 从磁盘读取 expert group E 的原始权重 (~数 GB) → CPU RAM
    2. 从 CPU RAM 加载 → GPU
    3. 执行 Batched GPTQ（使用 CPU buffer 中对应 X_E）
    4. 压缩后的 E' 写回磁盘
    5. 释放 GPU + CPU 中的原始 W_E
    # 每 expert 恰好读 1 次磁盘，写 1 次压缩版本
```

磁盘 I/O 分析：c2048 的 1.6T 参数需约 3.2TB bfloat16 存储；以顺序读 500MB/s (HDD) 计算，纯加载原始模型需约 6400s ≈ 1.8h。QMoE 实际总压缩时间 ~16h，其中加载原始模型 ~5h（论文指出磁盘较慢），实际计算为主。

术语一般如何实现？如何使用？
- 实现：Python mmap 或标准文件 I/O，配合 PyTorch `torch.load()` 按 shard 加载
- 对磁盘性能敏感——建议 SSD（NVMe 更好）以缩短 I/O 主导时间
- 适用：任何模型太大放不进 RAM 的场景下的 data-dependent 压缩/分析/评估
- 通用性：可直接与 activation offloading 和 expert grouping 联合使用

涉及论文标题：
- QMoE Sub-1-Bit Compression of Trillion-Parameter Models
