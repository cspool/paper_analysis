## List Buffer Data Structure for Calibration Data（校准数据的List Buffer数据结构）

术语是什么？
List Buffer 是 QMoE 为高效管理海量校准样本而设计的数据结构——一个大连续 CPU buffer 存储所有 calibration sample 的 token hidden states，配合 delimiter indices 标记各 sample 边界。List buffer 解决了两个核心操作：(1) Per-sample access for dense layers——按 delimiter 快速定位和提取单个 sample（dense layer 需 batch-by-batch 处理）；(2) Fully-vectorized querying of expert tokens——按 expert assignment 抽取所有分配给某 expert 的 token（MoE sparse part 需要）。Naive 替代方案（逐个样本迭代 + mask-based token selection）在处理 >100K samples 时 unusably slow。

从系统架构角度拆解术语：
```
# List Buffer 数据结构
B = [hidden_states_all_samples]  # 连续内存块
delimiters = [d0, d1, d2, ...]   # 各 sample 的起始/结束索引

# 示例：3 samples, 各含 128, 256, 192 tokens
B = [tok1_s1, tok2_s1, ..., tok128_s1, | tok1_s2, ..., tok256_s2, | tok1_s3, ..., tok192_s3]
delimiters = [0, 128, 384, 576]

# 操作1: Per-sample access (dense layer)
for sample_i in range(num_samples):
    start = delimiters[sample_i]
    end = delimiters[sample_i+1]
    X = B[start:end]  # O(1) slice on contiguous memory
    Y = DenseLayers(X)
    B[start:end] = Y  # 原地更新 (overwrite)

# 操作2: Expert token querying (sparse part)
# 对 expert E，提取所有分配给它的 token
# assignments = [token_i → expert_E_i] (sparse tensor)
for expert E in layer:
    # 向量化 index extraction (CPU-side, numpy/torch)
    indices_E = where(assignments == E)  # 各 sample 中的局部索引
    # translate to global B indices using delimiters
    global_indices = local_to_global(indices_E, delimiters)
    X_E = B[global_indices]  # 单个 contiguous read (masked gather)
    # 发送到 GPU 进行压缩
```

对比 naive masking：
```
# Naive: 对每个 sample 迭代 + mask
for each sample:
    for each expert:
        mask = (assignments == expert)
        X_E.append(X[mask])  # 逐 sample 逐 expert masking → O(S×E) GPU kernel launches
# List Buffer: 一次全局 index query → O(1) bulk read
```

术语一般如何实现？如何使用？
- 实现：NumPy/Torch CPU tensor + python list of delimiter indices
- 优势：(a) O(1) per-sample 访问；(b) 向量化 expert token 查询；(c) 原地更新避免内存膨胀
- 与 activation offloading 配合：B 存于 CPU RAM，每次仅小块数据经 PCIe 传输到 GPU
- 通用性：可推广到任何需要管理大量样本级中间结果的 ML pipeline

涉及论文标题：
- QMoE Sub-1-Bit Compression of Trillion-Parameter Models
