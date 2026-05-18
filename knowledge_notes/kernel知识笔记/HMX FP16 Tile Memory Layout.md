## HMX FP16 Tile Memory Layout

术语是什么？通过联网搜索让回答具体和精准。

HMX FP16 Tile Memory Layout是Hexagon NPU的HMX矩阵单元对FP16操作数要求的内存排列格式。每个tile为32×32 FP16 (2KiB)。Layout分两级：(1) Level-1 tile内部：每两行做permutation——[a0..a31; b0..b31]排列为[a0,b0,a1,b1,...,a31,b31]，等价于转置后的2×32子矩阵；(2) Level-2 tile间：weight tiles按column-major排列（HMX执行tile-level inner product，weight column-major匹配accumulation order）。Activation tile为row-major排列。所有HMX指令只能读取TCM内数据，因此runtime需将数据按此layout放入TCM。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// GEMM C[M,N] = A[M,K] × B[K,N] with HMX tiles
// 所有矩阵reshape为 [dim1/32, dim2/32, 32, 32] tiles
// 每tile内部: 2-row permutation memory order

for n in 0..N/32:        // Column tiles of B and C
    for m in 0..M/32:    // Row tiles of A and C
        C[m][n] = 0
        for k in 0..K/32:
            C[m][n] += hmx_mma(A[m][k], B[k][n])
            // hmx_mma: load A tile & B tile from TCM → 32×32×32 inner product
            // internally FP32 accumulate → output FP16 tile
        // Post: per-column scale + bias on output tile (HMX native)
```

Decode阶段观察：activation A shape [B, hidden_dim]，B通常=1。A reshape为[B/32, hidden_dim/32, 32, 32]，B=1时仅1个activation tile行有有效数据，31行空闲——这是HMX decode underutilization的核心。TTS增加B（如B=8），填充更多tile行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

权重tile layout转换离线完成（模型转换脚本，一次转换永久存储）。Activation runtime转换：HVX cross-lane shuffle指令对每两行做permutation，可与dequant重叠。不同量化格式(IQ4_NL等)通过LUT vlut16直接映射到HMX-compatible FP16。开源实现见llama.cpp-npu的GGUF conversion和htp-ops-lib的kernel代码。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

