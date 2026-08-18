## Asymmetric Numeral Systems（ANS / rANS / tANS / FSE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ANS 是 Jarek Duda 提出的熵编码族（arXiv:1311.2540）：用单个自然数状态 x 编码整个符号流，兼有 Huffman 的编码速度与算术编码的压缩率（接近 Shannon 熵）。rANS（range variant）：编码 s 时 x' = ⌊x/fs⌋·R + (x mod fs) + cs（fs 为归一化频率、cs 为累积频率、R=2^b 为精度基数）；解码：σ = T[x mod R]（查表取符号），x = fs·⌊x/R⌋ + (x mod R) − cs。tANS/FSE 把全部转移预计算为查表自动机。整个编码行为由一张几 KB 的"熵编码自动机表"（256 符号）确定；状态可任意初始化而不损压缩率，因此天然支持多条独立子流并行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文用 rANS 对 LLM 权重做 tile 级无损压缩：
```
# 离线编码（每投影矩阵）
freq = histogram(layer_weights); fs, cs = normalize(freq, R=2^12)  # 层共享 codebook
for tile in split(W, tile_shape):            # 与 GEMM tile 几何对齐
    x = x0[tile]                             # 独立初始状态 → 自包含 substream
    for s in reversed(tile):                 # rANS 逆序编码
        x = (x // fs[s]) * R + (x % fs[s]) + cs[s]
    stream[tile] = x; offset[tile] = pos     # 4B/条 offset 表 → tile 随机访问
```
Annotations：任意初始状态不损压缩率是 ANS 相对算术编码的关键性质——每个 tile 是独立 substream，可跳过前面 tile 直接解码（tile 级随机访问）。选型依据（论文 Table I）：LZ77/Zstd 字典指针链不可随机访问；算术编码位串行仅几 GB/s；Huffman 整数码长离熵界 5–10%；rANS/tANS/FSE 熵效率 >99%、字节级流式、可并行——是唯一同时满足"近 Shannon 码率 + tile 随机访问 + GPU 并行解码"的编解码族。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
工业实现：Zstd 的 FSE（tANS）、Apple LZFSE、Fabian Giesen 的 ryg_rans（byte-aligned 参考实现）、Meta DietGPU（GPU rANS，A100 解码 250–410 GB/s）。解码实现要点：符号查表（per-slot sym 表或 (fs,cs) 数组）+ 重归一化（状态低于阈值时读入 32/64-bit 块）。使用：近熵压缩 + 高吞吐场景——LLM 权重、HPC 数据搬运、GPU 集体通信压缩；本论文将其升级为 GPU 推理执行原语（解码直接写 shared memory 供 tensor core 消费）。

ENEC 补充视角（Ascend NPU 侧）：ENEC 论文把 ANS 移植到 Ascend 910B2 实测吞吐惨淡（Figure 1b），与 LZ77 类似，因此没有采用 ANS/变长熵编码，而是转向"块式定长编码 + 只压指数"路线。原因：ANS 的解码依赖符号查表（T[x mod R]）、重归一化分支与变长状态管理——需要条件分支、scatter/gather 和不规则变长访存，而 Ascend AIV 是无条件分支的 SIMD 向量单元，没有这些指令；且 Ascend 每个 AI core 是单一重线程、无 CUDA 式轻量线程间同步，ANS 惯用的"多条独立子流并行解码"无法高效落地。因此 ENEC 选择定长编码（每组 ≤m 或 n 位 + bit mask），把熵编码的不规则控制流替换成向量化位运算，这在压缩率上较 ANS 系略低（BF16 CR 1.35 vs DietGPU-Float 1.47 类），但换来 263–523 GB/s 的 NPU 端吞吐。ENEC 论文把 ANS 描述为"GPU 友好的熵编码族"、在 Ascend 上"从根本上不兼容"（与 DietGPU 的 ANS float codec 对比，见 kernel调度层 DietGPU 条目）。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs
