## Fixed-to-Variable Dictionary Code for GPU Decoding（面向GPU解码的定长到可变字典编码）

术语是什么？
Fixed-to-Variable Dictionary Code 是 QMoE 选择的压缩编码方案，与传统的 variable-to-fixed 熵编码（如 Huffman：变长 codeword → 固定长度符号）相反，采用 fixed-length codeword (UINT16) → variable-length sequence of symbols（最多 28 个三元权重）。这种 LZW-style 编码的选择完全由 GPU 解码的硬件约束驱动：(1) 固定长度 codewords 消除变长码的序列解码依赖——每个 codeword 可独立并行查表解码；(2) 一个 warp 的全 32 threads 可联合处理一个 codeword——解决了"二进制字包含不同数量符号导致 warp divergence"的问题；(3) UINT16 codeword 无需慢速 bit-extraction（vs 变长码的 bit-level 操作）。

从kernel调度角度拆解术语：
**为何不用 Huffman（变长编码）？GPU 解码的三个致命挑战**：
```
Challenge 1: 序列依赖
  Huffman: symbol_i 的起始位需要知道前 i-1 个 symbol 的变长码字长度
  → 无法并行解码连续 symbol
  QMoE: codeword_i 独立，32 threads 同时查表解码

Challenge 2: Warp Divergence
  Huffman: 每个二进制字(INT32)可能包含不同数量的解码 symbol
  → 不同线程解码不同数量 symbol → warp divergence → 大量浪费操作
  QMoE: 固定 16-bit codeword, 28 threads 处理 1 codeword/cycle
  → warp 内均匀，无 divergence

Challenge 3: Bit Operations
  Huffman: 大量 bit shifts, masks, variable-length reads
  → GPU 的 bit ops 慢且不友好（vs CPU/ASIC）
  QMoE: 仅 2-bit shift+mask 从 UINT32 提取 ternary 值
  → 极少量 bit ops, 被 memory latency 完全隐藏
```

字典格式设计（硬件约束驱动）：
```
// Codeword: UINT16 (2^16 = 65536 个条目)
// 每个条目映射到 2×UINT32 (64 bit):
//   UINT32[0] → threads 0-13 使用
//   UINT32[1] → threads 14-27 使用
//   每 UINT32 格式:
//     bits[27:0]: 14×2-bit ternary values
//     bits[31:28]: pair_count (0-14)
//   pair_count 存两次确保每半可独立解码
// 总计: 2^16 × 8 bytes = 512KB → GPU L2 cache resident
```
选择 14 对（28 权重）作为最大序列长度：4 bits 存 pair_count（0-14），恰好 fit UINT32 低 4 bits。

术语一般如何实现？如何使用？
- 字典生成：Algorithm 1 (max-priority queue)，以三元值对概率为优先级贪心扩展最高概率序列
- 全局字典：一个字典服务所有 MoE 层/expert（避免 per-expert 存储开销）
- 压缩率 vs 理论极限：20.07× (c2048) vs 25.40× (Shannon 极限, p0=0.886) → ~20% 差距，换取 GPU 快速解码
- 字典按概率降序排列 → 高频 codewords 更可能已在 L1 cache → 自动 prefetch

涉及论文标题：
- QMoE Sub-1-Bit Compression of Trillion-Parameter Models
