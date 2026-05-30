## LiquidQuant (LQQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LiquidQuant (LQQ) 是LiquidGEMM论文提出的硬件高效W4A8 dequantization算法。核心机制：通过rotation-based transformation将INT8先shift到UINT8域再量化到UINT4，利用two's complement同余性质（i ≡ j mod 2^8 → 相同二进制表示）设计无溢出dequantization。Dequantization公式：Q_i8 = (Q_u4 × s_u8 + a) XOR 0x80，其中a = 128 + min(Q_i8)预计算offline。关键洞察：(1) Q_u4∈[0,15], s_u8≤16 → Q_u4×s_u8≤240（UINT8安全）；(2) a∈[9,247]（UINT8安全）；(3) XOR 0x80等价于翻转MSB，实现条件性±128，使结果落在INT8的二进制表示内。仅需两条32-bit指令（IMAD + XOR）处理四个元素 vs QServe的QoQ算法10+指令。8个元素（含unpack）仅需7条指令（0.875指令/元素）。

从算法pipeline角度拆解术语：
```
量化 (离线):
  Q_i8 ∈ [-119, 119] (per-channel, protective range)
  Q_u8 = Q_i8 - min(Q_i8)        // shift到UINT8: [-119,119] → [0,238]
  s_u8 = round(max(Q_u8) / 15)   // ≤16
  Q_u4 = round(Q_u8 / s_u8)      // [0,15] UINT4
  a = 128 + min(Q_i8)            // [9, 247], precomputed per-group

Dequantization (在线, CUDA Cores):
  // Unpack (QServe method): 8×UINT4 → 2×32-bit regs
  // Dequantization (2 instructions per reg, 4 elements):
  reg = IMAD(reg, s_u8_bcast, a_bcast)  // multiply-add (1 instr)
  reg = XOR(reg, 0x80808080)            // flip MSB each byte (1 instr)
  // Result: INT8 binary in UINT8 registers → directly usable by WGMMA
```

术语一般如何实现？如何使用？
LQQ用CUDA实现（非PTX），利用IMAD和XOR原生GPU指令。s_u8和a预计算per-group（共K/64组/channel）。集成到CUTLASS/Cute warp-specialized kernel中，fuse dequantization到MMA mainloop。LQQ保持模型精度——WikiText2 perplexity和zero-shot准确率与QServe相当。

涉及论文标题：
- LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

---
