## Warp-Cooperative rANS 解码（warp 协作并行解码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把串行的 rANS 状态机用 warp 级并行化的技术：rANS 状态转移依赖前序符号，单流无法并行；解法是把每 tile 的压缩位流划分为 R 个独立 substream（ANS 状态可独立初始化），每 warp lane 维护一个 rANS 状态、各解一条 substream，一个 warp 并发推进 R 条流。本论文扩展 DietGPU 的做法，让解码结果按 GEMM 所需 swizzle 布局直写 shared memory。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm V.2 伪代码（论文）：
```
RansDecodeTile(A, stream, T~):
  init rANS state s per lane                      # 每 lane 一条 substream
  for i = 0 .. S_lane-1:
      x = s.value mod R                           # 取状态低 b 位 → 表槽
      (σ, f, c) = T~[x]                           # shared memory 查解码表
      w = DecodeSymbol(σ)                         # 符号
      (r, c) = symbol index → write A[r,c] = w    # 直写 shared memory tile
      s.value = f * floor(s.value/R) + (x - c)    # rANS 状态回溯
      while s.value < renorm_thresh:
          u = load_32bit(stream)                  # 跨 lane 交错 → coalesced
          s.value = (s.value << 32) | u
```
Annotations：R = 2^b（b=12 概率精度）；T~ 从全局 codebook 拷贝进 shared memory 实现低延迟高带宽查表；重归一化 load 因压缩流跨 lane 交错而天然 coalesced，保持全局内存吞吐；写 A 的 (r,c) 与交错解码序锁步，无需二次布局变换。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基座 DietGPU（Meta，A100 解码 250–410 GB/s、H100 约 592 GB/s）已提供 warp-cooperative rANS；本论文在此之上做三点改造：tile 粒度 substream 划分（对齐 GEMM tile）、直写 shared memory（不落全局内存）、与 tensor-core GEMM 融合流水。使用：GPU 上近熵解码、数据搬运压缩（all-gather/reduce-scatter 前压缩）、LLM 权重无损压缩后端。实现要点：每 lane 状态的独立重归一化、解码表常驻 shared memory 降低查表延迟。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
