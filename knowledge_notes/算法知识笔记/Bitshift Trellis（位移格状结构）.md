## Bitshift Trellis（位移格状结构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bitshift Trellis 由 Mao 和 Gray (2010) 在"随机排列 trellis 编码器"(RPTC)中引入，是 QTIP 的核心 trellis 结构。在 bitshift trellis 中，节点 i 到 j 有边当且仅当 ∃c ∈ Z, 0 ≤ c < 2^{kV} 使得 j = (i·2^{kV} mod 2^L) + c——即 j 的高 L-kV 位等于 i 的低 L-kV 位。这意味着：第 1 组 V 个权重仅依赖比特位置 {1,2,...,L}，第 2 组仅依赖 {kV+1,...,kV+L}，第 t 组仅依赖 {(t-1)kV+1,...,(t-1)kV+L}。解码时仅需 kV-bit 位移操作（所有硬件原生支持），且各组完全并行解码。无需存储 trellis 图结构（naive TCQ 需存储 2^L×2^{kV} 条边信息）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Bitshift trellis 解码流程（L=16, k=2, V=1）：
```
输入: packed bitstream B (L + kT bits, tail-biting aligned to 32-bit word)
1. 读取起始状态: state = B[0:16] (16-bit word)
2. 第 t=1 组: w_1 = codebook[state], 消耗 0 bits
3. for t = 2 to T:
     state = (state << 2) & 0xFFFF  # kV=2 bit 左移, 保留低 L=16 bits
     state |= B[16 + (t-2)*2 : 16 + (t-1)*2]  # 读入新 2 bits
     w_t = codebook[state]
4. 输出 Ŝ = [w_1, w_2, ..., w_T]
```
并行性：因 w_t 仅依赖 16-bit 窗口（允许 kV=2 bit 滑动），所有 w_t 可同时从编码中独立解码。对比 naive TCQ 需从第 1 bit 开始顺序遍历 trellis 图。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bitshift trellis 需要与伪随机 codebook 配合以避免相邻权重组的强相关性。QTIP 用 compute-based 近似高斯 codes 产生伪随机 codebook，效果接近随机高斯 codebook（RPTC 的原本方案需要存储或应用随机排列，开销过高）。在 GPU 上，bitshift 操作为单条指令（如 PTX shl），且各组并行解码在 CUDA thread 中实现。在 ARM CPU 上同样高效（所有 ARM 指令集均支持位移）。

涉及论文标题：
- QTIP: Quantization with Trellises and Incoherence Processing
