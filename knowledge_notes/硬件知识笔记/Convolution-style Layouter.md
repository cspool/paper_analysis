## Convolution-style Layouter

术语解释
Convolution-style Layouter是Focus SIC中的硬件地址生成模块，将systolic array GEMM tile输出的vectors按照原始图像帧的(Frame, Height, Width)坐标动态重组为2×2×2时空block结构，并实现无bank conflict的并行内存访问。

术语是什么？通过联网搜索让回答具体和精准。
Convolution-style Layouter是Focus SIC的关键硬件组件，解决两个核心问题：(1) 恢复prune后token的空间位置：SEC semantic pruning破坏了token的连续空间结构，layouter利用SEC生成的offset encoding恢复每个retained token的原始(Frame, Height, Width)坐标，将vectors重组为FHW-order的3D tensor布局；(2) 避免并行访问的内存bank conflict：为同时读取2×2×2 block内的8个vectors用于similarity matching，layouter使用确定性映射公式 `Bank = f mod 2 × 4 + r mod 2 × 2 + c mod 2` 和 `Offset = floor(r/2) × ceil(W/2) + floor(c/2)` 将8个vectors映射到8个不同的SRAM bank，实现无复制、无conflict的并行读取。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Convolution-style Layouter工作流程：
```
Input: GEMM tile output vectors (m=1024, each 32-dim)
       + SEC offset encoding (per-vector relative position)

1. Position Recovery:
   for each vector in tile:
       restore (f, r, c) = decode_from_offset_chain(offset_stream)
       # f: frame_id (0 or 1 for adjacent frames)
       # r: row_id in frame
       # c: col_id in frame

2. FHW Layout: reorder vectors by Frame → Height → Width

3. Block Formation: 2×2×2 sliding window (stride 1)
   Each block covers: 4 vectors from frame A + 4 vectors from frame B
   (h,w), (h,w+1), (h+1,w), (h+1,w+1) from each frame

4. Conflict-free Bank Mapping:
   For vector at (f, r, c):
       Bank[f,r,c] = f%2 × 4 + r%2 × 2 + c%2  → 0..7
       Offset[f,r,c] = floor(r/2) × ceil(W/2) + floor(c/2)
   All 8 vectors in any 2×2×2 block → distinct banks (0..7)
   → No bank conflict, all 8 vectors readable in 1 cycle

5. Buffer: 16KB layouter buffer stores 256-vector sliding window
```
与传统CNN accelerator的input duplication方法（up to 8× memory overhead）不同，convolution-style layout通过地址映射消除conflict，无需数据复制。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Layouter用SystemVerilog实现为SIC的地址生成和数据重排模块，16KB buffer (256 vectors × 32 dims × FP16 = 16KB)。Bank mapping基于简单算术（mod和division by powers of 2，可用bit slicing实现），组合逻辑延迟低。Layouter支持scaling：可扩展到更大block size（需要更多banks）或不同frame counts（调整Bank公式的系数）。开源RTL见https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

