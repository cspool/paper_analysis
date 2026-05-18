## MXFP4 / NVFP4 (Micro-Scaling FP4 Formats on Blackwell)

术语是什么？通过联网搜索让回答具体和精准。

MXFP4 (Microscaling FP4) 和 NVFP4 是 NVIDIA Blackwell 架构（B200/RTX 5090, Compute Capability 10.0/SM100）原生支持的微缩放4位浮点格式。每个FP4元素仅占4 bits（1-bit sign + 2-bit exponent + 1-bit mantissa），但每块（block）数据共享一个公共的FP8 scaling factor（由E5M2或E8M0格式表示），通过"微缩放"（microscaling）机制将实际上有限的值域动态放大，使4-bit可以表示原本FP16的数值范围。MXFP4是OCP (Open Compute Project) 制定的开放标准Microscaling Formats (MX) 系列中的一种：MX格式族包括MXFP8、MXFP6、MXFP4等，核心思想是将数据分组为block（通常32个元素），每组共享一个公共scale factor，在存储和传输时用低bit-width表示元素值，计算前用scale factor恢复。NVFP4是NVIDIA自己的对应实现，与MXFP4在bit layout上可能存在差异但理念一致。Blackwell的Tensor Cores直接支持mxfp4/nvfp4格式的mma指令，无需软件dequantization——硬件读取packed 4-bit数据和block-scale factor后直接在TC内完成乘累加。Blackwell B200的mxfp4 Tensor Cores峰值吞吐可达20 PFLOPS（FP16的4×以上）。

从硬件架构角度拆解术语：

Blackwell原生MXFP4 attention执行流程（以BitDecoding on RTX 5090, GQA decode step为例）：
1. **数据存储**：KV cache以packed MXFP4格式存储在HBM中，每32个元素共享一个E8M0 block-scale factor。K_packed ∈ INT32[N/8]（每个INT32存8个FP4值），K_scale ∈ FP8[N/32]
2. **数据加载**：cp.async/TMA加载Q (FP16) 和 K_packed (MXFP4 packed) 到shared memory
3. **直接mma**：Blackwell mxfp4 mma指令读取Q (FP16) 和 K_packed (mxfp4) → Tensor Core内部自动用block-scale factor做microscaling恢复值域 → 执行矩阵乘累加 → 输出FP32 accumulator。**无需软件dequantization步骤**
4. **P re-quantization**：softmax后的P矩阵 (FP16) 需on-the-fly quantize to mxfp4才能参与P×V mma（因V也是mxfp4 packed）。这是Blackwell的一个trade-off：省去了dequant但引入了P的re-quantization bottleneck，BitDecoding用warp parallelism缓解
5. **对比Hopper/Ampere**：Broader GPU上需lop3-based software dequant (INT4→FP16) + 标准FP16 mma，Blackwell的硬件原生支持将dequant latency降到0

术语一般如何实现？如何使用？

MXFP4/NVFP4的使用需要：① 数据按block-size对齐打包（32-element blocks），每block一个scale factor；② 使用Blackwell专用的mma PTX指令（如`mma.mxfp4`），指令接受的operand layout由硬件定义（类似Tensor Cores fragment layout）；③ scale factor的布局需与mma指令的expectation匹配（BitDecoding的layout-agnostic设计自动适应）；④ Block-scale的计算在量化时完成：per-block找absmax → 计算scale使值域适配FP4的表示范围 → 各元素除scale取最近的FP4表示。NVIDIA提供CUTLASS 3.x的MX format支持。开源使用参考BitDecoding: https://github.com/OpenBitSys/BitDecoding。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

