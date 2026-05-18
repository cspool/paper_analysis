## TMAC (Tile Multiplier-Accumulator) / Vector-Tile MAC（向量-瓦片乘累加器）

术语是什么？通过联网搜索让回答具体和精准。

TMAC（Tile Multiplier-Accumulator，也称Vector-Tile MAC）是RPU论文提出的一种专用矩阵乘法硬件单元，用于加速LLM decode中dominant的Vector-Matrix Multiplication (VMM)操作。每个TMAC由64个MAC单元组成，排列为8×8阵列，支持BF16乘法+FP32累加（BF16×BF16→FP32）。一个activation vector的8个元素被广播到8列weight matrix上，每cycle计算64个MAC操作。TMAC采用**weight-streaming, output-stationary dataflow**：activation保持不动（broadcast across columns），weight元素逐tile被stream进TMAC，partial sum驻留在local accumulator中直到一个stripe完成后再做列级tree-sum reduction。每Reasoning Core包含4个TMAC（共256 MACs/core），每TMAC通过1024-bit compute bus从Stream Decoder接收dequantized weight tiles。

从硬件架构角度拆解：

TMAC的硬件架构和数据流：
1. **Input**: Activation shard（64 BF16 values）从network buffer加载到靠近TMAC的activation register file（每tile column 8 entries, 128b），在tile columns间复用；Weight tiles通过1024-bit compute bus从Stream Decoder广播到所有active TMAC。
2. **Weight-streaming dataflow**: 每cycle weight元素沿tile行列方向stream-in，activation保持不动（broadcast across 8 columns），weight元素依次乘以8列activation vector element。每个tile的64个MAC并行执行。
3. **Output-stationary accumulation**: 每个MAC的FP32 accumulator保持partial sum在local register中，直到当前stripe（8个垂直堆叠tile跨所有weight shard列）完成。
4. **Column-wise tree-sum reduction**: Stripe内所有tile rows处理完后，每列的8个partial sums通过3-stage tree-sum adder（每stage将相邻两个accumulator相加）reduce为1个output value→write back to local output register。
5. **Stripe-based traversal**: Tile multipliers先在一个stripe内遍历tile-rows（column-first处理），处理完一列tile后做tree-sum reduction，再移到下一列。这样避免了inner-product style（需full activation on-chip）或outer-product style（高partial sum writeback bandwidth）的缺点。

每TMAC包含：8×8 MAC阵列（BF16 multiplier + FP32 accumulator）× 8 columns + 3-stage tree-sum reduction tree + local accumulator register。

术语一般如何实现？如何使用？

TMAC作为RPU Reasoning Core的核心计算单元，通过RPU ISA指令驱动。Compiler将PyTorch的torch.nn.Linear降为三阶段micro-kernel (Loading/Looping/Launching)，其中Looping阶段驱动TMAC执行stripe-based VMM。每TMAC连接Stream Decoder（on-the-fly dequantization BFP/MxFP/NxFP→BF16）和activation register file。TMAC的RTL实现使用SystemC + Catapult HLS，由VCS/Design Compiler/PowerPro在TSMC N16合成并project到N2，面积12,800 um²/TMAC，能耗25.6 pJ/TMAC-operation。论文未开源TMAC RTL。

涉及论文标题：
- RPU - A Reasoning Processing Unit

