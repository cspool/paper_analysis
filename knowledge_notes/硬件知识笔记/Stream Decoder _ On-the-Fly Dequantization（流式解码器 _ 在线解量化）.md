## Stream Decoder / On-the-Fly Dequantization（流式解码器 / 在线解量化）

术语是什么？通过联网搜索让回答具体和精准。

Stream Decoder是RPU论文提出的硬件模块，用于在权重数据从on-chip buffer流入compute pipeline的传输过程中实时（on-the-fly）将压缩的block-quantized weight tiles（BFP/MxFP/NxFP, 4-8 bit）解码还原为BF16标准精度。这使权重可以以压缩格式存储在off-chip HBM-CO中（减少off-chip capacity和energy），但计算仍以BF16 full precision执行。Stream Decoder支持多种block-quantization格式的configurable bitwidths（4-8 bit），具有format-flexible解码能力。

从硬件架构角度拆解：

Stream Decoder的硬件pipeline：
1. **Input**: Memory DMA从HBM-CO读取compressed weight tiles→写入memory buffer（compressed format）。
2. **Compute DMA**: 从memory buffer读取compressed weight tiles→stream into Stream Decoder。
3. **Decoding process**: Stream Decoder识别block format type（BFP/MxFP/NxFP）→读取block metadata（shared exponent per block）→对block内每个weight element执行：shared_exponent × mantissa → BF16 value。持续解码直到一个full batch of 64 BF16 values完整重构（对应一个weight tile）。
4. **Output**: 64 BF16 values通过1024-bit wide compute bus广播到所有active TMAC。

关键约束：block-quantized格式需要dynamic exponent broadcasting和alignment，这些conditional logic在Stream Decoder中通过专用硬件高效实现，而PIM架构由于DRAM内logic限制难以支持此类浮点操作。

术语一般如何实现？如何使用？

Stream Decoder作为RPU Reasoning Core的固定功能硬件模块，在所有VMM kernel执行中自动介入（weight必须先经过Stream Decoder dequantize才能进入TMAC）。支持的格式包括BFP（Block Floating Point [53]）、MxFP（Microscaling FP [15]）和NxFP（Narrow FP [39]）。Configurable bitwidth从4到8 bit，适应不同精度-带宽tradeoff。论文未单独开源Stream Decoder RTL，为RPU整体RTL model的一部分（SystemC + Catapult HLS）。Energy savings: on-the-fly dequantization使SRAM interface energy降低1.7×（因为compressed data movement减少）。

涉及论文标题：
- RPU - A Reasoning Processing Unit

