## GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出GyRot PE（Processing Element）微架构，实现fully integer-based dequantization datapath，支持W4A4 group quantization (G=32) + asymmetric quantization的高效运行时计算。核心PE设计：(1) 32-way INT4 dot product：每cycle执行32个4-bit activation (X0~31)与32个4-bit weight (W0~31)的点积，输出13-bit partial sum；(2) 重公式化integer dequantization pipeline：先乘activation scale SX (INT8)→加zero-point项 ZX×WSUM（WSUM=Σ_{i∈g} ŵ_i预计算）→乘weight scale SW (INT8)→全整数域完成dequantization，避免传统FP dequantization的type conversion和浮点开销；(3) 32-bit integer accumulator：全整数累加inter-group partial results→最终转FP16写output buffer；(4) FVU (Fused Vector Unit) 中FHT (Fast Hadamard Transform) unit：5-stage 32-way pipeline，含160 add/subtract units (32/stage)，支持O(n log₂ n)在线Hadamard rotation（当非线性层如SwiGLU/embedding介入时rotation无法fuse进weight），通过local register file + two-stage scheme支持scalable rotation up to 32×32=1024维，partial gating支持sub-32 power-of-two sizes (2/4/8/16/32)。实验比较PE级area和energy：GyRot-FP vs GyRot-INT vs Tender/MANT/LightRot PE在iso-throughput下28nm synthesis对比。GyRot-INT PE achieves 65.2% area和69.2% energy reduction over Tender。

- 后端平台是什么，配置是什么。
  GyRot custom accelerator：Samsung 28nm工艺，Synopsys Design Compiler综合，目标频率1GHz。PE array: 8×8×32 tensor organization (2048 parallel ops/cycle)。片上SRAM由commercial memory compiler生成。对比baseline（Tender/MANT/LightRot）均在相同28nm工艺、1GHz、iso-compute-area约束下综合评估。DRAM功耗由Micron DRAM Power Calculator (DDR4 model) 估算。

- 评估性能的软件/脚本是什么。修改了什么。
  RTL仿真验证PE功能正确性。Synopsys Design Compiler综合→area/power报告。对比分析：Tender PE（8-bit datapath，无group quantization）、MANT PE（G=64, FP16 SF, flexible data format）、LightRot PE（G=128, FP16 SF+FP16 ZP, floating-point dequantization）。GyRot的关键修改：(1) 将传统"先GEMM→后FP dequantize"的two-phase流程改为PE内部fused integer dequantization pipeline；(2) 预计算WSUM并broadcast到整行PE，消除per-PE重复计算；(3) FHT unit用add/subtract替代乘法实现Hadamard rotation，降低rotation硬件开销。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  论文未明确说明开源。PE计算流程（以G=32 W4A4为例）：
  1. 每个PE cycle：从input buffer读取32个INT4 activation X0~31→从weight buffer读取32个INT4 weight W0~31→32-way dot product → 13-bit partial sum
  2. Dequantization stage：partial sum × SX (INT8)→ + ZX×WSUM (WSUM预计算并broadcast到整行)→ × SW (INT8)→ dequantized result
  3. 32-bit integer accumulator：累加intra-group partial results→group边界处转FP16写output buffer
  4. FHT unit（需online rotation时）：load activation→5-stage add/subtract pipeline (每stage 32 parallel units)→完成Hadamard rotation→输出到quantization unit→进入PE array
  对比baseline：Tender无group quantization/dequantization overhead；MANT需FP16 SF乘法（FP multiplier per PE）；LightRot需FP16 SF+FP16 ZP运算（更重的FP dequantization）。GyRot-INT以INT8 SF/ZP全整数pipeline实现最低PE area和energy。

