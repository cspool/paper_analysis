## GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

- 属于硬件架构的实现是什么？实验比较什么？
  实现为GyRot accelerator，一个基于systolic array的专用LLM推理加速器，支持INT4精度下fine-grained group quantization (G=32)和rotation的融合计算。架构由以下模块组成：(1) 8×8×32 tensor PE array：8×8 systolic array，每个PE执行32-way INT4 dot product（每cycle 2048 parallel operations），output-stationary dataflow。PE内集成fully integer dequantization datapath（INT8 SX→ZX×WSUM→INT8 SW），32-bit integer accumulator。(2) FVU (Fused Vector Unit)：包含nonlinear/element-wise operation unit和5-stage 32-way FHT rotation unit（160 add/subtract units, 32/stage），支持在线Hadamard rotation最高1024维。FHT unit支持partial gating实现power-of-two sub-32 sizes (2/4/8/16/32)。(3) Input Buffer：64KB data + 8KB metadata (SX/ZX)，multi-bank结构保证带宽。(4) Weight Buffer：64KB data + 4KB metadata (SW)，multi-bank结构。(5) Global Buffer：512KB片上共享存储。(6) WSUM unit：8×32-way adder-tree，预计算per-group weight sum供整行PE共享。HAP permutation可fuse进weight（permutation-invariant），无runtime overhead。实验比较：在LLaMA-1/2/3模型上对比Tender (W8A8)、MANT (W4A8/W4A4)、LightRot (W4A4)，GyRot-INT取得geomean 1.42–3.40× speedup和1.20–3.64× energy efficiency提升。PE级对比：GyRot-FP相对Tender面积减45.6%/能耗减51.0%；GyRot-INT面积减65.2%/能耗减69.2%。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  未使用模拟器。GyRot accelerator PE及所有组件以RTL (SystemVerilog)实现，RTL仿真验证功能正确性。Synopsys Design Compiler综合（Samsung 28nm工艺），目标频率1GHz。片上SRAM由commercial memory compiler生成（同工艺节点）。DRAM功耗由Micron DRAM Power Calculator (DDR4 model)估算。所有baseline accelerator（Tender/MANT/LightRot）在相同iso-compute-area约束下、相同工艺节点和频率下重新综合以公平对比。

- 模拟器模拟什么的性能，修改了什么。
  不适用（RTL综合+功耗分析，非模拟器）。综合报告提供面积(mm²)和功耗(mW)数据。系统级性能（speedup/energy efficiency）基于cycle-level execution time建模，结合DRAM access energy。Table VIII给出Area/Power breakdown：PE Array (INT Tensor) 0.26 mm² (12.4%), 410.24 mW (55.4%)；PE Array (Dequant. + Accum.) 0.09 mm² (4.2%), 118.40 mW (16.0%)；WSUM unit 0.01 mm² (0.5%), 17.09 mW (2.3%)；Input Buf. 0.24 mm² (11.4%), 82.43 mW (11.1%)；Weight Buf. 0.23 mm² (10.8%), 64.38 mW (8.7%)；Global Buf. 1.20 mm² (57.1%), 41.63 mW (5.6%)；FVU 0.07 mm² (3.5%), 6.78 mW (0.9%)。Total: 2.10 mm², 740.95 mW。

- 开源情况。模拟器如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未明确说明开源。GyRot accelerator推理流程（以LLaMA-3-8B W4A4KV4为例）：
  1. Input buffer加载activation X及metadata (SX, ZX)，Weight buffer加载weight W及SW
  2. 若当前层为非线性层后的linear层→FVU执行online Hadamard rotation on activation（FHT unit: 5-stage add/subtract pipeline）→量化后送入PE array
  3. PE array (8×8×32 tensor) 执行systolic output-stationary dataflow：每个PE 32-way INT4 dot product→integer dequantization (SX→ZX×WSUM→SW)→32-bit int accumulate
  4. 跨group accumulation在PE accumulator中完成→最终结果转FP16写output buffer
  5. WSUM unit预计算group-wise weight sum，broadcast到整行PE共享
  6. 与GPU方案的关键区别：GPU上group quantization的dequantization在CUDA cores上以FP精度执行（INT GEMM→FP convert→FP dequantize→FP accumulate），GyRot在PE内以全整数完成dequantization和accumulation，消除type conversion开销和FP arithmetic overhead
