## Squat (EdgeQAT): SIMD-based Multi-Kernel Mixed-Precision Multiplier for Mobile LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现SIMD-based Multi-Kernel Mixed-Precision (MKMP) multiplier，支持sub-8-bit混合精度MAC操作。核心创新：(1) **INT4 concatenation**：将两个4-bit权重拼接存入一个16-bit寄存器，利用ARM `mla`指令（32-bit目标寄存器INT32）同时完成乘加，理论上将4-bit GEMM的计算量减半；(2) **Token Control Logic Module (TCLM)**：在推理时根据注意力分数动态分组token为8-bit和4-bit，分别用INT8 multiplier和INT4 multiplier执行；(3) **Compiler-level memory优化**：优化计算线程分配，重叠内存读取时间。
  实验比较：在OnePlus 11和Raspberry Pi 5上测量W4A4、W8A8以及多种W4A8混合比例（4:8=1:3/1:1/3:1）的端到端推理延迟（ms/Token），对比FP16 baseline。

- 后端平台是什么，配置是什么。
  - OnePlus 11：Snapdragon 8 Gen 2处理器，全部核心多线程计算。
  - Raspberry Pi 5：BCM2712四核Arm Cortex A76处理器，四核全用。
  - 指令集：ARMv8 SIMD（NEON），利用`vmlaq_s8()`等8-bit SIMD乘加指令。`mla`指令使用32-bit目标寄存器（INT32 datatype）在单指令内完成乘法和累加。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估方式：在目标设备上部署量化模型，测量1000次迭代的延迟取平均（ms/Token），输入序列长度128。
  - 修改内容（MKMP multiplier核心设计）：
    1. **INT4 Concatenation Kernel**（Figure 6）：将两个4-bit操作数拼接进一个16-bit寄存器。低比特优先策略（low-bit priority strategy）均匀利用位宽，最小化冗余零。16-bit宽乘法操作后内部拆分结果，维护数学精度。数学上将4-bit GEMM的乘加操作数减半（vs 传统扩展到8-bit再计算）。
    2. **INT4 Multiplier**：基于现有INT8 multiplier构建。将相邻行权重拼接，与共享激活值在SIMD kernel中相乘。利用SIMD mem机制，通过bit-shift和逐行求和累加中间值。INT4 multiplier节省50% INT8 multiplier硬件资源。
    3. **TCLM (Token Control Logic Module)**：Heapsort实现TopK重要token选择（marginal overhead）→ 分别拼接8-bit和4-bit token组 → 调用对应multiplier执行混合精度MAC。
    4. **Compiler-level优化**：针对LLM巨大内存读出的特点，优化并分配不同操作的计算线程，从编译器层面重叠内存读取时间。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源：https://github.com/shawnricecake/squant

  **MKMP Multiplier执行全过程（以LLaMA-58M W4A8(1:1)推理一个token为例）：**

  **阶段 1: Token Adaptive Quantization（TCLM）**
  - 输入：上层的output hidden state h ∈ R^4096 × N_tokens，最新的attention map attn
  - Step 1a: scores = attn[:, 0]（每个token对初始token的平均注意力）
  - Step 1b: threshold = Heapsort_TopK(scores, k=ρ*N)，ρ=0.5即为一半token
  - Step 1c: 按threshold分组 → x_8bit = concat(important tokens)，x_4bit = concat(less important tokens)

  **阶段 2: INT8 Multiplier（处理8-bit token组）**
  - 输入：W_q INT4 packed weights（离线量化），x_8bit INT8量化激活
  - Step 2a: SIMD加载packed INT4权重 & INT8激活
  - Step 2b: ARM `vmlaq_s8()` SIMD指令执行INT8×INT8乘加 → INT32累加寄存器
  - Step 2c: Dequantization：o_8bit = α_x·α_w·C_int32
  - 输出：FP16格式的部分结果

  **阶段 3: INT4 Multiplier（处理4-bit token组）**
  - 输入：W_q INT4 packed weights，x_4bit INT4量化激活
  - Step 3a: INT4 Concatenation：将相邻行权重各4-bit拼接为16-bit寄存器（low-bit priority策略）
  - Step 3b: 16-bit宽乘加指令执行（利用`mla`，32-bit目标寄存器），内部拆分保持数学精度
  - Step 3c: Bit-shift + row-by-row summation累加中间值
  - Step 3d: Dequantization：o_4bit = α_x·α_w·C_int32
  - 输出：FP16格式的部分结果
  - INT4 multiplier节省50% INT8 multiplier的硬件资源，理论计算量减半

  **阶段 4: 结果合并与输出**
  - o = concat_and_reorder(o_8bit, o_4bit)，按原始token顺序恢复
  - 输入到下一层的LayerNorm → QKV projection → ...

  **性能输出：**
  - LLaMA-58M OnePlus 11：FP16=4.54 ms/tok → W8A8=3.22 (1.41×) → W4A4=2.02 (2.24×)
  - LLaMA-58M Raspberry Pi 5：FP16=15.63 → W8A8=9.40 (1.66×) → W4A4=6.78 (2.31×)
  - GPT2-97M OnePlus 11：FP16=6.22 → W8A8=4.35 (1.43×) → W4A4=2.75 (2.26×)
  - GPT2-97M Raspberry Pi 5：FP16=23.04 → W8A8=13.75 (1.68×) → W4A4=9.74 (2.37×)
  - 混合精度W4A8(1:1)：Raspberry Pi上额外加速超40%（vs uniform 8-bit），同时保持W4A8精度优势
  - 模型越大加速越显著（GPT2-97M > LLaMA-58M），因内存访问减少带来的效率提升更大
