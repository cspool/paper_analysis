## SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  基于AutoGPTQ扩展CUDA kernel支持group-wise混合精度推理。核心实现：(1) 按group组织的混合精度weight memory layout：每个group内的元素按相同精度packed为整数，无需额外padding（因为group_size=128是整数类型的倍数，即使3-bit也可充分利用整数字节空间）；(2) 引入extra bit-widths array记录每个group的精度（每个group用2-bit编码，聚合为整数）；(3) CUDA kernel按group逐一dequantize并计算向量点积：一个thread处理一列连续内存的ŵ_int，与block内共享的input activation做点积，累加到结果矩阵；一个warp内32 threads的data access模式保持相似，确保对齐。实验比较GPTQ和SliM-LLM在A800上的推理速度（Token/s）、Weight Memory (WM)和Runtime Memory (RM)，分别在LLaMA-7B/13B/2-7B和LLaMA-2-70B的2/3-bit配置下（Table 5, Table 14）。

- 后端平台是什么，配置是什么。
  单张NVIDIA A800-80GB GPU。CUDA kernel基于AutoGPTQ框架开发，利用CUDA Warp的32-thread单元，当group size=128时确保warp内threads的code structure和data access logic相似。

- 评估性能的软件/脚本是什么。修改了什么。
  评估框架是修改版AutoGPTQ (https://github.com/AutoGPTQ/AutoGPTQ)。修改内容：(1) 量化后输出每个group的scales、zeros和bit-widths信息；(2) AutoGPTQ根据各group精度将weights和zeros pack为整数压缩表示（ŵ_int, ẑ_int），zeros沿channel方向统一精度；(3) 添加extra array存储每个group的2-bit精度标记；(4) 在GPU端实现按group mixed-precision dequantization kernel：逐group处理→thread负责一列连续数据的dequantization→与block共享input activation做vector dot product→通过所有logical blocks迭代完成全linear layer计算。评估原理：测试FP16 baseline和不同bit-width配置下的weight memory (WM)、runtime memory (RM)、perplexity (PPL)和token/s推理速度。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/Aaronhuang-778/SliM-LLM，基于AutoGPTQ扩展。
  
  混合精度GPU推理全流程（以LLaMA-7B 2-bit SliM-LLM为例）：
  
  1. **量化阶段输出**：SliM-LLM量化完成后，对每个Linear层输出：
     - scales: FP16, shape (n, m/group_size)
     - zeros: 对应各group的zero point（packable格式）
     - 每个group的bit-width: b_i ∈ {1, 2, 3}
  
  2. **Weight Packing**：
     ```
     for each group g_i (128 elements):
         将128个权重值用b_i-bit量化为整数
         将这些整数沿channel方向pack进32-bit整数（无需padding，因为128可被任意2的幂整除）
     packing后: ŵ_int ∈ R^{m* × n}  (m*是压缩后维度)
     bit-widths array: 每group用2-bit存{1,2,3}标示，聚合为整数
     ```
  
  3. **GPU推理Kernel**：
     ```
     // 对每个Logical Block（覆盖一段连续channel区域）
     for each block:
         加载共享的input activation片段到shared memory
         for each group in this block:
             读取bit-widths[g]确定精度b
             从ŵ_int中按累积偏移读取该group的packed整数
             for each thread (处理一列连续数据):
                 解包(dequantize): w_fp = (ŵ_int_val - z) * scale
                 向量点积: acc += w_fp · activation[shared]
         累加结果写入output matrix对应位置
     ```
  
  4. **性能权衡**：2-bit LLaMA-7B: WM 2.3G（vs GPTQ 2.2G），PPL 14.58（vs GPTQ 152.31），Token/s 61.2（vs GPTQ 83.9）。混合精度因额外bit-widths array和1-bit group的额外计算开销略有降速，但换取了大幅质量提升。
