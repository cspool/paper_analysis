## Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现：(1) **3:4 稀疏 5-bit 打包方案的 SIMD 高效推理 kernel**：将每 4 个三值权重（含恰好 3 个非零、1 个为零）打包为 5-bit 索引，利用 4-way 对齐模式对现代 CPU SIMD 向量通道友好，通过查表引擎（BitNet.cpp, T-MAC）替代浮点乘法为整数加法；(2) **与 BitNet I2_S（2-bit 打包）和 Tequila TL2（1.67-bit 打包）的运行时性能对比**。实验比较：(1) CPU 推理吞吐量（tokens/s, Intel i7-14700HX）；(2) 模型大小（MB, GGUF 格式）。

- 后端平台是什么，配置是什么。
  Intel i7-14700HX CPU，固定线程配置（AngelSlim 框架层面使用 2 threads）。查表引擎：BitNet.cpp（基于 ARM/x86 SIMD 的三值矩阵乘法 kernel）和 T-MAC（CPU 查表低比特部署 kernel）。推理使用 GGUF 格式。

- 评估性能的软件/脚本是什么。修改了什么。
  评估使用 AngelSlim 框架内置的性能测量脚本（论文未给出具体脚本名称），通过 llama.cpp 类推理引擎加载 GGUF 格式量化模型，测量 generation throughput（tokens/s）和模型大小。修改内容：Sherry 的 3:4 稀疏 5-bit 打包方案需要自定义 (de)packing kernel —— 在推理时将 5-bit 索引解码为三值权重 + 位置 mask，然后通过查表进行矩阵乘法。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/Tencent/AngelSlim（sherry 分支）

  **Sherry 推理 kernel 评估流程详解**：

  **Kernel 输入到性能输出全过程：**

  ```
  # ===== 离线阶段：模型打包 =====
  # 输入：训练好的三值权重矩阵 W ∈ {+1, 0, -1}^{N}，含 3:4 结构化稀疏
  # 处理：
  for each group of 4 consecutive weights (w_0, w_1, w_2, w_3):
      # 该组有 C(4,3) × 2³ = 32 种可能状态
      zero_pos = index of the single zero weight  # 4 种可能
      signs = sign of the 3 non-zero weights      # 2³ = 8 种可能
      packed_5bit = encode(zero_pos, signs)       # → [0, 31]  
      store packed_5bit in GGUF format

  # ===== 在线阶段：推理解包与计算 =====
  # 输入：packed 5-bit indices, scale factor α, activation vector X
  # 过程：
  for each group of 4 weights:
      # Step 1: Decode
      packed = load_5bit(idx)                    # 从 GGUF 内存读取 5-bit
      zero_pos, signs = decode(packed)            # 解出零值和符号
      # Step 2: Reconstruct ternary weights
      w_hat = [s_0, s_1, s_2, s_3] where s at zero_pos = 0  # 三值向量
      # Step 3: Multiply via LUT (Lookup Table)
      # 4-way 对齐天然适合 128-bit SIMD:
      # - 128-bit SIMD 处理 4 个 FP16 = 完美 1 组
      # - 256-bit SIMD 处理 4 个 FP32 = 完美 1 组
      y_g = α · (X[g*4]·w_hat[0] + X[g*4+1]·w_hat[1] + 
                 X[g*4+2]·w_hat[2] + X[g*4+3]·w_hat[3])
      # 或等效于查表: y_g = α · LUT[packed](X[g*4:g*4+4])

  # ===== 性能测量 =====
  # 输入：输入 tokens（seq_len=256~1024 for TTFT, output_len=p for generation）
  # 测量：
  # - Prefill latency (TTFT, ms): 从接收 prompt tokens 到生成第一个 token 的延迟
  # - Generation throughput (tokens/s): 每秒生成的输出 token 数
  # - Model size (MB): GGUF 文件在磁盘上的大小
  ```

  **SIMD 对齐优势（对比 BitNet 2-bit 和 Tequila 1.67-bit）：**
  - BitNet 2-bit: 4 权重 → 8 bits → 浪费 37.5% 存储，但 4-way 对齐 SIMD ✓
  - Tequila 1.67-bit: 3 权重 → 5 bits → 3-way pattern，512-bit SIMD = 10 group 余 2 权重 → 不完美对齐 ✗
  - Sherry 1.25-bit: 4 权重 → 5 bits → 4-way pattern，512-bit SIMD = 128 group 整除 → 完美对齐 ✓

  **评估结果（Table 3，Intel i7-14700HX）：**
  | Scale | Method  | Bits | Speed (t/s) | Size (MB) |
  |-------|---------|------|-------------|-----------|
  | 0.7B  | BF16    | 16   | 34.01       | 1360.0    |
  | 0.7B  | BitNet  | 2.0  | 132.13      | 256.56    |
  | 0.7B  | Tequila | 1.67 | 116.83      | 233.44    |
  | 0.7B  | Sherry  | 1.25 | 148.27      | 205.50    |
  | 3B    | BF16    | 16   | 7.55        | 6190.0    |
  | 3B    | BitNet  | 2.0  | 41.87       | 873.65    |
  | 3B    | Tequila | 1.67 | 38.80       | 846.01    |
  | 3B    | Sherry  | 1.25 | 45.55       | 712.40    |
