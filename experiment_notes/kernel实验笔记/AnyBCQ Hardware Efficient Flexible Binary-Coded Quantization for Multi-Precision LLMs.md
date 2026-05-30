## AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：为 AnyBCQ 多精度 LLM 设计的 CUDA kernel，直接对二值比特平面（binary bit-planes）操作。与 Any-Precision LLM 的 kernel（需比特转置 + centroid table lookup）不同，AnyBCQ kernel 直接按需加载 p 个比特平面，利用 BCQ 的 {-1,+1} 二值特性将运算简化为激活元素的加减，并通过 LUT-based GEMM 方案缓存高频重复的部分和结果以减少算术开销。每个比特平面的计算结果乘以对应缩放因子 α_i 后累加为部分和，p 个比特平面完成后输出最终结果。
  - 实验比较：GEMV kernel 延迟（µs），对比 cuBLAS FP16、Any-Precision LLM（2/3/4-bit）在三种模型层形状（Llama-3.1-8B、Phi-4-14B、Llama-3.1-70B 的线性层维度）下的延迟。端到端吞吐量（tokens/sec）对比 Any-Precision LLM。附录还包括 kernel 延迟分解（bit-transpose vs LUT lookup vs GEMM）和 A100/H100 跨 GPU 平台吞吐量对比。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A100 80GB HBM（主平台），NVIDIA H100（附录 A.3 跨平台验证）。
  - 运行环境：CUDA 12.6。
  - 测量工具：nvidia-smi（功耗/利用率）、CUDA clock64()（kernel 内周期级延迟分解）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估方式：自定义 CUDA kernel 基准测试脚本，测量不同矩阵形状 (M×K) 的 GEMV 延迟。端到端评估使用 HuggingFace 模型加载 + 自定义 kernel 的推理流程。
  - Kernel 设计修改（对比 Any-Precision LLM kernel，Figure 3）：

    **Any-Precision LLM kernel（baseline kernel，Figure 3a）：**
    1. 加载 M×K×p 比特平面张量
    2. 比特转置（bit-transpose）：将 p 个比特平面重新排列为 M×K p-bit 索引矩阵
    3. 通过 centroid table lookup 获取每个权重的反量化值
    4. 执行 GEMM

    **AnyBCQ kernel（论文方法，Figure 3b）：**
    1. 按需加载前 p 个 M×K 比特平面（仅加载需要精度的比特，不做冗余加载）
    2. 每个比特平面直接操作：因 B_i ∈ {-1,+1}，激活元素仅需加法/减法
    3. LUT-based GEMM 优化：预计算并缓存频繁重复的部分和组合
    4. 比特平面输出 × 对应 α_i → 部分和累加
    5. p 个比特平面完成后输出最终结果
    - 消除了 Any-Precision LLM kernel 的两个主要开销：bit-transpose（占延迟 35-58%）和 centroid table lookup（占延迟 9-17%）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 代码开源：https://github.com/naver-aics/anybcq（含 CUDA kernel 和基准测试脚本）

  **Kernel 执行全过程（以 Llama-3.1-8B 某线性层 M=4096, K=4096, p=2-bit 推理为例）：**

  | 阶段 | 描述 | 输入/操作 | 输出 |
  |------|------|-----------|------|
  | 1. 比特平面加载 | 从 HBM 加载前 p 个比特平面（每个 B_i ∈ {-1,+1}^{4096×4096}, 1 bit/元素）| packed binary tensor（M×K×p） | 寄存器中的 B_1, B_2 |
  | 2. 缩放因子加载 | 加载当前 p-bit 精度对应的缩放因子 | α_1, α_2 ∈ R（per-group, g=128） | 寄存器中的 α |
  | 3. LUT 构建 | 对激活向量每 w 个元素预计算 2^w 种可能的加减组合 | 激活向量片段 | LUT entries（FP16） |
  | 4. 比特平面计算 | B_1 比特平面：每个 M×K 元素用 sign 选择加/减激活值；LUT 查表代替逐一计算 | B_1, activation, LUT | partial_sum_1 |
  | 5. 缩放累加 | partial_sum_1 × α_1；同理对 B_2 执行步骤 4-5 | partial_sum_{1,2}, α_{1,2} | accumulated = Σ α_i · (B_i ⊙ activation) |
  | 6. 输出 | p 个比特平面累加完成，写回 HBM | accumulated ∈ R^{M} | 输出向量 |

  **性能结果（GEMV latency, µs, A100）：**
  - M=4096, K=4096, 2-bit: AnyBCQ=223 (×1.33 vs cuBLAS), Any-Precision LLM=230 (×1.29)
  - M=14336, K=4096, 2-bit: AnyBCQ=319 (×2.67 vs cuBLAS), Any-Precision LLM=353 (×2.41)
  - M=4096, K=14336, 2-bit: AnyBCQ=315 (×2.78 vs cuBLAS), Any-Precision LLM=356 (×2.47)
  - M=8192, K=28672, 2-bit: AnyBCQ=742 (×4.00 vs cuBLAS), Any-Precision LLM=971 (×3.06)

  **端到端吞吐量（Llama-3.1-8B, tokens/s, A100）：**
  - 2-bit: AnyBCQ=245 vs Any-Precision LLM=228 vs FP16=105
  - 3-bit: AnyBCQ=212 vs Any-Precision LLM=196 vs FP16=105
  - 4-bit: AnyBCQ=186 vs Any-Precision LLM=169 vs FP16=105

  **Kernel 延迟分解（Any-Precision LLM kernel，Table 7）：**
  - Bit-transpose 占 35-58% 延迟（最大开销）
  - LUT lookup 占 9-17%
  - GEMM + memory 等其余操作占 31-50%
  - AnyBCQ 通过消除 bit-transpose 和 centroid lookup 这两项开销获得加速。
