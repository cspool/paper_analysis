## AFPQ Asymmetric Floating Point Quantization for LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：在 FasterTransformer 框架中实现了低比特 NF4-asym 反量化（dequantization）kernel。W4A16 模式下，低比特权重需要在推理时从 4-bit 反量化到 FP16，再与 FP16 激活值进行计算。具体实现：将两个 4-bit 量化权重打包存储在 1 个 byte 中；反量化时，先通过 LUT（查找表）将 NF4 值转为 FP16 值，再用 scale_pos/scale_neg 进行非对称反量化得到最终 FP16 权重。
  - 实验比较：在 A6000 GPU 上测量 LLaMA2-7B 和 LLaMA2-13B 端到端推理延迟，batch_size=1，输入序列长度 128，输出 20 token，对比 FP16、INT4、NF4-sym、NF4-asym 四种推理系统。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A6000（单卡）。
  - 推理框架：FasterTransformer（https://github.com/NVIDIA/FasterTransformer）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：FasterTransformer，测量端到端延迟（ms）。
  - 修改内容：在 FasterTransformer 中新增 NF4-asym dequantization kernel。该 kernel 实现：
    1. 从 packed byte 中解包两个 4-bit NF4 权重索引。
    2. 通过查找表（LUT）将 NF4 索引映射到对应的 FP16 值。
    3. 使用两组 scale（scale_pos 用于正值、scale_neg 用于负值）对 FP16 值进行非对称反量化。
    4. 将反量化后的 FP16 权重与 FP16 激活执行标准矩阵乘法。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 代码开源：https://github.com/zhangsichengsjtu/AFPQ
  - kernel 执行全过程（以 LLaMA2-7B NF4-asym 推理为例）：
    1. **输入**：packed 4-bit NF4 权重矩阵（每 2 个 4-bit 值占 1 byte），每组 128 个权重对应的 scale_pos 和 scale_neg（FP16 格式）。
    2. **Dequantization 过程**：
       - Step 1 解包：从 byte 中提取高 4-bit 和低 4-bit，分别得到两个 NF4 索引（0-15）。
       - Step 2 LUT 映射：通过预置的 16 项 FP16 LUT `[0, 0.0796, 0.1609, 0.2461, ..., 1]` 的正负版本，将索引转为对应的 FP16 值。正负号由解包时的额外 bit 或独立通道判断（论文未详细说明符号位的具体编码方式）。
       - Step 3 非对称反量化：`w_fp16 = (w_nf4 > 0 ? scale_pos : scale_neg) * |w_nf4|`
    3. **矩阵乘法**：反量化后的 FP16 权重与 FP16 激活值执行标准 GEMM。
    4. **输出**：FP16 格式的输出 activation，传入下一层。
    5. **性能结果**：NF4-asym 在 LLaMA2-7B 上延迟 265.54ms（vs FP16 415.06ms，speedup 1.56x），在 LLaMA2-13B 上延迟 485.42ms（vs FP16 788.01ms，speedup 1.62x）。论文指出 NF4-asym 的 kernel 相比 INT4/NF4-sym 有额外开销，有待后续优化。
