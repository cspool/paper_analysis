## QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：QeRL 论文的核心 kernel 贡献是使用 **Marlin kernel**（Frantar et al., 2024）加速 NVFP4 量化权重的推理。Marlin 是一个面向 auto-regressive 大语言模型推理的 mixed-precision kernel，原生支持 FP16×INT4 操作。QeRL 将 Marlin kernel 适配到 NVFP4×BF16 操作：在 rollout 阶段，policy model 的量化权重以 NVFP4 格式存储（仅 5.9GB for 7B model vs 15.2GB BF16），通过 Marlin kernel 执行快速的 NVFP4 dequant + BF16 matrix multiplication，大幅加速 rollout 生成。AQN 噪声注入被设计为融入 RMSNorm scale 参数，避免破坏 NVFP4×BF16 乘法 kernel 的兼容性，无需额外矩阵乘法开销。kernel 的加速效果随模型规模增大而增强：32B 模型 rollout 达到 2.0× speedup。
  - 实验比较：(a) rollout 吞吐量对比：NVFP4(Marlin) vs BF16 baseline，batch=2/4/8；(b) end-to-end GRPO 训练速度对比：QeRL vs LoRA vs QLoRA；(c) 不同 LoRA rank 下的推理吞吐量；(d) 启用/不启用 gradient checkpointing 的端到端延迟；(e) 模型大小对比（GPU 显存占用：7B 5.9GB vs 15.2GB BF16）。

- 后端平台是什么，配置是什么。
  - NVIDIA H100 80GB GPU。使用 vLLM 引擎进行 rollout。环境：CUDA≥12.4.1，需要支持 NVFP4 的 GPU（如 H100, B100, RTX 5090）。Marlin kernel 依赖 CUDA 的 mixed-precision 矩阵乘法原语。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：vLLM 引擎（用于 rollout 推理吞吐量测量），GRPO/DAPO 训练脚本（端到端速度测量）。论文修改：将 Marlin kernel 适配到 NVFP4 格式（原版 Marlin 针对 INT4/FP16 设计），并在 kernel 前向过程中保持 NVFP4×BF16 操作路径不被 AQN 噪声注入破坏（通过将噪声融入 RMSNorm 而非直接在量化权重上做加法实现）。speedup 测试基于训练前 30 step 的平均速度，rollout throughput 在 batch=1 设置下测试。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/NVlabs/QeRL (Apache 2.0)。Marlin kernel 原版开源：https://github.com/IST-DASLab/marlin。
  - kernel 输入到输出全流程：
    1. **输入**：NVFP4 量化权重 \tilde{W}（4-bit）+ FP32 全局 scale S_FP32 + FP8(E4M3) block-wise scale S_E4M3（block size=16）+ BF16 输入激活 X
    2. **kernel 内**：(a) 从 GPU global memory 加载 4-bit packed weights 到 shared memory；(b) 对每个 block 内的 16 个元素执行 dequant：ŵ = s_FP32 × s_E4M3[block_idx] × dequant_4bit_to_fp16(w̃_block)；(c) 执行 BF16 matrix multiplication: output = X · \hat{W}^T；(d) 加 LoRA adapter 输出: output += X · (BA)^T（可选的 fused LoRA compute）
    3. **输出**：BF16 激活张量
    4. **评估原理**：vLLM 引擎在 rollout 阶段调用 Marlin-accelerated 的 NVFP4 线性层，测量每个 forward pass 的 wall clock time 和 tokens/s。通过固定输入长度=256 tokens、最大生成长度=2048 tokens、控制 vLLM GPU memory utilization 来公平对比 BF16 vs NVFP4 的吞吐量差异。
