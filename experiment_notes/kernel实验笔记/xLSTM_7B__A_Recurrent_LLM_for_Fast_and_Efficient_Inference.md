## xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Fused Generation Kernels for mLSTM Cell。在自回归生成时，mLSTM cell 的 recurrent 公式（Eq. 1-9）包含 outer-product、多个 dot-product 和逐点操作（pointwise operations），在标准实现中会分解为多个独立的 GPU kernel 调用。每个 kernel 调用都需要从 GPU HBM 加载输入并存储输出，增加了慢速内存操作的占比。论文开发了 fused GPU kernels（Triton 编写），将这些中间结果保持在 GPU 的 compute chip（SRAM/register）上，避免不必要地传输到 GPU 内存（HBM）。此外，论文的 chunkwise-parallel 训练 kernel 基于 FlashLinearAttention 技术（Yang et al., 2024b），在训练时对序列分块并行处理。
  - 实验比较：在 Fig. 4-7 中通过整体推理速度 benchmark 间接评估 kernel 效果。xLSTM 7B 比 Falcon-Mamba（Mamba 1）和 Codestral-Mamba（Mamba 2）快约 50% 的生成吞吐，在 prefill 长度 0 时甚至快于 Llama 系列 Transformer；在 65536 token prefill 吞吐测试中，xLSTM 7B 比 Codestral Mamba 高约 70%。论文未对单个 fused kernel 进行独立的 micro-benchmark。

- 后端平台是什么，配置是什么。
  - GPU: 单 NVIDIA H100 GPU
  - 推理精度：论文未明确说明（从 HuggingFace model card 可推断为 bfloat16）
  - 推理框架：HuggingFace transformers + torch.compile + PyTorch CUDA Graphs
  - 训练：128× NVIDIA H100 GPU，FSDP + activation checkpointing

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估脚本：使用 HuggingFace transformers 库中的各模型实现，统一用 torch.compile 和 PyTorch CUDA Graphs 优化后测量
  - Kernel 代码开源在 https://github.com/NX-AI/mlstm_kernels（Triton-based）
  - 修改内容：将 mLSTM cell 在生成模式下的多个独立 GPU kernel（outer-product, dot-product, pointwise ops）融合为单个 fused kernel，减少 HBM 读写次数；训练时使用 chunkwise-parallel kernel 替代 naive 实现
  - 对比对象：Llama-2-7B、Llama-3.1-8B（attention-based）、Falcon-Mamba-7B（Mamba 1 architecture）、Codestral-Mamba-7B（Mamba 2 architecture）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - Triton fused kernel 开源：https://github.com/NX-AI/mlstm_kernels
  - 模型实现开源：https://github.com/NX-AI/xlstm（PyTorch）和 https://github.com/NX-AI/xlstm-jax（JAX）
  - **Fused Generation Kernel 原理**：
    1. **输入**：当前 token 的 input vector x_t ∈ R^d，上一时间步的 recurrent state (C_{t-1}, n_{t-1}, m_{t-1}, h_{t-1})
    2. **Kernel 执行流程（单次 fused kernel 调用）**：
       - Gate computation: 计算 i_tilde, f_tilde（scalar soft-capped pre-activations）和 o_tilde（vector output gate pre-activation）
       - Max state update: m_t = max(log(σ(f_tilde)) + m_{t-1}, i_tilde)
       - Gate activation: f_t = exp(log(σ(f_tilde)) + m_{t-1} - m_t), i_t = exp(i_tilde - m_t)
       - Memory Update (in SRAM): C_t = f_t * C_{t-1} + i_t * (k_t^T v_t)（outer product 在片上计算）
       - Normalizer Update: n_t = f_t * n_{t-1} + i_t * k_t
       - Hidden State Retrieval: h_tilde = C_t^T @ q_norm / max(|n_t^T @ q_norm|, exp(-m_t))
       - Output: h_t = o_t ⊙ Norm(h_tilde)
    3. **输出**：当前时间步的 hidden state h_t 和更新后的 recurrent state (C_t, n_t, m_t)
    4. **关键优化**：所有中间结果（k_t, v_t, q_t, gate values, C_t 更新中的 outer product 结果）都在 GPU SM 上的 SRAM/Register file 中保持，不写回 HBM。只有最终的 h_t 和 state 写回。由于 mLSTM 不使用 softmax attention，没有 QK^T 矩阵的全序列计算，每次 recurrent step 的 FLOPs 恒定。
  - **评估原理**：
    1. 使用 HuggingFace transformers 加载各模型
    2. 用 torch.compile 对模型计算图进行 JIT 编译优化
    3. 用 PyTorch CUDA Graphs 捕获重复的推理步骤，消除 kernel launch overhead
    4. 在单 H100 GPU 上，batch size 1，测量：(a) 在不同 prefill 长度（0 到 128K tokens）下生成 100 token 的吞吐（tokens/sec）；(b) 在不同生成长度下的生成时间和 GPU 内存占用；(c) Time To First Token 延迟；(d) 在 65536 token 下不同 batch size 和 context length 的 prefill 吞吐
