## Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Skywork-MoE 的训练基础设施基于内部开发的 Skywork-Megatron 框架（基于 Megatron-LM 23.06 分支），核心 kernel/运行时调度创新包括：
    (1) **Expert Data Parallelism (EDP)**：自定义并行策略，定义为 Size_EP = Size_TP。在注意力层以 Tensor Parallelism 方式运行，在 FFN/MoE 层以 Expert Parallelism 方式运行，同一数据同时穿越 TP Group 和 EP Group。相比 Megatron-LM Core 0.6.0 的 EP（Size_EP = Size_DP * Size_TP，受 expert 数量上限限制）和 ETP（Size_EP = Size_DP，AllToAll 通信开销随 TP 增大），EDP 对中等 expert 数量（≤64）的模型优化了门控层 token 路由的 AllToAll 通信。
    (2) **Unbalanced Pipeline Parallelism**：打破均匀层分割（如 [6,6,6,6]），采用非均匀分割（如 [5,5,5,5,4]）减少 pipeline bubble time 达 10%。梯度重计算（checkpointing）也按 stage 差异化配置，平衡各 stage 的内存使用和计算开销。
    (3) **通信优化**：实现了 expert parallelism 相关通信缩减、kernel fusion、通信与计算重叠等优化，最终达到 38% MFU 和 690 tokens/GPU/sec。
  - 实验比较：(a) Uniform vs Non-uniform PP bubble time 对比（图 6），24 层 Transformer 模型 pipeline bubble time 减少约 10%；(b) 训练吞吐量：38% MFU on 1536 A800 GPUs, 690 tokens/GPU/sec；(c) EDP 与 EP/ETP 的理论对比分析（通信开销、扩展性约束）。

- 后端平台是什么，配置是什么。
  - GPU：192 节点 × 8 × NVIDIA A800-80G SXM = 1536 GPUs
  - 节点内互联：400 GB/s NVLink
  - 节点间互联：800 Gb/s RoCE 网络
  - 并行配置：12-way pipeline parallelism + 4-way tensor-expert parallelism (EDP) + 32-way data parallelism + ZeRO-1
  - 设备 mesh：Attention weights 为 [Size_PP, Size_DP, Size_TP]，Expert weights 为 [Size_PP, Size_DP, Size_EP]

- 评估性能的软件/脚本是什么。修改了什么。
  - 基础框架：Megatron-LM 23.06 分支
  - 内部框架名：Skywork-Megatron
  - 修改内容：(a) 实现自定义 MoE 架构（门控层、expert 层、tailored distributed parallel strategy）；(b) 实现 EDP 并行策略（设备 mesh 动态切换 Attention TP ↔ Expert EP）；(c) 实现 Unbalanced PP（非均匀层分割 + 差异化的梯度重计算配置）；(d) expert parallelism 通信缩减、kernel fusion、通信-计算 overlap 优化。
  - 训练配置：学习率采用多阶段调度（unique learning rate schedule per stage），cosine decay。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：GitHub https://github.com/SkyworkAI/Skywork-MoE，训练代码基于 Megatron-LM，包含自定义 MoE 架构实现。重点关注 `megatron/model/moe/` 目录下的 expert 并行与门控实现。
  - 训练框架全流程（Skywork-Megatron, 1536 A800 GPUs, Skywork-MoE 146B/16 experts）：
    ```
    输入：tokenized text batch (micro_batch_size, seq_len=8192)
    
    1. Data Loading & DP Sharding
       - 32-way data parallelism，每个 DP group 处理不同的 micro-batch
       - ZeRO-1 分布 optimizer state
    
    2. Pipeline Stage Partitioning (12 PP stages, Unbalanced)
       - 52 层非均匀分割到 12 个 PP stage
       - 最后 stage 少 1 层以补偿 loss calculation 的计算开销
       - 梯度重计算按 stage 差异化（buffer 大的 stage 少存 activations）
    
    3. Attention Layer (TP Group)
       - Device Mesh: [Size_PP, Size_DP, Size_TP=4]
       - Self-Attention: QKV projection → RoPE → Flash Attention → Output projection
       - TP 切分 head 维度，每 GPU 处理 36/4=9 heads
    
    4. MoE Layer (EP Group via EDP)
       - Device Mesh 切换: [Size_PP, Size_DP, Size_EP=4]
       - Gating: 4 GPUs × 4 experts each = 16 experts total
         - Gate forward + Logit Normalization (z_tilde = λ*(z-μ)/σ)
         - Softmax + Top-2 selection
       - Token Dispatch: AllToAll 通信在 4 EP GPUs 间路由 tokens 到目标 expert
       - Expert FFN (SwiGLU): 每 expert 独立计算 FFN(W_gate, W_up, W_down)
       - Token Combine: AllToAll 通信将 expert 输出送回原 GPU
       - Weighted sum: y_i = (g1*E1(x_i) + g2*E2(x_i)) / (g1+g2)
    
    5. Communication Optimizations
       - Expert parallelism AllToAll 通信缩减
       - Kernel fusion (如 gate + dispatch 融合)
       - 通信与 expert FFN 计算 overlap
    
    6. Loss & Backward
       - Cross-entropy loss + Σ α^(l) * L_aux^(l) (52 adaptive coefficients)
       - Backward through MoE (gradient through gating + expert FFN)
       - ZeRO-1 all-reduce gradients across 32 DP ranks
    
    输出：更新后的模型参数，throughput = 690 tokens/GPU/sec, MFU = 38%
    ```
