## DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction

- baseline方法是什么？
  Baseline 有两个层面：(1) **Standard MoE Inference without Computation Dropping**：预训练 MoE 模型的 naive 推理，每 token 在每层激活 Top-K 个 experts，所有 activated experts 的 FFN 计算全部执行，不做任何 token-expert 计算丢弃。Expert 按照预训练时的粒度（如 Mixtral-8×7B 的 8 experts, Top-2）运行。在分布式场景下使用标准 EP + ETP（Expert-Tensor Parallelism）部署，通信模式为 "AlltoAll+AllGather" 或 "ReduceScatter+AlltoAll"，存在多轮 kernel launch 和同步开销。(2) **Prior Sparsity-based Acceleration Methods**：EES (Efficient Expert Skipping) 根据第二高 gating score 与第一高 score 的比值动态跳过 expert 计算，但 accuracy degradation 显著（GSM8K -2.4%）；EEP (Efficient Expert Pruning) 静态剪枝不常用 experts 实现模型压缩，但 accuracy loss 大（r=6: -8.0%, r=4: -25.9%）；Wanda 等权重剪枝方法在 2:4 sparsity 下 GSM8K accuracy 下降 50.7%。这些方法的共同缺陷：(a) 以动态 tensor-level sparsity 换静态压缩，破坏 MoE 的动态路由优势；(b) 高 drop/prune rate 下 accuracy 剧烈下降；(c) 细粒度 neuron-level sparsity 难以在现有 GPU hardware/kernel 设计上翻译为实际 speedup；(d) 未利用 neuron-level 激活稀疏性（SwiGLU FFN 中大量 neuron 的 gating score × activation 乘积接近零但不为零）；(e) 未考虑 EP 分布式推理中的 load imbalance 问题。

  **Baseline 全栈执行例子（以 Mixtral-8×7B, 8×H20, TP=8, 推理 batch of tokens 为例）**：
  - **算法层**: Standard MoE with 8 experts, Top-2 gating, SwiGLU FFN experts (d_ffn=14336), 32 decoder layers with MoE layers alternating with attention. 每 token 激活 2/8 experts，所有激活的 expert FFN 计算完整执行。
  - **系统框架层**: SGLang framework，使用 TP=8 做 tensor parallelism for non-expert layers，EP 的通信使用标准 ETP 模式（AlltoAll + AllGather）。无 token-expert computation dropping，无 load-aware thresholding。
  - **编译框架层**: 论文未明确说明（SGLang Python/Triton-based execution）。
  - **Kernel调度层**: 标准 Triton grouped-GEMM kernel for expert computation。所有 activated experts 使用完整权重矩阵 W₁, W₂, W₃ 计算。Gating 函数使用标准 top-k + softmax。
  - **硬件架构层**: 8× NVIDIA H20 GPU，单节点 NVLink/NVSwitch 互联。每个 token 的 MoE 计算量 = 2×(3×d_model×d_ffn) FLOPs = 2×3×4096×14336 FLOPs。每个 EP device 负载不均，总推理时间由最繁忙 device 决定。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **DualSparse-MoE**，核心创新是协调利用 MoE 架构中天然存在的 **双重稀疏性**（tensor-level + neuron-level），并通过 expert partition 在 post-training 阶段增强 tensor-level sparsity 而无需重训练。

  **(1) Expert Partition（解决baseline中专家粒度固定问题）**：Baseline 中 MoE 的 expert granularity 在 pre-training 时确定，部署时无法改变。论文提出两种 post-training expert 划分方法，保持数学一致性：(a) Complete Transformation 将 E experts 划分为 E×P 个 finer-grained experts，提高 fine-tuning accuracy（Mixtral 8→32 experts: fine-tuning loss 降低，downstream accuracy +0.59%）；(b) Partial Transformation 保持 gating network 不变，仅重映射 expert indices，支持 S-ETP 优化。

  **(2) 2T-Drop with Neuron Reconstruction（解决baseline中单一阈值丢弃的精度损失）**：Baseline 的简单 token-expert dropping 面临显著 accuracy degradation。论文的解决方案：(a) Static Neuron Importance Profiling：在 calibration samples 上对每个 expert 内 neuron 做 importance profiling（四种方法：accumulated gate、abs gate、gate-up、abs gate-up），按重要性排序后重构为 major（高重要性）和 minor（低重要性）sub-expert；(b) Dual-Threshold Dropping：对 major sub-expert 使用较低阈值 T²_major（保守保留），对 minor sub-expert 使用较高阈值 T²_minor（激进丢弃），gating score 在 dual-threshold 之间的 experts 仅计算 major half neurons。结果：~25% drop rate 下仅 loss 0.08%-0.28% average accuracy（Mixtral: 71.12→71.04, OLMoE: 65.91→65.63, DeepSeek: 67.83→67.65）。

  **(3) Load-Aware Thresholding（解决baseline中EP负载不均问题）**：Baseline 的 EP 推理中，所有 device 使用相同 drop threshold，但不同 device 负载差异大，均匀丢弃在 overloaded device 上加速不够、underloaded device 上精度损失不必要。论文方案：每个 device 根据 actual_load / ideal_load 比值动态调整 threshold，overloaded device 用高 threshold（激进丢弃）、underloaded device 用低 threshold（保守保留），以最小精度损失实现负载均衡。结果：load-aware 2T-Drop → 1.41× MoE module speedup, 1.13× end-to-end speedup，仅 0.5% average accuracy loss。

  **(4) S-ETP（解决baseline中ETP通信复杂问题）**：Baseline 的 ETP 使用 "AlltoAll+AllGather" 或 "ReduceScatter+AlltoAll" 多轮通信，引入额外 kernel launch 和同步开销。S-ETP 通过 partial transformation 将 TP 职责转移到算法层面，仅需单次 AlltoAll 通信。结果：real H20 带宽提升 3.0%-29.9%，NVL72 模拟提升 10.2%-80.4%。

  **DualSparse-MoE 方法全栈执行例子（以 Mixtral-8×7B, P=4 (32 experts), 8×H20 TP=8, 推理一个 batch of tokens, ~25% drop rate 为例）**：
  - **算法pipeline层**: Preprocessing: (a) partial transformation: 8→32 experts, P=4, Top-2→Top-8；(b) neuron importance profiling on MMLU calibration, 按 accumulated absolute gate value 排序，每个原 expert 重构为 major (top 50%) + minor (bottom 50%) sub-expert。Inference: 每 token 计算 gating scores → normalize → 对每个 activated expert 判断 normalized score 与 dual thresholds (T²_major=0.07, T²_minor=0.09) 的关系 → 决定 skip/major-only/full 计算 → ~24% token-expert pairs dropped。
  - **系统框架层**: SGLang framework + DualSparse-MoE modifications。Preprocessing 完成 expert partition + neuron reconstruction。Inference 时 gating function 融合 dual-threshold decision logic。通信使用标准 AlltoAll（可选择 S-ETP 简化模式）。如启用 load-aware thresholding：gather 各 device 负载 ratio → 动态调整各 device 的 drop threshold。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 优化的 Triton grouped-GEMM kernel 支持变粒度计算模式（skip/major-only/full）。Token-expert dispatch 根据 dual-threshold decision 将 token 分组到不同的 GEMM kernel 调用。2T-Drop 的细粒度计算与 1T-Drop 粗粒度计算实现相近的实际 speedup（因为优化避免了额外 kernel launch）。
  - **硬件架构层**: 8× NVIDIA H20 GPU, NVLink intra-node。~24% drop rate → 1.17-1.23× MoE module speedup, 1.07-1.12× end-to-end speedup。Tensor-level 丢弃粒度天然适配 GPU grouped-GEMM，区别于 neuron-level sparsity 需要专用 hardware/kernel 才能在低 drop rate 下实现 speedup。
