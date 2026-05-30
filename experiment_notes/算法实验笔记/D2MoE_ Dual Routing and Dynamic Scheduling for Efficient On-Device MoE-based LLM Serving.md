## D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving

- 属于算法pipeline的实现是什么？实验比较什么？
  实现包括三项算法创新：(1) **Token-Adaptive Bit-Width Selection**：在每层 MoE 的每个 expert 前放置一个轻量级可训练的 bit-width router，动态选择每个 token 激活的 expert 的最优 bit-width。通过 Quantized Expert Capacity（{c_k}_{k=1}^K，预定义各 bit-width 的 token 容量上限，超限 token 随机丢弃）和 Dynamic Bit-Width Selection Loss（CE loss + α/L·Σp_k·b_k 正则化项，平衡 accuracy 和 bit-width 选择频率）实现动态 bit-width 分配。(2) **Matryoshka Weight Quantization (MWQ)**：首先用非对称量化 (asymmetric quantization) 将 expert 权重量化到最低 bit-width b_1（如 INT2），保留 block-level compensation（类似 GPTQ）；再用 binary residual quantization 对残差权重 R_{b_{k-1}} 逐次递增加 1-bit（±1 权重），使高 bit-width 嵌套低 bit-width，实现 INT2⊂INT3⊂INT4 的套娃结构，避免存储多份量化版本。(3) **Token-Expert-Weight Three-Level Co-Design**：将专家选择从"仅选 expert ID"扩展为"同时选 expert ID 和 bit-width"的 dual routing 范式（Figure 1），在算法层面实现精度-内存-延迟三目标联合优化。

  实验比较：(a) D2MoE-V1 (b_1=2, b_K=4) vs Hold-in-Memory (INT8, 全量 GPU 内存)、Matryoshka-Free (GPTQ INT2/3/4 独立存储+按需加载)、Hold-in-Memory-AWQ (INT4, 全量 GPU)、EdgeMoE (离线 profiling 固定 bit-width)、MoQE-DynaIO (统一 bit-width + 按需加载)，在 LLaMA-MoE-3.5B 和 Mixtral 8×7B 上的 PPL 和 5 个 zero-shot benchmarks (PIQA/ARC.e/BoolQ/HellaSwag/Winogrande)；(b) D2MoE 扩展到 dense LLM (LLaMA2-13B) 对比固定 INT4 GPTQ 的吞吐和峰值内存。

- 硬件平台是什么，配置是什么。
  离线预处理阶段（bit-width router 微调 + MWQ）：GPU server 配备 NVIDIA RTX 2×A6000。在线推理阶段：Environment 1 — NVIDIA RTX 3060 (6GB GPU memory) + Intel Core i7-11800H (32GB CPU) + Samsung 970 EVO (1TB, 3.5GB/s disk read)；Environment 2 — NVIDIA Jetson AGX Orin 64GB (SoC GPU) + ARM Cortex-A78AE + Samsung 970 EVO (1TB)。单卡推理，非分布式。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-MoE-3.5B（8 experts/layer, Top-2 routing）和 Mixtral 8×7B（8 experts/layer, Top-2 routing），均为 decoder-only MoE-based sparse LLMs。Dense 实验使用 LLaMA2-13B。训练数据：C4 数据集，2048 random 2048-token segments 用于训练 bit-width routers，128 random 2048-token segments 用于 MWQ calibration。Benchmarks：WikiText2 (PPL)、PIQA、ARC.e (ARC-Easy/Challenge)、BoolQ、HellaSwag、Winogrande（使用 lm-evaluation-harness）。吞吐评估：input/output length 均为 128。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供公开开源代码仓库。论文提及实现约 2,500+ LOC Python + CUDA，基于 PyTorch、Triton（I/O-compute 并行编程）和 CUDA（NVIDIA Ampere/Ada Lovelace 架构）。MWQ 算法流程：

  ```
  === Token-Adaptive Bit-Width Selection (Online Inference) ===
  Input: token x, layer l

  Step 1: Original MoE routing (select top-K experts)
    gate_logits = W_gate @ x                   // [E] expert selection logits
    topk_indices = TopK(Softmax(gate_logits), K=2)  // select 2 experts
  
  Step 2: Bit-width routing (per selected expert)
    For each selected expert e_i:
      bw_logits = W_bw_router[e_i] @ x        // [K_bw] bit-width logits
      bw_probs = Softmax(bw_logits)            // e.g., [INT2_prob, INT3_prob, INT4_prob]
      b_k = argmax(bw_probs)                   // selected bit-width
      // Loss: L = CE(p(x), q(x)) + α/L * Σ_l Σ_k p_k^l(x) * b_k
      // Second term pushes towards lower bit-width to save memory
  
  Output: [(expert_id_1, bit_width_1), (expert_id_2, bit_width_2)]

  === MWQ Quantization (Offline, per expert weight matrix W) ===
  Input: W ∈ R^{s×h} (FP16), candidate bit-widths {b_1, b_2, ..., b_K} e.g., {2, 3, 4}

  # Step 1: Asymmetric quantization to b_1 (e.g., INT2)
  Q_W_b1 = round(W / s_b1 + z_b1)              // [s×h] quantized to b_1 bits
  W_hat_b1 = (Q_W_b1 - z_b1) * s_b1            // dequantized approximation
  # Optimize s_b1, z_b1 via: argmin ||WX - W_hat_b1 X||_2^2

  # Step 2: Binary residual quantization for b_2...b_K
  R_b1 = W - W_hat_b1                           // residual
  For k = 2 to K:
    Q_W_bk = round(R_{b_{k-1}} / s_bk)         // binary residual (±1 values) 
    W_hat_bk = W_hat_b1 + Σ_{i=2}^{k} s_bi * Q_W_bi  // nested reconstruction
    R_bk = R_{b_{k-1}} - s_bk * Q_W_bk
    # Optimize s_bk via: argmin ||WX - W_hat_bk X||_2^2

  # Key property: W_hat_b2 ⊂ W_hat_b3 ⊂ ... ⊂ W_hat_bK (nested)
  # Storage: W_hat_b4 = W_hat_b2 + s_b3*Q_W_b3 + s_b4*Q_W_b4
  # vs traditional: need separate INT2, INT3, INT4 weights (3× storage)

  Output: {Q_W_{b_i}}_{i=1}^K (nested quantized weights)

  === MWQ Dequantization (Online, to get b_k-bit weight) ===
  Input: {Q_W_{b_1}, ..., Q_W_{b_k}}, {s_{b_1}, ..., s_{b_k}}, {z_{b_1}}
  
  W_fp16 = (Q_W_b1 - z_b1) * s_b1              // base b_1 weight to FP16
  For i = 2 to k:
    W_fp16 += s_{b_i} * Q_W_{b_i}               // accumulate binary residuals
  # W_fp16 is now equivalent to a b_k-bit quantized weight

  === Bit-width router training loss ===
  For batch S with T tokens, L layers, K bit-widths:
    L_CE = (1/T) * Σ_x CE(p(x), q(x))          // cross-entropy with FP16 teacher
    L_reg = (α/L) * Σ_l Σ_k p_k^l(x) * b_k     // regularization favoring low bit-width
    L_total = L_CE + L_reg
  # p_k^l(x): probability fraction for k-th bit-width at layer l
  # α controls accuracy-efficiency trade-off
  ```

  Quantized Expert Capacity：每个 bit-width expert 的 token 容量上限为 c_k·T（Σc_k=1），如 D2MoE-V1 中 {0.3, 0.4, 0.3} 对应 INT2/3/4。超限 token 随机丢弃，防止 fine-tuning 时 bit-width router 过拟合特定 token 序列。
