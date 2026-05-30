## QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QuaRot 通过随机 Hadamard 变换消除 LLM 激活值中的离群值 (outlier)，实现端到端 4-bit 量化（权重、激活值、KV cache）。核心 pipeline 分为两阶段：(1) **Stage 1 权重修改**：利用计算不变性（computational invariance）将随机 Hadamard 矩阵 Q 融入权重矩阵，消除跨层激活值离群值；在 FFN 的 down-projection 前和注意力模块内部插入在线 Hadamard 变换，消除层内激活值离群值；(2) **Stage 2 量化**：用 GPTQ 或 RTN 对权重进行 per-column 对称量化（INT4），激活值在线 per-token 对称量化（INT4），KV cache 用 asymmetric group-wise 量化（group size=128）。最终所有矩阵乘法均在 INT4 下完成，无需保留任何高精度通道或 outlier feature。伪代码核心流程：(a) 生成随机 Hadamard Q = H_d diag(s), s∈{±1}^d → (b) 离线权重修改：W_gate ← Q^T diag(α) W_gate, W_up ← Q^T diag(α) W_up, W_down ← H W_down Q → (c) 在线推理：X_norm = RMSNorm(X) → X_q = round(clip(X_norm/s_x, -7, 7)) → Y_int = CUTLASS_INT4_GEMM(X_q, W_q) → Y = dequant(Y_int, s_x, s_w) → (仅 W_down 前) X_h = Hadamard(X) → X_hq = quant(X_h) → Y_int = GEMM(X_hq, W_down_q) → Y = dequant(...) → YQ（旋转后的输出）。
  - 实验比较：(a) 4-bit 量化后的 WikiText-2 困惑度 vs SmoothQuant, OmniQuant, QUIK-4B, Atom-128G（Table 1）；(b) 零样本任务精度（PIQA, WinoGrande, HellaSwag, Arc-Easy, Arc-Challenge, LAMBADA）vs FP16 baseline（Table 2）；(c) RTN vs GPTQ 权重量化消融（Table 3）；(d) group-wise 量化不同 group size (64/128/256) 消融（Table 4）；(e) KV cache 不同 bit-width 组合消融 (2/3/4-bit for K and V)（Table 6）；(f) Random Orthogonal vs Hadamard 消融（Table 8）；(g) FP16 vs FP32 Hadamard 变换精度消融（Table 10）；(h) 4-bit weight-only 量化消融（Table 7）；(i) LLAMA-3 和 Phi-3-mini 模型扩展实验（Table 11-13）。

- 硬件平台是什么，配置是什么。
  - 量化准备（离线）：单张 NVIDIA A100 GPU。LLAMA2-70B 模型修改耗时 5 分钟，GPTQ 量化耗时 2 小时。校准集：WikiText-2 训练集 128 样本（sequence length=2048）。
  - 性能评估：NVIDIA RTX 3090 GPU。CUDA 12.1。PyTorch 框架 + Hugging Face Transformers。CUTLASS 库做 INT4 TensorCore GEMM。FlashInfer 库做量化 KV cache attention。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLAMA-2 家族（7B/13B/70B），LLAMA-3（8B/70B），Phi-3-mini-4k-instruct
  - 数据集：WikiText-2 训练集用于 GPTQ 校准（128 samples, seq=2048）
  - Benchmarks：WikiText-2 困惑度（语言生成质量）；六项零样本任务——PIQA, WinoGrande, HellaSwag, Arc-Easy, Arc-Challenge, LAMBADA（使用 LM Evaluation Harness 默认参数）
  - 量化配置：GPTQ（默认）或 RTN 权重量化 + per-token 对称激活量化（clipping ratio=0.9）+ asymmetric group-wise KV cache 量化（group size=128, clipping ratio=0.95）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/spcl/QuaRot
  - 算法 pipeline 张量计算流程（LLAMA-2 FFN 层 + Attention）：
    1. **离线权重修改（Stage 1）**：
       - 生成随机 Hadamard: Q = H_d diag(s), s_i ∈ {±1} 随机采样, H_d 为 Walsh-Hadamard 矩阵（O(d log d) 变换）
       - RMSNorm 吸收: α 融合到相邻权重 → W_gate = diag(α) W_gate, W_up = diag(α) W_up
       - 跨层旋转: W_gate ← Q^T W_gate, W_up ← Q^T W_up, W_down ← H W_down Q（H 为精确 Hadamard 矩阵）
       - 注意力 head-wise: W_v ← W_v (I⊗H_{d_h}), W_out ← H (I⊗H_{d_h}) W_out，其中 H = (I⊗H_{d_h})(H_{n_h}⊗I)
       - W_k, W_q 不做离线修改（因 RoPE 存在），改为在线 head-wise Hadamard 旋转
    2. **前向推理（Stage 2）**：
       - RMSNorm（无 scale, FP32）：x_norm = x / ||x||
       - 激活量化（per-token symmetric INT4）：s_x = max(|x_norm|, dim=row) × 0.9 / 7, x_q = round(clip(x_norm/s_x, -7, 7))
       - INT4 GEMM (W_gate/W_up): Y_int32 = x_q × W_q^T（TensorCore, INT32 accumulator）
       - Dequant: Y_fp16 = (Y_int32 ⊙ s_x^T ⊙ s_w) / scale_factor → cast to FP16
       - SiLU gate: Y_gate ⊙ σ(Y_up)（FP16 element-wise）
       - 在线 Hadamard（仅 W_down 前）：Y_h = Walsh-Hadamard(Y_fp16)（O(d log d), FP16 或 FP32）
       - 再次量化 + INT4 GEMM (W_down) + dequant → 旋转后输出 YQ
    3. **Attention 模块（量化 KV cache）**：
       - Q/K/V projection（INT4 GEMM, 同上）
       - Post-RoPE 在线 head-wise Hadamard：Q_h = Q (I⊗H_{d_h}), K_h = K (I⊗H_{d_h})（head-wise Walsh-Hadamard, O(d_h log d_h) per head）
       - KV cache 量化：K_q = round(clip((K_h - z)/s_k, 0, 15))（asymmetric group-wise, group=128）
       - Attention 计算：P = softmax(Q_h K_h^T / √d_h) → Y = P V_h（FlashInfer 实现，online softmax + 反量化）
       - 在线 Hadamard head 变换（out-projection 前）：Z_h = Z (H_{n_h}⊗I)（reshape + Walsh-Hadamard）
       - Out-projection: INT4 GEMM (W_out) + dequant
    4. 使用方法：`python quarot.py --model meta-llama/Llama-2-70b-hf` → Stage 1 权重融合 → GPTQ 量化 → 保存量化模型 → 推理加载
