## QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QWHA 提出一种基于 Walsh-Hadamard Transform (WHT) 的量化感知 PEFT 适配器（WHA），配合量化感知初始化方案（AdaAlloc + Refinement）。核心设计分为三部分：
    (1) **WHA 适配器设计**：权重更新公式 ΔW = F H^{-1}，其中 H 是固定的 WHT 矩阵（±1 元素，通过 Kronecker 积 H_N = H_2 ⊗ H_{2^{n-1}} 递归构造），F = Scatter(c, E) 是可训练的稀疏系数矩阵。WHA 相比 LoRA 具有 full-rank 表示能力（rank ≈ min(d_in, d_out)），仅使用单个变换（而非传统 FT-based adapter 的双变换），且 WHT 的方形波基函数相比 DCT/DHT 的正弦基函数更擅长捕获量化误差的异常值（outlier），能集中更多能量在少量系数中（Pareto hill index η 最小）。
    (2) **AdaAlloc 参数选择**：按输出通道的激活误差大小按比例分配参数预算 p_i = p · ||(ΔW_Q X)_{i,:}||_F^t / Σ_j ||(ΔW_Q X)_{j,:}||_F^t，保证每个通道至少分配若干参数（维持 full rank），同时在每个通道内选择 |v B^{-1}| 最大的位置（即 |(ΔW_Q H)_{i,j}| 最大的系数），既减少量化误差又维持 fine-tuning 能力。
    (3) **Refinement 值精化**：对已选参数位置，通过最小二乘法重新投影优化参数值：x^* = v B'^T (B' B'^T)^{-1}，其中 v = (ΔW_Q)_{i,:} R, B = H^{-1} R, R = U Σ^{1/2}（XX^T 的矩阵平方根），B' 由选中索引对应的 B 行组成。此步骤使选中的 basis vectors 能补偿未选中向量的影响，大幅降低层输出误差（例如 Key 投影：Refinement 前 0.62 → 后 0.27，缩小 2.3x）。
  - 实验比较：(1) Main evaluation: QWHA vs CLoQ (LoRA-based QA-PEFT), SHiRA (sparse adapter), LoCA (DCA-based), SSH (DHA-based), GPTQ_MagR (quantized only) — 在 LLaMA-3.1-8B / LLaMA-3.2-3B / Mistral-7B-v0.3 上，4/3/2-bit 量化，CSQA 和 GSM8k benchmark；(2) Ablation on adapter type: WHA vs DCA vs DHA vs Sparse（均使用 AdaAlloc + Refinement）；(3) Ablation on parameter selection: Random vs Magnitude vs LoCA vs SSH vs AdaAlloc（均使用 WHA + Refinement）；(4) QWHA vs CLoQ 的 accuracy vs parameter budget 曲线（Figure 6）；(5) 训练效率：各方法在 Alpaca 训练时间对比（batch size 1/2/4/8/16），WHT 1D vs 2D 训练时间对比；(6) 消融：温度 t (0.25/0.5/1.0/1.5/2.0)，量化 group size (32/64/128/256)。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100 80GB GPU。训练批次大小 4-16（GSM8K 用 batch=4, Alpaca 用 batch=6）。校准集：128 条 WikiText-2 序列（长度 2048），用于量化和适配器初始化的 calibration。PyTorch 框架 + AdamW optimizer。量化方案：GPTQ + MagR，group size 64，适配器应用于所有线性层（q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaMA-3.1-8B（32 layers, d=4096），LLaMA-3.2-3B（28 layers, d=3072），Mistral-7B-v0.3（32 layers, d=4096）。
  - 数据集（微调）：Stanford-Alpaca（52k 指令微调样本）；GSM8k（数学推理训练集）。
  - Benchmark（评估）：
    - CSQA (CommonsenseQA, Zero-shot)：覆盖 7 个多选题基准——ARC-Challenge, ARC-Easy, BoolQ, HellaSwag, OpenBookQA, PiQA, WinoGrande。使用 lm-evaluation-harness 评测。
    - GSM8k (Zero-shot CoT)：算术推理，测试集 zero-shot chain-of-thought。
  - 量化校准集：WikiText-2，128 条序列 × 2048 token。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://github.com/vantaa89/qwha
  - 软件依赖：PyTorch, Transformers (HuggingFace), fast-hadamard-transform (Dao-AILab, https://github.com/Dao-AILab/fast-hadamard-transform), GPTQ (Frantar et al., 2023)
  - 算法 pipeline 核心流程（QWHA, 4-bit 量化, LLaMA-3.2-3B, P(r=64)）：
    1. **量化阶段**：
       - 使用 GPTQ + MagR 对预训练权重 W_0 做 4-bit 量化，group size 64：
         ```
         s = max(|W_group|) / (2^(b-1) - 1)  # per-group quantization scale
         W_Q_tilde = clamp(round(W_0 / s) - z, 0, 2^b - 1)  # INT4
         W_Q = (W_Q_tilde + z) * s  # dequantized float
         ΔW_Q = W_0 - W_Q  # quantization error
         ```
    2. **Calibration 阶段**（收集激活统计，用于初始化）：
       - 128 条 WikiText-2 序列前向传播，收集各层激活 X
       - 计算 Hessian 平方根：XX^T = U Σ U^T → R = U Σ^{1/2}
       - 预计算 B = H^{-1} R（H 为 WHT 矩阵，通过 fast-hadamard-transform kernel 计算）
    3. **QWHA 初始化阶段（Algorithm 1）**：
       ```
       for each layer i with linear weight W_Q:
         # Step 1: AdaAlloc - 通道级参数分配
         for each output channel j in 0..d_out-1:
           error_j = ||(ΔW_Q X)_{j,:}||_F  # 各通道的输出误差
         p_j = floor(p * error_j^t / sum_k(error_k^t))  # t=1 默认
         余数分配给最小分配的通道，确保 sum(p_j) = p
         
         # Step 2: Per-channel parameter selection
         for each output channel j:
           v = (ΔW_Q)_{j,:} R  # 投影的量化误差
           dense_sol = v B^{-1} = (ΔW_Q H)_{j,:}  # 稠密解
           # 选取 |dense_sol| 最大的 p_j 个位置作为 E_j
           E_j = TopK_pj_Index(|dense_sol|)
           
           # Step 3: Value Refinement
           B' = B[E_j, :]  # 选中行
           c_j = v B'^T (B' B'^T)^{-1}  # 最小二乘精化
         
         # 构建稀疏矩阵 F
         F = Scatter(c, E)  # F[E[l,0], E[l,1]] = c[l]
    4. **Fine-tuning 阶段**：
       ```
       for each training step:
         for each linear layer:
           # WHA 前向传播
           ΔW = F H^{-1}  # F 为稀疏，H^{-1} 通过 fast Hadamard kernel
           Y = (W_Q + α * ΔW) X  # α_effective ≈ 1.0 (α_explicit=4000/d_in)
         # 反向传播：仅更新系数 c（F 中的非零值），E 和 H 固定
         # Loss = cross_entropy(logits, labels)
         # Optimizer: AdamW, lr = 3e-5 (LLaMA-3.2-3B, 4-bit, Alpaca)
       ```
    5. **推理阶段**：
       - WHA 适配器的额外推理：ΔW X = F (H^{-1} X)，通过 fast Hadamard kernel 实现，仅用加法和减法（无矩阵乘法）
       - 合并：Y = W_Q X + ΔW X。
       - 推理吞吐：184.6 tokens/sec (batch=128, prefill=2048, gen=64)，仅比 LoRA (188.1) 低 1.9%，远优于 DCA/DHA (92.4 tokens/sec, 下降 50.9%)
