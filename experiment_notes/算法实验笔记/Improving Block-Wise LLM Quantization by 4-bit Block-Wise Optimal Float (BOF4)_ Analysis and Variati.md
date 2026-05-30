## Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4): Analysis and Variations

- 属于算法pipeline的实现是什么？实验比较什么？
  - **BOF4**：基于改进 Lloyd's EM 算法，最小化 block-wise absmax 量化后网络权重的端到端 MSE 或 MAE，计算信息论最优的 4-bit 量化码本（codebook）。关键创新在于 centroid 更新公式考虑了 block maximum 的分布权重（MSE 用 `w_b^max` 的平方加权，MAE 用 `w_b^max` 加权），而非直接最小化归一化权重的量化误差。
  - **BOF4-S**：将 block-wise absmax normalization 改为 signed absmax normalization，即归一化常数取 signed absolute block maximum（保持符号而非取绝对值）。这使得归一化后只需固定 2 个 reconstruction level（0 和 1），而非 3 个（-1, 0, 1），降低量化误差。
  - **OPQ（Outlier-Preserving Quantization）**：混合精度策略，将 outlier weights 以 bfloat16 + 64-bit 位置索引单独存储。Outlier 判定基于 `|w_{b,i}| > σ_b * F_M^{-1}(q)`，其中 σ_b 为 block 内标准差，F_M^{-1}(q) 为绝对 block maxima 分布的 q-分位数（q=0.95）。Outlier 在量化前替换为 0，改善归一化后权重分布与理论分布的吻合度。
  - 实验比较：与 NF4（QLoRA 中的 NormalFloat）和 AF4（AbnormalFloat）比较以下指标：
    - (a) 合成数据（Gaussian weights）的 MAE/MSE 随 block size I 变化曲线
    - (b) 真实 LLM 权重的 MAE、MSE、perplexity（WikiText-2, LAMBADA）
    - (c) 下游任务准确率（MMLU, ARC-Challenge, HellaSwag, PIQA, SIQA, WinoGrande）
    - (d) QLoRA 微调后指令跟随（IFEval）和代码生成（HumanEval+, MBPP+）的准确率

- 硬件平台是什么，配置是什么。
  - 微调：1× NVIDIA A100 40GB，每轮 < 8 小时
  - 评估：NVIDIA RTX 3080 10GB 或 A100 40GB
  - 推理运行时测试：NVIDIA RTX 4070 Ti Super

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-3.1（3B, 8B）、Llama-3.2（3B, 8B）、Qwen-2.5（0.5B, 3B, 7B）、Mistral-7B-v0.3
  - 数据集：WikiText-2（perplexity，rolling log-likelihood，max seq len 2048）、LAMBADA（perplexity）
  - Benchmark：MMLU（few-shot）、ARC-Challenge、HellaSwag、PIQA、SIQA、WinoGrande
  - 微调数据集：Unnatural Instructions（指令跟随）、Magicoder-OSS-Instruct-75K（代码生成）
  - 超参数：AdamW optimizer, lr=4e-5, β1=0.9, β2=0.999, batch size=16, 1875 steps, max_grad_norm=0.3, LoRA dropout=0.1, 不做 double quantization

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/ifnspaml/bof4
  - **Block-wise absmax 量化流程**：
    ```
    # Step 1: Block partitioning
    W_flat = W.reshape(-1)  # flatten weight tensor
    blocks = W_flat.reshape(B, I)  # B blocks, each size I
    
    # Step 2: Block-wise normalization
    for b in 1..B:
        w_max[b] = max_i |blocks[b, i]|           # absmax (BOF4)
        # 或 w_max[b] = blocks[b, argmax_i |blocks[b,i]|]  # signed (BOF4-S)
        x[b, i] = blocks[b, i] / w_max[b]          # normalize to [-1, 1]
    
    # Step 3: Scalar quantization
    for b in 1..B:
        for i in 1..I:
            l = argmin_ℓ |x[b,i] - x̂(ℓ)|           # nearest codebook entry
            Ŵ[b, i] = w_max[b] * x̂(l)              # decode
    ```
  - **BOF4 EM 算法（MSE 优化，Monte-Carlo 方法）**：
    ```
    # Initialize codebook x̂(1..16), fix x̂(8)=0, x̂(1)=-1, x̂(16)=1
    # Sample W ~ N(0,1) with shape B×I
    repeat until convergence:
        # 1. Assignment (nearest neighbor)
        for each block b, weight i:
            x = w[b,i] / w_max[b]
            ℓ = argmin_j |x - x̂(j)|
            assign x to region R_ℓ
        
        # 2. Centroid update (MSE) - modified from standard Lloyd's
        for ℓ in 2..15 (skip fixed levels):
            # Collect all x_k in R_ℓ across all blocks, with w_k = w_max of their block
            # Weighted mean (Eq. 6):
            x̂(ℓ) = Σ_k (w_k² * x_k) / Σ_k (w_k²)
    ```
  - **BOF4 EM 算法（MAE 优化）**：centroid 改为 weighted median（Eq. 8）：`x̂(ℓ) = median_W(x_1..x_K; w_1..w_K)`，即加权绝对偏差最小的点。
  - **BOF4 vs BOF4-S 的区别**：BOF4 固定 `x̂(1)=-1, x̂(8)=0, x̂(16)=1`（3 个固定值）；BOF4-S 固定 `x̂(8)=0, x̂(16)=1`（2 个固定值），因为 signed normalization 后归一化权重分布只在 x=1 有一个离散概率质量 `1/I`。
  - **OPQ 算法流程**：
    ```
    # Step 1: Block-level standard deviation
    for b in 1..B:
        σ_b = std(blocks[b, :])  # corrected sample std
    
    # Step 2: Outlier detection
    threshold = F_M^{-1}(0.95)  # 95th percentile of absolute block maxima
    for each weight w[b,i]:
        if |w[b,i]| > σ_b * threshold:
            mark as outlier -> store in bfloat16 + 64-bit position index
            replace w[b,i] = 0 in tensor
    
    # Step 3: Quantize non-outlier weights with BOF4(-S) as usual
    # Step 4: Decoding: read outlier positions, restore bfloat16 values
    ```
  - **码本示例**（BOF4-S MSE, I=64）：`x̂ = [-0.8568, -0.6693, -0.5235, -0.4005, -0.2911, -0.1900, -0.0939, 0.0, 0.0888, 0.1795, 0.2743, 0.3760, 0.4887, 0.6189, 0.7791, 1.0]`
