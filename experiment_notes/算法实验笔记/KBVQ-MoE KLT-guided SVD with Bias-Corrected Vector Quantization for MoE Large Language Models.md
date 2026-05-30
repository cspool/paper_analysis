## KBVQ-MoE KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - **KBVQ-MoE**：第一个专门为 MoE 架构设计的向量量化（VQ）框架，包含两个创新模块：
    - **IDRE（Input-driven Redundancy Elimination）**：KLT 引导的 SVD 分解。先对输入激活做 Karhunen–Loève Transform（KLT）构建输入相干基（input coherence basis），将各 expert 权重投影到此空间形成统一表示 `W̄`，再对 `W̄` 做 SVD 提取 top-k 主导共享分量 `W_share`（保留全精度），剩余 expert-specific 分量 `W_quant` 交给 VQ 量化。KLT 确保提取方向同时对齐输入能量和跨 expert 权重能量。
    - **BCOS（Bias-Corrected Output Stabilization）**：对 expert-specific 权重做 VQ 量化后，以 channel-wise affine compensation（scale s + bias b）校正量化输出，使得校正后的每个 channel 的 mean/variance 与 FP16 输出对齐。s 和 b 通过 MMSE 闭式解计算：`s_j ≈ σ_{y_j}/σ_{ŷ_j} - 1`, `b_j = μ_{y_j} - (1+s_j)μ_{ŷ_j}`。
  - 实验比较：
    - 与 RTN、GPTQ（scalar quantization）、MoEQuant（MoE 专用量化）、Direct VQ（直接向量量化）比较，在 2-bit 和 3-bit 下的 WikiText2 perplexity 和 7 个零样本任务（ARC-E, ARC-C, HellaSwag, LAMBADA-openai, LAMBADA-standard, PIQA, WinoGrande）的 Avg Acc。
    - Plugin 实验：将 IDRE+BCOS 作为插件集成到 GPTVQ 和 VPTQ 中对比性能提升。
    - 消融实验：KLT vs 无 KLT 的 SVD；不同 SVD 截断秩 k/n 比例；IDRE 和 BCOS 各自贡献；BCOS 中 mean 和 variance 校正各自贡献。
    - 与 MoE 结构压缩方法（Sub-MoE, D2-MoE, EAC-MoE）在 Mixtral-8×7B 上的公平对比。
    - 更具挑战性 benchmark：MMLU, MathQA, GSM8K, HumanEval。
    - 解码速度测试：BF16 vs Quantized 的 tokens/s 加速比。

- 硬件平台是什么，配置是什么。
  - 量化实验：NVIDIA RTX A6000 GPU
  - MoE 压缩方法对比实验：NVIDIA RTX A100 GPU, PyTorch 2.1
  - 解码速度测试：论文未明确说明测试 GPU，仅报告 Qwen1.5-MoE-A2.7B 在 1k input tokens 下 BF16 为 22.31 tokens/s，2-bit quantized 为 35.24 tokens/s（加速 1.58×）

- 模型是什么。数据集和bench分别是什么。
  - 模型：Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B, Mixtral-8x7B, DeepseekV2-Lite（以及 DeepSeekMoE-16B 用于 challenge benchmark）
  - Calibration 数据集：RedPajama 数据集，random seed 42，随机采样 256 条，sequence length 4096
  - Perplexity 评测：WikiText2，sequence length 4096
  - 零样本评测（7 个数据集）：Arc-Challenge, Arc-Easy, HellaSwag, LAMBADA-openai, LAMBADA-standard, PIQA, WinoGrande
  - 挑战性 benchmark：MMLU, MathQA, GSM8K, HumanEval
  - 评测工具：LM-Evaluation-Harness (v0.4.0)

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：论文未明确说明开源链接
  - KBVQ-MoE 算法完整流程（Algorithm 1 from Appendix A.8）：
    ```
    === Pre-Process: KLT-guided SVD (IDRE) ===
    输入: expert weights {W^(i)}_{i=1..n}, 输入激活 X
    
    1. 计算输入协方差: C_X = 1/(B-1) X^T X ∈ R^{ic×ic}
    2. 特征分解: C_X = U_KLT Λ_KLT U_KLT^T
    3. 输入相干基: U_X = U_KLT Λ_KLT^{1/2}
    4. for i=1 to n:
         W̃^(i) = W^(i) U_X          # 投影到输入相干空间
    5. 堆叠所有 expert: W̄ = [W̃^(1); ...; W̃^(n)] ∈ R^{(n·oc)×ic}
    6. SVD: W̄ = (U Σ V^T)^T
    7. 选 top-k: U_k = U_{:,1:k}, V_k = V_{:,1:k}, Σ_k = Σ_{1:k,1:k}
    8. 按 expert 划分 V_k: V_k = [Σ_k V_k^(1); ...; Σ_k V_k^(n)]
    9. for i=1 to n:
         U_share = U_X^{-1} U_k              # ic×k 共享映射
         W_share^(i) = (U_share (V_k^(i))^T)^T   # oc×ic 共享分量
         W_quant^(i) = W^(i) - W_share^(i)       # 残差（expert-specific）
    
    === Quantization: Vector Quantization of W_quant ===
    11. for i=1 to n:
         将 W_quant^(i) 划分为 d 维子向量 {z}
         用 K-means++ 初始化 codebook C = {c_1,...,c_K}
         训练 VQ codebook via K-means (100 iterations)
         for each sub-vector z:
           q = argmin_j ||z - c_j||^2
           z_q = c_q
         得到 W_quant,VQ^(i)
    
    === Post-Process: Bias Correction (BCOS) ===
    13. 定义量化权重: Ŵ^(i) = W_share^(i) + W_quant,VQ^(i)
    14. 从 calibration data 估计 per-channel 统计量:
          μ_y, σ_y  (原始输出 y = W^(i)x)
          μ_ŷ, σ_ŷ  (量化输出 ŷ = Ŵ^(i)x)
    15. 计算校正参数:
          s_j = σ_{y_j} / σ_{ŷ_j} - 1
          b_j = μ_{y_j} - (1+s_j) μ_{ŷ_j}
    16. 校正输出: y_corr = (1+s) ⊙ (Ŵ^(i)x) + b
    ```
  - 关键超参数：
    - SVD 截断秩 k: 推荐 k = ic/128（full rank 的 1/128），此时平均 bit-width 增加约 0.08 bits
    - VQ 子向量长度 d: 设置为 4
    - K-means: K-means++ 初始化，100 iterations
    - BCOS 额外参数: 每层仅 2·oc 个参数（scale + bias per channel），推理时额外 FLOPs < 0.1%
  - 压缩效果：Qwen1.5-MoE-A2.7B 在 2-bit 下压缩率 87%，有效位宽 2.08 bits；实际存储从 27.9GB(FP16) 降至 4.3GB
