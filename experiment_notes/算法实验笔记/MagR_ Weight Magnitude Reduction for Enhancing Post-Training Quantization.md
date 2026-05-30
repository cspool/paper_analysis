## MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MagR 是一种基于 channel-wise ℓ∞-regularized 最小二乘优化的权重预处理技术，通过 Proximal Gradient Descent（近端梯度下降）配合 ℓ₁-ball 投影来迭代减少预训练权重的最大幅度，从而缩小量化步长 δ、降低量化误差。MagR 作为非线性预处理变换，不需要在推理时对特征做逆变换。MagR + RTN、MagR + OPTQ、MagR + OPTQ†（加30轮 coordinate descent）以及 MagR + QuIP 作为复合方案。
  - 实验比较：在 LLaMA1（7B–65B）和 LLaMA2（7B–70B）上，对 W2A16、W3A16、W4A16（per-channel）和 W2A16g128、W3A16g128、W4A16g128（per-group）配置进行 weight-only quantization，对比 RTN、OPTQ、AWQ、OmniQuant、QuIP。指标为 WikiText2/C4 perplexity 和 PIQA/ARC-Easy/ARC-Challenge/Winogrande 零样本准确率。还报告了 MagR 预处理耗时和总量化耗时。

- 硬件平台是什么，配置是什么。
  - 单张 NVIDIA A100 GPU（80GB 显存）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaMA1（7B, 13B, 30B, 65B）和 LLaMA2（7B, 13B, 70B），使用 HuggingFace 实现。
  - 数据集/bench：WikiText2 和 C4（语言生成 perplexity 评估，context length 2048），128 个 calibration samples；PIQA、ARC-Easy、ARC-Challenge、Winogrande（零样本任务，使用 lm-eval-harness）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/AozhongZhang/MagR
  - 算法流程（per-channel MagR for one linear layer）：
    1. 输入预训练权重 W_hat ∈ R^{m×n}，Hessian H = XᵀX ∈ R^{m×m}（X 为 calibration 特征矩阵），迭代次数 K，步长 η = 1/λ_max(H)，惩罚参数 α。
    2. 初始化 W⁰ = W_hat。
    3. 对 k = 0,...,K-1：
       - 梯度下降步：V^k = W^k - η · H · (W^k - W_hat)
       - 近端算子（列级 ℓ₁-ball 投影）：W^{k+1} = V^k - ηα · proj_{‖·‖₁≤1}(V^k/(ηα))
    4. 返回预处理后的权重 W = W^K。
    5. 在预处理后的 W 上应用标准 uniform quantizer（含可选的 δ 缩放因子 β ≤ 1）：δ = β · (max(w)−min(w))/(2^b−1)。
  - 核心思想：特征矩阵 X 近似秩亏（见表2，fraction rank 均值 70%–84%），因此存在无数 w 满足 Xw ≈ Xw_hat，可在保持层输出的前提下大幅降低 w 的 ℓ∞ 范数（最大幅度），缩小量化步长。
  - ℓ₁-ball 投影用 O(m log m) 的排序+软阈值算法（Algorithm 2/3）。
  - per-group 扩展：将 V ∈ R^{m×n} reshape 为 R^{d×((m/d)·n)} 后独立做 ℓ₁-ball 投影。
  - 参数设置：K=150，α=10⁻³（per-channel）/ 10⁻⁴（per-group），β ∈ [0.8, 0.95] 取决于 bit-width。
