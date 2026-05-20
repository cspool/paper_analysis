## RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  提出RTMPQ（Rotated Token-wise Mixed Precision Quantization）算法，通过Hadamard旋转抑制channel-wise outlier后迁移至token维度，再用token-wise mixed precision量化处理双维度outlier。核心设计：(1) Hadamard Rotation：利用Hadamard矩阵（元素+1/-1的正交矩阵）乘activation矩阵，平滑channel维度的极端值，将irregularity从channel维度迁移至token维度。乘法可通过Fast Walsh-Hadamard Transform (FWT)在O(mn log n)复杂度内高效实现。(2) Token-wise Mixed Precision：旋转后outlier呈纯token-wise分布，通过per-token maximum activation value做top-k选择（5% outlier比例），outlier token用INT8量化、其余token用INT4量化，outlier set对称扩展到weight矩阵。(3) 四种精度组合的cross-precision乘法（W4A4/W4A8/W8A4/W8A8），使用INT32累加器防止overflow。实验比较perplexity（WikiText2）和zero-shot accuracy（ARC-C/E, HellaSwag, LAMBADA, PIQA, WinoGrande六个下游任务），对比BF16、INT4 uniform、MixQ、Atom、QuaRot等baseline。Qwen3-8B上RoMeo perplexity 10.97优于QuaRot 11.53、MixQ 14.76，Qwen3-8B zero-shot average 64.41 vs BF16 70.42 vs QuaRot 63.32。

- 硬件平台是什么，配置是什么。
  NVIDIA GeForce RTX 4090 GPU (24GB memory, Ada Lovelace, peak INT4 performance 8× over FP16)。软件: Python 3.12, PyTorch 2.8.0, CUDA 12.8, HadaCore (Hadamard变换), CUTLASS, Triton。多卡serving时Qwen3-14B用2×RTX 4090 TP，Qwen3-32B用4×RTX 4090 TP。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen3 (8B/14B/32B)、Llama-3.1 (8B/70B)。数据集：WikiText2（perplexity评估），六个zero-shot下游任务：ARC-Challenge/ARC-Easy, HellaSwag, LAMBADA, PIQA, WinoGrande（通过lm-eval library评估）。Perplexity用batch size 2、sequence length 2048；下游任务batch size 32。Outlier比例固定为5%（activation和weight各5%）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：https://github.com/thu-pacman/RoMeo。RTMPQ算法pipeline：
  1. 离线阶段：对weight矩阵左乘H^T（Hadamard转置）完成离线旋转→识别weight outlier（per-column max value top-k）→量化weight为INT4+INT8 mixed precision
  2. 在线推理（以Qwen3-8B单token前向为例）：
     a. Activation Hadamard Rotation：输入activation X (FP16, shape [batch, hidden])右乘Hadamard矩阵H→X'=XH，使用FWT高效实现，channel-wise outliers被平滑
     b. Token-wise Outlier Detection：计算X'的per-token max value→top-k选择5% outlier tokens→存入outlier set O_A
     c. Mixed Precision Quantization：outlier tokens→INT8量化（含scaling factor）；normal tokens→INT4量化
     d. Cross-Precision GEMM：四种精度组合（W4A4/W4A8/W8A4/W8A8）的矩阵乘法→INT32 accumulator→dequantize with per-token scaling factors
     e. Post-mul Overwrite：高精度outlier计算结果overwrite对应位置输出
  3. 理论加速比公式：S = 1 / (P_INT4/4 + (1-P_INT4)/2)，其中P_INT4为纯INT4计算比例。当m=n=4096, k_a=k_w=256时，P_INT4=88%，理论加速比3.57×
