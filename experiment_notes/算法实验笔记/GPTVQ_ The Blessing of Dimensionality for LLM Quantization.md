## GPTVQ: The Blessing of Dimensionality for LLM Quantization

- **属于算法pipeline的实现是什么？实验比较什么？**
  提出 GPTVQ，一种将 GPTQ 框架扩展到非均匀向量量化（VQ）的后训练量化方法。核心实现：(1) **GPTVQ 算法**：将权重矩阵按 d 维列组逐块量化（2D VQ 为默认配置），使用 Hessian 加权误差补偿（沿用 GPTQ 的 Cholesky 分解 + lazy batch update），量化误差沿 d 维坐标累积后一次性更新剩余权重；(2) **EM 初始化**：用加权马氏距离或 k-Means++ 初始化 codebook，E-step 用 Hessian 加权的距离函数（公式 5）找最优质心，M-step 用 Moore-Penrose 伪逆闭式解更新质心；(3) **Codebook update**：GPTVQ 结束后通过梯度下降进一步最小化层输出 MSE（公式 7）更新 codebook 值；(4) **Blockwise data normalization**：在 codebook 初始化前对权重子行按 log-scale 做 per-block max 归一化（4-bit 缩放因子），改善 VQ 误差；(5) **4-bit codebook 量化**：将 codebook 进一步量化到 INT4，通过预缩放 + EM 初始化 + GPTQ 内逐组缩放实现极小精度损失。

  实验对比：
  - Weight-only 量化 baseline：RTN、GPTQ、AWQ、OmniQuant（均匀量化，group size 128/64）
  - VQ baseline：AQLM、QuIP#（其他向量量化方法）
  - 位宽配置：2.125/2.25/3.125/4.125 bpv，1D/2D/4D VQ
  - 模型：Llama-1 (7B/13B/30B/65B)、Llama-2 (7B/13B/70B)、Llama-3 (8B/70B)、Mistral-7B-v0.1、Mixtral-MoE-8x7B-v0.1、BLOOM-560M（消融）
  - 评估指标：WikiText2 perplexity、PIQA/ARC-easy/ARC-challenge/BoolQ/HellaSwag/Winogrande 零样本准确率平均
  - 消融：EM 初始化方法（Mahalanobis vs k-Means++）、EM 迭代次数（10-100）、Codebook update 有无、scaling block size、codebook SVD vs INT8 量化 vs 无压缩
  - 与 LoRA 结合：GPTVQ + LoRA adapter（frozen/trained），对比 QLoRA/LoftQ，评估 WikiText2 PPL + GSM8k 准确率

- **硬件平台是什么，配置是什么。**
  量化校准：单张 NVIDIA H100 GPU。Llama-v2-7B 量化时间约 30 分钟 - 1 小时，Llama-v2-70B 约 3-11 小时（vs AQLM 的 35 小时 on H100）。移动端推理：Snapdragon X Elite 平台，Windows + Clang 18.1 with Polly。校准数据：WikiText2 训练集 128 sequences × 2048 tokens。与 AQLM 对比时使用 SlimPajama 校准集（4096 samples × 2048 tokens）。

- **模型是什么。数据集和bench分别是什么。**
  - 模型：Llama-1 (7B/13B/30B/65B)、Llama-2 (7B/13B/70B)、Llama-3 (8B/70B)、Mistral-7B-v0.1、Mixtral-MoE-8x7B-v0.1、BLOOM-560M
  - 校准数据：WikiText2 训练集（128 sequences, 2048 tokens，默认）；SlimPajama（4096 samples × 2048 tokens，AQLM 对比）
  - Perplexity：WikiText2（validation set, sequence length 2048）
  - 零样本任务：PIQA、ARC-easy、ARC-challenge、BoolQ、HellaSwag、Winogrande（LLM-evaluation-harness）
  - GSM8k（LoRA adapter 实验）
  - 对 Llama3 零样本平均省略 BoolQ（对齐 Huang et al. 2024 协议）

- **开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。**
  论文声明 GPU kernel 代码 "will be made available in the future"（尚未开源）。算法伪代码见论文 Algorithm 1 (GPTVQ) 和 Algorithm 2 (QuantGroup)。

  **GPTVQ 算法核心流程（Algorithm 1, 论文第 4 页）**：

  输入：权重矩阵 W ∈ R^{r×c}，逆 Hessian H^{-1}，block size B，VQ dimensionality d，质心数 k，group size l（假设每 column 一个 group）

  ```
  1. N_b ← c/B                     # 总 block 数
  2. m ← l/r                       # group 的 column 数
  3. Q ← 0_{r,c}                   # 量化后权重
  4. E ← 0_{r,c}                   # 误差矩阵
  5. N_g ← rc/l                    # 总 group/codebook 数
  6. C_g ← 0_{d,k}, g=1,...,N_g    # codebook 初始化
  7. H^{-1} ← Cholesky(H^{-1})^T   # Cholesky 分解得到上三角矩阵
  8. for i = 0, B, 2B, ..., N_b·B do
  9.     if i % m == 0 then        # 新 group 开始时
  10.        g ← i/m               # 当前 group 索引
  11.        C_g ← init_codebook(W_{:, i:i+m-1})  # EM 初始化 codebook
  12.    end if
  13.    Q_{:, i:i+m-1} ← QUANTGROUP(W_{:, i:i+m-1})  # 量化当前 group
  14.    W_{:, i+B:} ← W_{:, i+B:} − E · [H^{-1}]_{i:i+B, i+B:}  # lazy update
  15. end for
  ```

  **QuantGroup 子算法（Algorithm 2, 论文第 4 页）**：

  ```
  1. function QUANTGROUP(W)  # W ∈ R^{r×m}
  2.   for j = 0, d, 2d, ..., l do
  3.     P = j, ..., j+d-1   # 当前 d 维列的索引
  4.     Q_{:,P} ← VQ_quant(W_{:,P}, C_g)  # 用 C_g 中最优质心量化
  5.     E_{:,P} ← (W_{:,P} - Q_{:,P})[H^{-1}]_P  # 计算误差
  6.     U ← Σ_{p=0}^{d-1} E_{:,j+p} [H^{-1}]_{p, j+d-1:B}  # 累积误差
  7.     W_{:, j+d-1:B} ← W_{:, j+d-1:B} - U  # 补偿剩余权重
  8.   end for
  9. end function
  ```

  **Codebook 初始化 EM 算法**：

  目标：min_{I, c} Σ_{m=0}^{k} Σ_{i∈I_m} (x^{(i)} - c^{(m)})^T D^{(i)} (x^{(i)} - c^{(m)})
  - D = diag(1/[H^{-1}]_{11}, ..., 1/[H^{-1}]_{cc})（Hessian 加权）
  - E-step：固定质心 c^{(m)}，为每个 d 维向量 x^{(i)} 分配最优质心（公式 5）
  - M-step：固定分配，闭式解 c^{(m)} = (Σ D^{(i)})^{+} Σ D^{(i)} x^{(i)}（Moore-Penrose 伪逆）

  **Codebook Update（附录 A）**：
  初始化完成后，固定 codebook 索引，用梯度下降（PyTorch）最小化 ||WX - Q(C)X||²_F，其中 Q(C) 是 codebook C 的查找操作。每步更新 C 后重建 Q 并继续。

  **推理时的解压缩流程（移动端 CPU）**：
  1. DRAM → SoC cache：加载 VQ 编码 (indices + LUT + scale)
  2. TBL 指令解码 6-bit index → 8-bit signed int（每个维度 1 条指令，2D VQ 需 2 条）
  3. 逐元素 scale × decoded_int → 反量化到 native data type
  4. 矩阵-向量乘法（SIMD 加速）

  **Bits per value 计算**：bpv = log₂(k)/d + kdb_c/l，k=质心数，d=VQ 维度，b_c=codebook bit-width，l=共享同一 codebook 的权重数

  **关键结果**：
  - Llama-v2-70B W2@g128：GPTVQ 2D PPL 4.72 vs OmniQuant 6.55（↓1.83）
  - GPTVQ 2D 4D at 3.125 bpv：WikiText2 PPL 5.83/7.00（Llama-v3-8B）vs FP16 6.14
  - 压缩时间：Llama-v2-7B 2.5h vs AQLM no BFT 18.3h（7.3× 加速）
  - 与 LoRA 结合：GPTVQ 4D 2.125 bpv + LoRA-trained Llama-v2-7B WikiText2 PPL 5.83 vs LoftQ NF2 20.9 (GSM8k)
