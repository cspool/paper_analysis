## QTIP: Quantization with Trellises and Incoherence Processing

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QTIP 是一种 weight-only 后训练量化（PTQ）方法，使用 **Trellis Coded Quantization (TCQ)** 替代 Vector Quantization (VQ) 实现超高维量化（有效维度 256，VQ 被限制在 ≤8 维）。核心设计包含三部分：(1) **Bitshift Trellis**——硬件友好的 trellis 结构，节点 i→j 有边当且仅当 j = (i·2^{kV} mod 2^L) + c，第 t 组权重仅依赖连续 L-bit 窗口，支持并行解码且无需存储 trellis 图结构；(2) **Compute-based Random Gaussian Codes**——三种免查找表或混合计算码：**1MAD**（MAD+LCG+4×8-bit 求和，2 指令生成近似高斯值）、**3INST**（LCG+XOR FP16 magic number 尾数/指数位，3 ALU 指令，m1+m2 近似高斯）、**HYB**（hash+2^Q×2 LUT 2D codebook，2 指令/权重摊销）；(3) **Incoherence Processing**——Random Hadamard Transform (RHT) 使权重近似 i.i.d. 高斯分布，匹配 TCQ 的源编码假设。QTIP 作为 QuIP# BlockLDLQ 框架的 drop-in 替换量化器，将 Tx×Ty 权重块作为高维序列用 Viterbi 算法（O(2^L T) 时间）量化。Algorithm 4 的近似 tail-biting trellis 使编码比特数与硬件字长对齐。
  - 实验比较：(1) QTIP 纯计算码 (1MAD/3INST, L=16, V=1, Tx=Ty=16) vs QuIP#、AQLM——Llama 2 (7B/13B/70B) 上 2/3/4-bit perplexity（Table 3），无 fine-tuning 即超越含 fine-tuning 的 QuIP# 和 AQLM；(2) QTIP HYB 混合码 (L=16, V=2, Q=9, 2KiB codebook) vs QuIP#、AQLM、GPTVQ-2D、PV-Tuning——Llama 1/2/3/3.1/3.2 全系列 perplexity 和 zeroshot 准确率；(3) 消融：L (8/10/12/16) vs MSE（Table 10）、V (1/2/4) vs MSE（Table 11）、tail-biting 近似 vs 最优解（Table 2）；(4) i.i.d. 高斯源量化失真（Table 1）：Lloyd-Max SQ 0.118 vs QuIP# E8P VQ 0.089 vs QTIP 256D TCQ 0.069 vs D_R 0.063。

- 硬件平台是什么，配置是什么。
  - GPU：RTX 6000 Ada (960GB/s 显存带宽) 用于推理吞吐评测；RTX 3090、RTX A6000 Ampere 用于跨平台解码速度（Table 17）。Together AI 提供计算资源。PyTorch + HuggingFace Transformers。解码利用 16×16 MMA tile 进行矩阵乘法。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama 1 (7B/13B/30B/65B)、Llama 2 (7B/13B/70B)、Llama 3 (8B/70B)、Llama 3.1 Instruct (8B/70B/405B)、Llama 3.2 Instruct (1B/3B)。
  - 校准数据：RedPajama（Hessian 生成：Llama 1/2 用 6144 seq × 2048 tokens，Llama 3 用 4096 seq × 8192 tokens，405B 用 2048 seq × 8192 tokens）。
  - 评估：Wikitext2、C4（perplexity, OPTQ 方式）；LM Eval Harness（zeroshot: ARC-C, ARC-E, BoolQ, PiQA, WinoGrande, HellaSwag）；推理吞吐 (tokens/s, batch_size=1 decode)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Cornell-RelaxML/qtip
  - 依赖：PyTorch, HuggingFace Transformers, QuIP# BlockLDLQ, fast-hadamard-transform, Neil Sloane Hadamard 矩阵, RedPajama。
  - 算法核心（QTIP + BlockLDLQ, Algorithm 5 伪代码）：
    ```
    输入: W ∈ R^{m×n}, H ∈ R^{n×n}, T_x, T_y, L, k, V, code C.
    Ŵ ← 0_{m,n}
    LDL^T ← T_y-block LDL decomposition of H
    A ← L - I
    for j = n/T_y-1 down to 0:
      x ← W_{:,jT_y:(j+1)T_y} + (W_{:,jT_y:} - Ŵ_{:,jT_y:}) A_{jT_y:(j+1)T_y}
      x ← x.reshape(m/T_x, T_x T_y)
      x̂ ← Viterbi(x, (L,k,V) bitshift trellis, C)  # 逐行 TCQ
      Ŵ_{:,jT_y:(j+1)T_y} ← x̂.reshape(m, T_y)
    输出: Ŵ
    ```
  - Viterbi 量化：在 bitshift trellis G 上最小化 Σ ||C_{x_i} - s_i||²，动态规划 V_t(y) = min_{(x,y)∈G} V_{t-1}(x) + ||C_y - s_t||²，O(2^L T) 时间。
  - 1MAD 码：x ← (ax+b) mod 2^32 (LCG) → x ← sum of four 8-bit unsigned ints → (x-510)/147.8 → 近似 N(0,1)。2 inst: MAD + vabsdiff4。
  - 3INST 码：x ← (ax+b) mod 2^32 → XOR bottom 16 bits with magic FP16 m's mantissa/exp/sign → XOR top 16 bits → m1+m2 → 近似高斯。3 inst: MAD + lop3 + add。
  - HYB 码：x ← x²+x mod 2^32 (hash) → idx = (x>>(15-Q)) & (2^Q-1) → v = C[idx] (2^Q×2 LUT) → sign-flip v[1] via XOR bit 15。摊销 2 inst/weight。C 可 fine-tune。
  - 关键：QTIP HYB codebook 仅 2KiB (2^9×2)，比 AQLM 的 1MiB 小 512×，可放入 L1 cache。解码仅需 ≤4 GPU 指令/权重，达到 >80% 峰值显存带宽。
