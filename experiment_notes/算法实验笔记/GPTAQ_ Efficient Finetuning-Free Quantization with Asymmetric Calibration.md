## GPTAQ: Efficient Finetuning-Free Quantization with Asymmetric Calibration

- **属于算法pipeline的实现是什么？实验比较什么？**
  提出 GPTAQ，一种改进 GPTQ 的 finetuning-free 量化方法。核心创新是**非对称校准（Asymmetric Calibration）**：GPTQ 每层独立最小化 `||ŵX - wX||²`（对称校准），GPTAQ 改为最小化 `||ŵX - wX̃||²`，其中 X̃ 是前一层全精度模型的输出激活，X 是前一层量化后的实际激活。这显式补偿了前层量化误差累积导致的输入激活偏差。方法基于 Optimal Brain Compression (OBC) 框架推导出闭式解，Δw 包含两项：(1) 与 GPTQ 相同的量化误差补偿项 `−((ŵ_q − w_q)/H_{qq}^{-1})·H_{q,:}^{-1}`；(2) 新增的残留误差补偿项 `rX^T H_{-q}^{-1}`。为实现高效计算，提出四个优化步骤：任意顺序处理列（支持行并行）、残差分解（将 R 分解为逐神经元分量避免重复计算）、Cholesky 重构化（数值稳定 + 矩阵融合）、Lazy-Batch 更新（提高 GPU 利用率）。仅需比 GPTQ 多约 20 行代码。

  实验对比：
  - Vision Transformer（W4A4, W2A4）：DeiT-S/B 上与 PTQ4ViT、APQ-ViT、PD-Quant、RepQ-ViT、GPTQ 对比
  - Language Transformer（W4A4, W2A4）：LLaMA2-7B/13B/70B、LLaMA3-8B/70B 上与 OmniQuant、QLLM、DuQuant、QuaRot+GPTQ、SpinQuant+GPTQ 对比
  - Weight-Only Quantization（3-bit per-group）：LLaMA2-7B/13B、LLaMA3-8B-Instruct 上与 AWQ、GPTQ 对比
  - 超大模型：EVA-02（90% ImageNet top-1）和 LLaMA3.1-405B 的 W4A4 量化
  - 评估指标：ImageNet Top-1 准确率（Vision）、WikiText2/C4 Perplexity（Language）、PiQA/ARC-E/ARC-C/HellaSwag/Winogrande/BoolQ 零样本准确率
  - 消融实验：ΔW 两项的各自贡献（仅第一项=GPTQ、仅第二项、两项联合=GPTAQ）、激活量化顺序（A→W vs W→A）
  - 效率分析：P 矩阵计算的并行 vs 非并行实现延迟对比、GPTQ vs GPTAQ 逐层延迟对比

- **硬件平台是什么，配置是什么。**
  单张 NVIDIA A100 GPU（所有量化校准实验）。PyTorch 2.4.1-cu12.4。HuggingFace Transformers 框架。GPU Hours 报告：LLaMA3-8B QuaRot+GPTQ 需 0.2h，QuaRot+GPTAQ 需 0.3h；LLaMA3-70B QuaRot+GPTQ 需 1.8h，QuaRot+GPTAQ 需 2.7h。SpinQuant 需额外 4-28 GPU-hours（8×A100）。

- **模型是什么。数据集和bench分别是什么。**
  - 模型：DeiT-S、DeiT-B（Vision Transformer）；LLaMA2-7B/13B/70B、LLaMA3-8B/70B、LLaMA3.1-405B（Language Transformer）；EVA-02（Huge Vision Transformer）
  - 校准数据：ImageNet 训练集 128 samples（Vision）；WikiText2 训练集 128 sequences × 2048 tokens（Language）；C4 128 sequences（Weight-Only）
  - Perplexity：WikiText2、C4
  - 零样本推理：PiQA、ARC-Easy、ARC-Challenge、HellaSwag、Winogrande、BoolQ
  - ImageNet Top-1 准确率（Vision）

- **开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。**
  开源代码：https://github.com/Intelligent-Computing-Lab-Yale/GPTAQ

  **算法pipeline核心流程（Algorithm 1：GPTAQ 单层量化）**：

  输入：FP 权重 W (m×n)，校准输入 X，FP 输入 X̃（来自全精度前向），Block size B
  ```
  1. H ← X X^T                                    # 计算 Hessian
  2. ΔX X^T ← (X̃ - X) X^T                        # 输入偏差 × 激活转置
  3. L ← Inverse_Cholesky(H + λ₁I)               # Cholesky 分解
  4. P ← ((ΔX X^T L) ⊙ M_U) L^T                  # Theorem 4.2：并行计算 P 矩阵
     其中 M_U 是严格上三角掩码矩阵（对角线以上为1）
  5. Q ← 0_{m×n}, E ← 0_{m×B}                    # 初始化量化权重和误差缓冲
  6. for i = 0, B, 2B, ... do                     # 按 block 迭代
  7.     for j = i, i+1, ..., i+B-1 do             # block 内逐列量化
  8.         Q_{:,j} ← quant(W_{:,j})              # 量化当前列
  9.         E_{:,j-i} ← (W_{:,j} - Q_{:,j}) / L_{jj}  # GPTQ 误差项
  10.        W_{:,j:(i+B)} ← W_{:,j:(i+B)} - E_{:,j-i} L_{j,j:(i+B)}^T  # 补偿 block 内
  11.    end for
  12.    # Lazy-batch 更新 block 外列
  13.    W_{:,(i+B):} ← W_{:,(i+B):} - E·L_{i:(i+B),(i+B):}^T          # GPTQ 项
  14.                  + W_{:,i:(i+B)} P_{i:(i+B),(i+B):}              # GPTAQ 新增项
  15. end for
  ```

  **与 GPTQ 的核心差异**：
  - GPTQ 的补偿项：δw = −(w_q − ŵ_q)/L_{qq} · L_{q,q:}（仅量化误差补偿）
  - GPTAQ 的补偿项：在第 14 行新增 `W_{:,i:(i+B)} P_{i:(i+B),(i+B):}` 项，补偿前层量化导致的输入激活偏差累积
  - P 矩阵通过 Theorem 4.2 的并行公式一次性计算：`P = ((ΔX X^T L) ⊙ M_U) L^T`，利用 CUDA 优化，<1ms 完成（vs 非并行实现 >10⁴× slower）

  **残差分解原理（Step 2）**：
  - 直接法需要每次迭代重新计算 R = W X̃ − W X，复杂度 O(mnk)，k >> n 时极慢
  - 分解法：R = Σ_{q=1}^n W_{:,q} ΔX_{q,:}，预计算一次后，第 q 次迭代仅关注第 q 个神经元的残差分量 `W_{:,q} ΔX_{q,:} X_{:,q:}^T H_{-q}^{-1}`，复杂度降至 O(mn)

  **整模型量化流程（Algorithm 2）**：
  ```
  for i = 1 to b-th block:
      Move block[i] to GPU                     # 每次仅一个 block 在 GPU
      X̃ ← block[i](X̃)                          # 全精度前向，缓存各层 FP 输入
      if AQ enabled: enable activation quantization
      for each layer in block[i]:
          Compute H and ΔX X^T for layer
          Run GPTAQ Algorithm 1 for layer
          Quantize layer weights
      X ← block[i](X)                          # 量化 block 输出
      Move block[i] to CPU
  ```

  **关键结果**：
  - W4A4 DeiT-S：72.8%（GPTQ 71.9%，+0.9%）；W2A4 DeiT-S：46.8%（GPTQ 38.4%，+8.4%）
  - W4A4 LLaMA3-70B WikiText2 PPL：6.93（GPTQ 9.44，↓2.51）；QuaRot+GPTAQ avg accuracy 69.1%（QuaRot+GPTQ 62.4%，+6.7%）
  - W4A4 LLaMA3.1-405B WikiText2 PPL：3.48（GPTQ 5.82，↓2.34）
  - W4A4 EVA-02 ImageNet Top-1：88.30%（GPTQ 86.48%，+1.82%；FP16 90.05%）
  - 额外延迟：GPTAQ 比 GPTQ 多 30-40%（大维度时），小维度时 <10%
