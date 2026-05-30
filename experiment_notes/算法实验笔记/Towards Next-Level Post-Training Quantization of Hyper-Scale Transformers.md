## Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 *aespa*（attention-centric efficient and scalable post-training quantization algorithm）算法，是一种面向超大规模 Transformer 模型的 PTQ 方案。核心策略是：逐层（layer-wise）执行量化以保持效率，但以注意力输出（attention-wise）重构为目标来考虑跨层依赖。具体实现分两步：(1) 使用 Z-FOLD 结合提出的 Hessian 计算量化参数（scale 和 zero-point）；(2) 使用 AdaRound 基于提出的精炼损失函数优化 weight-rounding policy。精炼量化目标分别为：W_V 最小化 `tr(ΔW_V·E[XA^TAX^T]·ΔW_V^T)`、W_Q 最小化 `tr(E[K^TK]·ΔW_Q·E[XX^T]·ΔW_Q^T)`、W_K 最小化 `tr(E[Q^TQ]·ΔW_K·E[XX^T]·ΔW_K^T)`。通过预计算 E[XX^T]、E[XA^TAX^T]、E[K^TK]、E[Q^TQ]，每轮迭代仅需 O(d_h d^2) FLOPs，相比传统 block-wise 方法 O(B d_h L·max{d,L})，在 OPT-125M 上约快 10 倍。仅做 weight-only 量化（激活保持 FP16）。实验比较：(a) block-wise PTQ：BRECQ、OmniQuant、AffineQuant；(b) layer-wise PTQ：RTN、OPTQ、Z-FOLD。在 WikiText-2、C4、PTB 上评估 PPL，并用 ARC-c/e、HellaSwag、MMLU 评估零样本推理性能。精度涵盖 INT2/3/4/6，特别在 INT2 精度下优势显著（如 OPT-6.7B INT2，aespa PPL=15.71 vs OmniQuant=4900+）。

- 硬件平台是什么，配置是什么。
  单卡 NVIDIA A100 GPU（80 GB），LLaMA2 实验使用 NVIDIA H100 GPU。校准数据集使用从 C4 随机采样 128 段 2048 token 的序列。量化处理时间：OPT-125M 约 5 分钟完成 INT2 量化（BRECQ 需要 ~2 小时），OPT-1.3B 约 1.24 小时（BRECQ 需要 ~10.7 小时）。

- 模型是什么。数据集和bench分别是什么。
  模型：OPT（125M/350M/1.3B/2.7B/6.7B/13B/30B）、BLOOM（560M/1.1B/1.7B/3B/7.1B）、LLaMA（7B/13B/30B）、LLaMA2（7B/13B）。校准数据集：C4（128 segments × 2048 tokens）。评估基准：WikiText-2、C4、PTB（perplexity）、零样本任务 ARC-c、ARC-e、HellaSwag、MMLU。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/SamsungLabs/aespa（NeurIPS 2024 官方实现，CC BY-NC 4.0，Python，5 commits），包含 quantize.py、quantizer.py、quant_utils.py、aespa.py、main.py 及定制模型文件（modeling_llama_custom.py、modeling_bloom_custom.py）。

  算法 pipeline 伪代码（参考论文 Algorithm 1 及 Table 4）：
  ```
  def aespa_quantization(W, X_calib):
      # Step 1: Pre-compute Hessian matrices
      H_xx = E[XX^T]                              # for all layers
      H_v = E[X A^T A X^T]                        # for W_V only
      E_ktk = E[K^T K]                            # for W_Q
      E_qtq = E[Q^T Q]                            # for W_K

      # Step 2: Determine quantization parameters (scale, zero-point)
      for each layer l:
          if l in {W_Q, W_K, W_V}:
              H = corresponding_attention_hessian  # use attention-aware Hessian
          else:
              H = H_xx                            # standard Hessian for FFN/out-proj
          # Z-FOLD: optimize step size s to minimize tr(ΔW·H·ΔW^T)
          s = argmin_s tr(ΔW(s)·H·ΔW(s)^T)

      # Step 3: Initialize W_int via OPTQ (optional but empirically faster)
      W_int = OPTQ_round(W, s)

      # Step 4: Optimize weight-rounding policy via AdaRound with proposed losses
      for iter in range(2000):
          for W_Q:  loss = tr(E[K^T K]·ΔW_Q·H_xx·ΔW_Q^T) + λ·round_reg
          for W_K:  loss = tr(E[Q^T Q]·ΔW_K·H_xx·ΔW_K^T) + λ·round_reg
          for W_V:  loss = tr(ΔW_V·H_v·ΔW_V^T) + λ·round_reg
          for other: loss = tr(ΔW·H_xx·ΔW^T) + λ·round_reg
          update W_int via gradient descent (lr=0.015)

      return W_int, s
  ```
  张量计算示例（W_V 量化，基于 Equation 17-18）：预计算阶段对校准数据全量计算 `H_V = 2 * mean(X @ A^T @ A @ X^T)`，形状为 [d, d]。每次迭代计算损失 `loss = sum((ΔW_V @ H_V) ⊙ ΔW_V)`，无需执行 attention forward pass，复杂度 O(d_h d^2) 与校准数据量无关。
