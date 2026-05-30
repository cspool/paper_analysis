## Training Dynamics Impact Post-Training Quantization Robustness

- 属于算法pipeline的实现是什么？实验比较什么？
  研究训练过程中的超参数（学习率、学习率调度策略、weight decay、weight averaging）对模型 PTQ 鲁棒性的影响。核心发现：量化误差主要由学习率衰减驱动，而非训练数据规模；较大的稳定学习率、weight averaging（LAWA/model soup）和更大的 weight decay 均能降低 PTQ 误差。实验比较：(a) 不同开源模型训练轨迹的量化误差演变（OLMo/OLMo2/SmolLM3/Apertus/OpenSci/Amber），涵盖 3-bit 和 4-bit GPTQ；(b) 受控实验：不同 token 预算、学习率大小（1e-3 到 6e-3）、WSD vs cosine 调度、不同 weight decay（λ）、AdamW vs AdamC 优化器下的量化误差；(c) LAWA weight averaging vs 中间 cooldown 的性能对比；(d) 量化误差与 loss landscape 几何性质（Hessian sharpness/trace）的关联。

- 硬件平台是什么，配置是什么。
  自训练实验：最多 8 块 NVIDIA A100-80GB GPU。下游评估使用 vLLM。开源模型评估使用 HuggingFace Transformers + GPTQModel/GPTQ backend。

- 模型是什么。数据集和bench分别是什么。
  开源模型：OLMo-1B/7B、OLMo2-1B/7B/13B/32B、SmolLM3-3B、Apertus-8B、OpenSci-1.3B、Amber-7B。自训练模型：Pythia-160M 和 70M 参数 Transformer（Vaswani 架构），在 FineWebEdu 上训练最多 100B tokens（seq_len=2048, batch_size=0.5M tokens）。校准数据集：C4（GPTQ 量化，group_size=128）。评估：held-out RefinedWeb（validation loss）；下游 benchmark：ARC-Challenge、ARC-Easy、OpenbookQA、PIQA、HellaSwag、WinoGrande、MathQA、PubMedQA、SciQ、Social IQa、CommonsenseQA、MMLU（12 个任务，5-shot）。评估框架：LM-eval-harness + vLLM。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未明确给出自研代码仓库。量化使用 GPTQModel（https://github.com/modelcloud/gptqmodel）和 HuggingFace 内置量化后端。预训练使用 plainLM（https://github.com/Niccolo-Ajroldi/plainLM）。Hessian 分析使用 PyHessian。

  核心分析 pipeline 伪代码：
  ```
  def analyze_ptq_robustness(checkpoints, lr_schedule, quant_method="GPTQ"):
      results = []
      for step, ckpt in enumerate(checkpoints):
          # Step 1: 量化模型
          W = ckpt.weights                        # FP16/BF16 weights
          W_q = quantize(W, method=quant_method)  # 3-bit or 4-bit GPTQ
          W_hat = dequantize(W_q)                 # reconstruct from scales + lowbit

          # Step 2: 计算相对交叉熵损失
          ce_full = cross_entropy(f(X; W), y)
          ce_quant = cross_entropy(f(X; W_hat), y)
          rel_ce = (ce_quant / ce_full) - 1.0

          # Step 3: 记录学习率状态
          lr = lr_schedule[step]
          results.append((step, lr, rel_ce, ce_full))

      return results

  # WSD 调度下的发现:
  # - 稳定阶段（constant lr=3e-3）：量化误差温和上升 ~1%
  # - 衰减阶段（lr→0）：量化误差急剧飙升 ~20%
  # - 峰值学习率越大（6e-3 > 3e-3 > 1e-3），衰减后量化误差越小
  # - 不同 token 预算（10B-100B）在 cooldown 后有可比的量化误差
  ```

  Weight averaging 介入方式（LAWA）：
  ```
  def lawa_quantization(checkpoints, window=5):
      # 沿训练轨迹滑动平均
      avg_weights = mean(checkpoints[-window:])  # 最近 K 个 checkpoint 均匀平均
      W_q = quantize(avg_weights, method="GPTQ", bits=3)
      # LAWA 的量化误差可比肩甚至优于 lr cooldown 后的模型
      return W_q
  ```

  Hessian 几何分析 pipeline：
  ```
  def hessian_analysis(W, val_data):
      # Hutchinson estimator for Hessian trace
      z = rademacher_sample(dim=W.numel())
      Hv = hessian_vector_product(loss, W, z)
      trace_est = z @ Hv                          # unbiased trace estimator

      # Power iteration for max eigenvalue (sharpness)
      lambda_max = power_iteration(loss, W, n_iters=50)

      # 发现：lr 衰减时 sharpness 和 trace 同时激增
      # 较大峰值 lr → 更平坦的 loss basin → 更低量化误差
      return lambda_max, trace_est
  ```
