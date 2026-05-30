## DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是将 MoE 专家剪枝重新定义为连续优化问题，提出 Differentiable Expert Pruning (DiEP)，包含三个核心组件：(1) **Intra-layer/Inter-layer 双层次可微分数**：定义 intra-layer importance scores α（每层内各专家的相对重要性）和 inter-layer importance scores β（每层对整体模型的贡献权重），通过 softmax 归一化得到 continuous relaxation；(2) **交替梯度优化**：以 α:β = 3:1 的比例交替更新两个参数组，目标函数为 L = L_ce(y, F'(x; α, β)) + λ·∥F'(x; α, β) − F(x)∥_F（λ=0.01），其中正则项为 reconstruction regularization term，鼓励剪枝后模型与原模型输出一致；(3) **全局排序剪枝**：最终 s_i^(l) = α_i^(l) · β^(l)，全局排序所有专家重要性，按照 sparsity ratio r 统一删除 bottom-K（K = N·L·r）最不重要的专家，实现跨层非均匀剪枝。(4) **Adaptive Expert Skipping 在线推理加速**：在推理时为每个 token 跳过冗余专家计算，γ = γ1 × γ2，其中 γ1 是 calibration data 中 routing weight ratio w_e1/w_e0 的中位数，γ2 是基于 CKA similarity 的专家输出相似度比。当 w_e1 < γ·w_e0 时跳过专家 e1。实验比较：在 Mixtral 8×7B、Mixtral 8×7B-Instruct、Mixtral 8×22B、Deepseek-MoE-16B、Qwen2-57B-14A 五个 MoE 模型上，在 25% 和 50% expert sparsity 下对比 M-SMoE (merge)、Expert Trimming (activation frequency)、NAEE (exhaustive search)、S-SMoE (similarity-based merge) 的 MMLU/OpenBookQA/BoolQ/RTE 等 zero-shot benchmark 性能。消融实验验证 α、β 组件重要性、交替更新比例、λ 超参数、epoch 数、calibration data size 的影响。

- 硬件平台是什么，配置是什么。
  4× NVIDIA A800 GPU。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) Mixtral 8×7B（32 MoE layers, 8 experts/layer, top-2 activation）；(2) Mixtral 8×7B-Instruct（同架构指令微调版）；(3) Mixtral 8×22B（56 MoE layers, 8 experts/layer, top-2, 141B total/39B activated）；(4) Deepseek-MoE-16B（28 layers, 64 experts/layer, 2 shared + 6 routed per token）；(5) Qwen2-57B-14A（28 MoE layers, 64 experts/layer, top-8 activation）。Calibration 数据：C4 dataset 随机采样 128 条序列用于 differentiable search。Evaluation benchmarks：MMLU（57 subtasks, 4 domains）、OpenBookQA、BoolQ、RTE；附录中额外包含 ARC-c、ARC-e、HellaSwag、WinoGrande。Domain-specific 验证：GSM8K 数学推理数据集（使用 C4 和 MATH 两种 calibration data）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供开源代码链接。方法基于 HuggingFace Transformers + LM Evaluation Harness (lm-eval-harness) 实现。核心算法流程如下：
  ```
  # Algorithm: DiEP Differentiable Expert Pruning
  Input: 校准数据集 D_cal (128 samples from C4), 完整 MoE 模型 F,
         初始化 α_i^(l) = 1, β^(l) = 1, λ = 0.01, epochs = 10
  
  for epoch in 1..E:
    for batch in D_cal:
      # Forward with continuous relaxation
      ᾱ_i^(l) = softmax(α_i^(l))                     # [N_experts] per layer
      y'^(l+1) = β^(l) · Σ_i ᾱ_i^(l) · FFN_i(x^(l)) # Eq.5, 加权专家输出
      L = L_ce(y, F'(x; α, β)) + λ · ∥F'(x; α, β) − F(x)∥_F  # Eq.7
  
    # Alternating update (3:1 ratio)
    for step in 1..3:                                # α updates (3 steps)
      α ← α − η_α · ∇_α L(α, β)                     # fix β, update α
    for step in 1..1:                                # β updates (1 step)
      β ← β − η_β · ∇_β L(α, β)                     # fix α, update β

  # Global pruning after optimization
  for each expert i in layer l:
    s_i^(l) = α_i^(l) · β^(l)                        # Eq.10, global importance
  K = N_layers · N_experts · r                        # total experts to prune
  P = bottom-K indices sorted by s_i^(l) globally      # select least important
  m_i^(l) = 0 if i ∈ P else 1                        # Eq.11, pruning mask
  
  # Adaptive Inference Skipping
  γ1 = median(w_e1 / w_e0) over calibration data     # per-layer routing ratio
  γ2 = ρ(y_e0, y_e1) / mean(ρ(y_ei, y_ej))          # CKA similarity ratio
  γ = γ1 · γ2                                         # per-layer skip threshold
  During inference: if w_e1 < γ · w_e0, skip expert e1
  ```
  关键张量维度：α ∈ R^(L×N)（L 层，N experts/layer），β ∈ R^L。剪枝后 α 仅需约 0.01% 额外参数量。Pruning time: Mixtral 8×7B 仅 0.23h（vs NAEE 1.31h 的 exhaustive search）。Deepseek-MoE-16B pruning 仅 0.28h（vs NAEE ≈94000 days 因 exhaustive search 不可行）。50% sparsity 下 Mixtral 8×7B 推理获得 1.28× speedup、48% GPU memory reduction，保留约 92% 原模型性能。
