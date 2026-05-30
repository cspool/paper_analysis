## MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是MoDES，一个training-free的MoE MLLM动态expert skipping框架。包含三个核心算法组件：(1) GMLG（Globally-Modulated Local Gating）——将离线校准的全局逐层重要性因子α^{(l)}（通过KL divergence量化整层跳过对最终输出的影响）与局部routing概率π_i^{(l)}相乘得到expert重要性分数s_i^{(l)} = α^{(l)} · π_i^{(l)}；(2) DMT（Dual-Modality Thresholding）——为text token和vision token分别设置跳过阈值τ_t和τ_v，根据s_i^{(l)} < τ_t·I_t + τ_v·I_v判定该expert是否跳过；(3) Frontier Search——利用f(τ_t,τ_v)和g(τ_t,τ_v)的单调性在O(ND)时间内找到最优阈值对(τ_t, τ_v)，替代O(ND²)的naive exhaustive search，搜索时间从数天降至数小时。

  实验比较的算法baseline包括：NAEE（routing probability-based skipping，单层内阈值判定）、MC-MoE（attention-aware expert protection + skipping）、DiEP（differentiable expert pruning + adaptive skipping）、直接降低top-k的k值。所有baseline从LLMs场景适配到MLLMs的top-k（k>2）setting。

- 硬件平台是什么，配置是什么。
  8×H200 GPU用于calibration、search和accuracy evaluation；单张H200 GPU用于inference speed测量。Software: PyTorch transformers库，flash-attention2，lmm-eval评估框架。为inference speedup编写了自定义CUDA kernel实现MoE层内的双模态阈值判定和Group GEMM。

- 模型是什么。数据集和bench分别是什么。
  模型：3个MLLM系列——Kimi-VL-A3B-Instruct（64 experts/layer, k=6, 26 MoE layers）、Qwen3-VL-MoE-30B-A3B-Instruct（128 experts/layer, k=8）、InternVL-3.5-30B-A3B-HF（128 experts/layer, k=8）、InternVL-3.5-GPT-OSS-20B-A4B-Preview-HF（32 experts/layer, k=4）。数据集：GQA（1024 samples用于calibration和search）；8个image understanding benchmarks（TextVQA, ChartQA, MMStar, MMBench, MMVet, MME, RealWorldQA, COCO2017-Cap）+ 5个video understanding benchmarks（MVBench, EgoSchema, VideoMME, LongVideoBench, VideoM-MMU）。评估框架：lmm-eval，MMBench和MMVet使用DeepSeek-V3.1进行生成文本评分。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/ModelTC/MoDES
  
  MoDES算法pipeline（以Qwen3-VL-MoE-30B-A3B-Instruct的l-th MoE层，top-k=8个expert为例）：

  **离线阶段（Calibration + Search）：**
  ```
  # Step 1: Calibrate global importance α^{(l)} for each MoE layer
  C = randomly sample 1024 examples from GQA dataset  # calibration set
  for l in 1..L:  # for each MoE layer
      for each example c_j in C:
          prob_j = original_model(c_j)          # full model output distribution
          prob_j^{(l)} = model_with_layer_l_skipped(c_j)  # skip all experts in layer l
      α^{(l)} = (1/N) * sum_j D_KL(prob_j || prob_j^{(l)})
      # α^{(l)}大 → 浅层贡献大 → 跳过影响大 → 应少跳过
      # α^{(l)}小 → 深层贡献小 → 跳过影响小 → 可多跳过
  normalize: α̃^{(l)} = α^{(l)} / sum_{l'=1}^L α^{(l')}

  # Step 2: Frontier Search for optimal (τ_t, τ_v)
  B = {τ^{(1)}, τ^{(2)}, ..., τ^{(D)}}  # D=100 grid points in (0,1)
  target_skip_ratio = ρ  # e.g., 0.85 for 88% skipping
  for each (τ_t=q, τ_v=p) pair on frontier (Algorithm 1):
      evaluate f(τ_t, τ_v) = KL divergence between original and skipped model
      evaluate g(τ_t, τ_v) = fraction of experts skipped
      # Monotonicity: larger thresholds → more skipping → higher KL divergence
  (τ_t*, τ_v*) = argmin f(τ_t, τ_v) s.t. g(τ_t, τ_v) ≥ ρ
  # Time complexity: O(ND) vs naive O(ND²), ~45x speedup
  ```

  **在线推理阶段（per-token, per-MoE-layer）：**
  ```
  # Input: token x^{(l)} ∈ R^d at layer l, with modality indicator
  # x^{(l)} can be text token (I_t=1, I_v=0) or vision token (I_t=0, I_v=1)

  # Step 1: Standard MoE routing
  r^{(l)} = Router_l(x^{(l)})                    # routing logits: [M]
  π^{(l)} = softmax(r^{(l)})                      # routing probabilities: [M]
  S^{(l)} = topk_indices(π^{(l)}, k)              # top-k expert indices

  # Step 2: GMLG - compute importance scores with global modulation
  for i in S^{(l)}:
      s_i^{(l)} = α̃^{(l)} * π_i^{(l)}              # Eq.(3): global × local importance
      # α̃^{(l)} pre-computed offline, π_i^{(l)} from router

  # Step 3: DMT - modality-specific expert skipping
  τ = τ_t * I_t + τ_v * I_v                        # select threshold by modality
  active_experts = {i ∈ S^{(l)} : s_i^{(l)} ≥ τ}   # Eq.(5): keep only important experts

  # Step 4: Compute output with only active experts
  y^{(l+1)} = sum_{m ∈ active_experts} π_m^{(l)} · Expert_m^{(l)}(x^{(l)})
  # In practice: skipped experts → sentinel expert ID → filtered out during dispatch/gather
  ```

  关键设计要点：
  - GMLG在inference时无额外开销——α^{(l)}预计算，s_i^{(l)}仅需一次乘法
  - DMT对vision token的τ_v < τ_t（vision token expert冗余度更高），跳过更多vision experts
  - Frontier search exploit单调性：更大的τ → 更多跳过 → g递增、f也递增，只需O(ND)搜索
  - 校准数据鲁棒：GQA/COCO/VMMMU上α^{(l)}趋势一致，性能差异<1%
