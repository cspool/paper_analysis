## MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是自定义CUDA kernel用于MoE MLLM推理中的dual-modality thresholding和高效的expert计算执行。具体包括：(1) 在router kernel内部实现双模态阈值判定——在计算router logits和top-k后，使用branch-free masked comparison与modality-specific threshold比较，直接将跳过的expert路由设为sentinel expert ID（如M+1），不引入额外的kernel launch或独立的decision pass；(2) Sentinel-aware dispatch/gather——在MoE dispatch/gather阶段自动过滤sentinel entries，跳过专家加载和计算；(3) Group GEMM执行——使用Grouped General Matrix Multiplication将所有活跃experts的矩阵乘法合并到单个统一的kernel launch中并发执行，每个expert的计算为独立的sub-task；(4) Offline profiling + kernel tuning——对不同的代表性激活模式进行离线grid search，确定最优的kernel tile sizes，确保不同动态负载下的高计算吞吐。实验比较baseline kernel与原始模型（k=8/6/4 top-k routing）的prefill/decoding延迟和吞吐量（tokens/s）。

- 后端平台是什么，配置是什么。
  单张NVIDIA H200 GPU（用于inference speed测量），8×H200 GPU用于calibration、search和accuracy evaluation。Software: PyTorch、transformers库、flash-attention2。自定义CUDA kernels用于MoE层thresholding和Group GEMM。

- 评估性能的软件/脚本是什么。修改了什么。
  使用自定义CUDA kernels测量实际wall-clock inference speed。修改：(1) Router kernel——在router内部嵌入thresholding逻辑：router → top-k → apply modality-specific threshold via masked comparison → assign sentinel IDs to skipped routes；(2) MoE dispatch/gather kernel——增加sentinel filtering逻辑，通过检查expert ID是否等于M+1来过滤；(3) Group GEMM kernel——使用离线profiled的tile sizes，支持动态expert激活模式下的并发矩阵乘法。关键设计决策：thresholding逻辑嵌入现有kernel（不增加额外kernel launch），sentinel filtering仅需warp-level的少量元素操作（overhead最小）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/ModelTC/MoDES

  评估原理：
  1. Build：在8×H200 GPU环境中编译自定义CUDA extensions（router + MoE dispatch/gather + Group GEMM kernels）
  2. 加载pre-calibrated GMLG参数α̃^{(l)}和frontier-search找到的最优阈值(τ_t*, τ_v*)
  3. 使用prefill batch size=8、decode sequence length=1024进行inference
  4. 测量prefill time（ms，含所有token的首次forward pass）和decode time per iteration（ms，单token自回归生成）
  5. 计算speedup = original_time / MoDES_time

  全过程（以Qwen3-VL-MoE-30B-A3B-Instruct在单H200上，88% expert skipping ratio，decode阶段为例）：
  ```
  Host: 加载模型 + pre-computed (α̃^{(l)}, τ_t*, τ_v*)
  
  对于每个decode iteration（单个text token）:
  For each transformer layer l in 1..L:
    ┌─ Attention Layer (standard) ─────────────────────────────────┐
    │  Q = X @ W_Q; K = X @ W_K; V = X @ W_V                     │
    │  flash-attention2 → attention_output                         │
    │  X = RMSNorm(attention_output + residual)                    │
    └──────────────────────────────────────────────────────────────┘
    
    ┌─ MoE FFN Layer (MoDES customized) ──────────────────────────┐
    │                                                               │
    │  Router Kernel (fused with thresholding):                    │
    │    ① r = router(X)                        // 128 experts    │
    │    ② π = softmax(r)                                         │
    │    ③ topk = topk_indices(π, k=8)          // 8 candidates   │
    │    ④ for i in topk:                                         │
    │         s_i = α̃^{(l)} · π_i              // pre-computed α̃  │
    │    ⑤ τ = τ_t (text token)                                   │
    │    ⑥ mask = (s_i < τ)  // branch-free comparison            │
    │    ⑦ topk[i] = mask ? M+1(sentinel) : topk[i]              │
    │    → 输出: topk with sentinel entries for skipped experts    │
    │                                                               │
    │  MoE Dispatch/Gather:                                        │
    │    ① for expert_id in topk:                                  │
    │         if expert_id != M+1:                                 │
    │           dispatch token to expert_id's input buffer         │
    │    ② sentinel entries automatically filtered — no compute    │
    │                                                               │
    │  Group GEMM Kernel (single launch for all active experts):   │
    │    ① active_experts = unique(topk) - {M+1}                  │
    │    ② GroupedMatMul(                                         │
    │         X_inputs: [X_active1, X_active2, ...],              │
    │         weights: [W_expert_active1, W_expert_active2, ...]  │
    │       )                                                      │
    │       → Each expert as independent sub-task                 │
    │       → Tile sizes from offline profiling grid search       │
    │       → Single kernel launch, concurrent execution          │
    │    ③ weighted sum: y = Σ π_i · E_i(X)                       │
    └──────────────────────────────────────────────────────────────┘
  
  输出性能：
    - Prefill speedup: ~2.16× (batch=8, Kimi-VL-A3B-Instruct)
    - Decode speedup: ~1.26× (seq_len=1024, Kimi-VL-A3B-Instruct)
    - Qwen3-VL-MoE-30B: ~2.03× prefill, ~1.24× decode
    - 跳过88% expert仅保留小部分活跃expert计算，masked comparison和sentinel filtering开销<1%
    - Decode speedup小于prefill的原因是：(i) decode阶段为memory-bound，(ii) decode仅处理text token，跳过率低于prefill的vision+text混合
  ```

  关键kernel设计要点：
  - Branch-free masked comparison：避免warp divergence，所有threads执行相同操作
  - Sentinel filtering in dispatch：在expert输入buffer分配阶段即过滤，不浪费计算资源
  - Group GEMM offline profiling：由于expert activation pattern因token而异，通过pre-profile多种代表性pattern确定最优tile sizes，运行时按最接近pattern选择
  - Dequantization集成（与MC-MoE结合）：支持2.5-bit和1.5-bit权重，MoDES+量化可达到~10.67×压缩比
