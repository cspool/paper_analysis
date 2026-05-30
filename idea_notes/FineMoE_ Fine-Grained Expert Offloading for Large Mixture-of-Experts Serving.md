## FineMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving

- baseline方法是什么？
  - **Coarse-grained Expert Offloading**（以 MoE-Infinity、ProMoE、Mixtral-Offloading、DeepSpeed-Inference 为代表）：现有 expert offloading 方法在 **request-level（粗粒度）** 粒度上追踪和预测 expert activation pattern：
    - **MoE-Infinity**：track request-level expert hit counts (Expert Activation Matrix)，synchronous expert prediction and prefetching。prefetch distance > 1 层时无法观测到足够的 expert trajectory history，只能对所有层使用 request-level 聚合统计（最流行 experts）进行 prefetching。
    - **ProMoE**：stride-based speculative prefetching，需要 per-layer NN predictor 训练（millions of params per layer），训练和 retraining 开销大。
    - **Mixtral-Offloading**：layer-wise speculative prefetching + LRU cache。synchronous prefetching with prefetch distance = 1（无法隐藏 prefetch latency）。
    - **DeepSpeed-Inference**：layer-wise parameter offloading without expert awareness，pure on-demand loading（无 prefetch），latency 最高。
  - **Coarse-grained 的三个核心缺陷**：
    1. **Insufficient latency-memory trade-off**：要么低延迟大内存（MoE-Infinity），要么高延迟小内存（DeepSpeed-Inference），无法同时优化。
    2. **Low expert hit rate**：request-level 聚合抹去了 iteration-level 的细粒度 expert selection pattern（图 3a 的 heatmap 对比：iteration-level 有明显 pattern，request-level 熵高、可预测性低）。Shannon entropy 分析表明 coarse-grained pattern 的 entropy 显著高于 fine-grained（图 3b），且 entropy 随 iteration 累积逐渐升高并 plateau（图 3c），说明 request-level 聚合使 expert pattern 越来越不可预测。
    3. **Ignorance of MoE models' and prompts' heterogeneity**：不同 MoE 模型（Mixtral-8×7B 8 experts/layer vs Qwen1.5-MoE 60 experts/layer）和不同 prompts（semantic diversity）在 one-fits-all 方式下被同等对待，失去了按模型和 prompt 特征自适应优化的机会。
  - **全栈执行例子**（Baseline MoE-Infinity, Mixtral-8×7B, LMSYS-Chat-1M）：
    - **算法 Pipeline 层**：Gate network 输出 top-K expert selection → request-level expert activation count 在所有 iteration 上聚合 → 以 historical request 的 aggregated activation counts 作为 prediction signal。因为 decoder-only MoE + load-balancing loss 导致 balanced routing（expert activation 接近均匀分布），aggregated counts 的预测能力弱。
    - **系统框架层**：MoE-Infinity on HuggingFace Transformers。Expert Activation Matrix stored in CPU memory → synchronous prediction before each MoE layer → expert 从 CPU to GPU via PCIe → forward computation。Synchronous design 导致 prefetching latency 无法与 computation overlap。
    - **编译框架层**：论文未明确说明。
    - **Kernel 调度层**：CUDA Runtime API for expert memory management。Expert Cache 使用 LFU 策略。Multi-GPU EP with round-robin expert distribution。Synchronous prefetching 阻塞 forward：每层必须先完成 expert prediction + prefetching 才能执行 computation。
    - **硬件架构层**：6× RTX 3090 24GB + PCIe 4.0 32GB/s。Expert miss → on-demand loading latency = T_e per miss 直接增加到 iteration latency。Request-level 粗粒度预测导致 expert hit rate 低 → 大量 on-demand loading → 高 inference latency。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FineMoE 方法**：从 coarse-grained (request-level) 切换到 **fine-grained (iteration-level)** expert offloading，通过三个核心创新：
    1. **Expert Map（取代 Expert Activation Matrix）**：记录每个 iteration 中每层 gate network 输出的完整概率分布 P_l^{(i)} ∈ R^J（而非 binary hit count）。保留 gate network 对每个 expert 的 confidence/preference 信息，支持退化恢复 coarse-grained 信息（top-K + 聚合）。粒度从 request-level 变为 iteration-level，解决"aggregation destroys predictability"的问题。
    2. **Dual-Similarity Expert Map Search**：利用两种 fine-grained 指标搜索最准确的 historical expert map：(a) **Semantic similarity**——semantic embeddings（embedding layer output）的 cosine similarity。基于"语义相似 prompts 有相似 expert 选择"的假设，用于 prefetch distance d 以内的初始层（尚无 trajectory history）。Pearson correlation 验证 semantic similarity score 与 expert hit rate 正相关；(b) **Trajectory similarity**——已观察到的前 (l-d) 层 expert probability distributions 的 cosine similarity。用于第 l ∈ [d+1, L] 层。随 iteration 推进，越来越多的 trajectory history 可被利用，提高预测准确度。
    3. **Similarity-aware Adaptive Expert Selection**：并非固定 top-K 选择，而是根据 search confidence 动态调整。δ_l = Clip(1 - similarity_score, 0, 1)——高 similarity 时 δ 低，只需选 highest-probability 的少数 experts；低 similarity 时 δ 高，选更多 experts 防止 miss。类似 confidence-based 的 adaptive exploration：系统自信则 lean（省内存），不自信则 wide（保准确）。
    4. **Asynchronous Publisher-Subscriber Architecture**：将 map searching + expert prefetching 与 inference forward pass 解耦（弥补 MoE-Infinity synchronous design 的缺陷）。Inference process 作为 Publisher 持续写入 context → Expert Map Searcher 作为 Subscriber 异步消费 context 并 prefetch。
  - **对应解决 Baseline 缺陷**：
    - **Coarse-grained → Fine-grained** → iteration-level pattern tracking（expert map）降低 Shannon entropy（图 3b），提高 pattern predictability → expert hit rate 提升 39%（vs SOTA）。
    - **No semantic awareness → Semantic-based search** → 利用 input prompt 的 semantic embedding 为缺少 trajectory history 的初始层提供有效的 expert prediction → 解决 prefetch distance 内无法观测 trajectory 的问题。
    - **Fixed prefetching → Adaptive δ prefetching** → 根据 search confidence 动态调整 prefetch 量，high confidence 时节省 GPU memory，low confidence 时增加 coverage → 在 latency-memory trade-off 上找到更优均衡点（6GB cache limit 时 TPOT 降低 16-36% vs baselines）。
    - **Synchronous → Asynchronous** → map searching 和 prefetching 的 overhead 不进入 critical path → iteration overhead < 1%（< 50ms）。
    - **One-fits-all → Model/prompt heterogeneity aware** → 不同 MoE 模型（Mixtral/Qwen/Phi）independent profiling 确定最优 prefetch distance (3/6/4) → 适应不同 expert 数量和 layer 深度的模型。
  - **全栈执行例子**（FineMoE, Mixtral-8×7B, LMSYS-Chat-1M）：
    - **算法 Pipeline 层**：Embedding layer → semantic embedding extraction → cosine similarity search 在 Expert Map Store（1K maps）→ 前 d=3 层用 semantic match 的 expert map 指导 prefetch。Iteration i, Layer 4：收集 P_1, P_2, P_3 的 probability trajectory → cosine similarity with historical maps → 选 best match 的 P_4 → δ_4 = Clip(1-score_traj, 0, 1) → 选 experts 直至 Σp ≥ δ_4 且 count ≥ K=2 → prefetch E_prefetch。每层 Gate network 仍按原始 top-K 做最终 expert 选择（lossless），prefetching 只是预测性加载。
    - **系统框架层**：HuggingFace Transformers + FineMoE Expert Map Store (Python/PyTorch/NumPy) + Expert Cache (C++, CUDA)。Publisher-Subscriber 异步通信。Prefetch distance d=3 使 map searching + prefetching latency 被 overlap 到 attention + gate + expert 计算中。
    - **编译框架层**：论文未明确说明。
    - **Kernel 调度层**：GPU task pool 异步调度 expert prefetch 任务 → PCIe 4.0 32GB/s cudaMemcpyAsync(host→device) → Expert Cache hash map 更新。LFU + probability-based eviction (PRI^{evict} = 1/(p*freq))。Expert miss 时暂停所有 prefetching → 立即 on-demand cudaMemcpy → 恢复 prefetching。Prefetch priority = p/(l-l_now) 确保近层的 high-probability experts 优先。
    - **硬件架构层**：6× RTX 3090 24GB + NVLink + PCIe 4.0 32GB/s。Expert Map Store < 200MB CPU memory（1K maps）。Round-robin EP 将 experts 分布到 6 个 GPU → per-GPU cache 独立管理。FineMoE 在 6GB GPU cache limit 时 TPOT 降低 16-36% vs baselines，在 48GB+ 时所有方法趋近（因足够 cache 几乎所有 experts）。A100 (80GB, single GPU, no EP) 上提升减小（fast inference + single GPU 减少了 offloading 收益），但仍一致优于 baselines。
