## MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts

- baseline方法是什么？
  Baseline 是标准 Hugging Face Transformers 的 device_map offloading 和 Mixtral-Offloading。两者均采用反应式策略管理 expert 参数——GPU 计算到某个 MoE layer 时，同步触发 PCIe H2D 传输所需的 experts，GPU compute units 在此期间停滞等待。
  全栈执行例子（以 Phi-MoE 在 A100-40G 上解码一个 token 为例）：
  - **算法层**：Standard autoregressive decoding。token hidden state h → Self-Attention (FP16, on-GPU) → MoE layer: Router(W_gate * h) → softmax → top-2 selection (say Expert 5, Expert 7) → **I/O Stall Begins**: GPU 发起从 host DRAM 加载 Expert 5 和 Expert 7 的 FP16 权重（每个 expert ~6400×4096×3 ≈ 150MB FP16）→ PCIe 4.0 x16 传输耗时 ~9.4ms → GPU compute: gate_proj + up_proj + SiLU + down_proj (~0.2ms) → token output。Figure 4 profile 显示 Mixtral-8x7B 中 Memory 操作（主要是 PCIe 传输）占总时间 98.9%，GPU compute < 15%。
  - **系统框架层（Serving）**：HuggingFace Transformers device_map 将 MoE expert 权重静态映射到 CPU RAM，GPU 端无 expert cache 或预取逻辑，每 token 每层触发同步 PCIe 传输。Mixtral-Offloading 引入 per-expert on-demand swapping 和 LRU cache，但 cache 策略是反应式的（LRU 响应已发生的访问模式而非预测未来），在高 entropy expert activation（图 5: Qwen-1.5MoE 每层激活熵接近理论最大值）下命中率仅 29.2%（16GB 配置, expert capacity=6）。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch + CUDA。
  - **Kernel调度层**：标准 PyTorch FP16 GEMM kernels (cuBLAS)，fine-grained MoE 中单个 expert 的 GEMM 矩阵太小（Qwen2-MoE: inter_dim=1408），GPU SM utilization 很低。Marlin 量化后端在 MoE 场景同样面临 kernel launch overhead 问题，甚至慢于 PyTorch FP16 baseline（图 11）。
  - **硬件架构层**：NVIDIA A100-40GB GPU + CPU DRAM + PCIe 4.0 x16（32GB/s 双向理论带宽）。Baseline 的核心瓶颈：数据依赖（expert selection 必须在 token 的 attention hidden state 产生后才能确定）→ I/O 在关键路径上 → GPU compute unit 闲置率 > 85%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoE-SpeQ 提出 **quantized speculative decoding × expert offloading co-design**——用 INT4 量化 MoE 模型（GPTQ）作为高速 draft model，在 I/O latency 期间做 useful computation（生成 draft tokens + 预测 expert activation patterns），将预测转换为 lookahead-driven prefetching 以隐藏 PCIe 传输延迟。
  全栈执行例子（同一 Phi-MoE token 在 MoE-SpeQ 下的执行路径）：
  - **算法层（Speculative Decoding with Quantized Draft）**：前一步 target verify 完成后共享 KV cache → Draft model (INT4, fuseMoE kernel, on-GPU) 自回归生成 k 个候选 token。每 draft token 的 MoE forward：Router(FP16, on-GPU) → softmax → top-2 → ELB entry = (expert_id, gating_score)。关键设计：(a) **Hybrid Precision**——Router/Attention/Shared Experts 保持 FP16 以保证 routing 保真度（router 量化误差通过 softmax 放大导致误路由→ELB 污染→cache miss）；MLP expert 主体 INT4 激进压缩速度和内存（43% VRAM 节省: 13.40GB→7.68GB）。(b) **KV Cache Sharing**——draft 在 target 的高精度 KV cache 上运行而非独立 cache，直接提升 token 接受率（>90% vs Eagle 的 80%）。(c) **Quantized Draft as Expert Predictor**——INT4 量化模型预测 target 的 top-4 expert selection 达 90.9% 准确率，优于专用 one-layer-ahead predictor (84.7%)，且单次 forward 预测所有层。
    **解决 Baseline 缺陷(1)**：将不可预测的 expert selection 变为可预测——量化 draft 的 90.9% fidelity 使系统获得 k 步 future token 的 expert 需求 lookahead，打破"必须先算 attention 才知道要哪些 experts"的串行依赖。
  - **Serving调度层（Expert Scheduler + Speculative Governor）**：
    - Expert Scheduler 的三阶段流水线：Phase I 利用 cache hits（locality-aware priming）→ Phase II 对 ELB 中部高 confidence 条目选择性预取（adaptive bandwidth-guided）→ Phase III 对尾部全部缺失 experts aggressive prefetch（cache saturation）。lookahead-aware eviction 替换最不可能被后续使用的 expert。
    - Speculative Governor 的 Amortization Roofline Model: 定义两个 Roof——Compute Roof（horizontal, Θ_max when I/O perfectly hidden）和 I/O Roof（sloped, Θ = B_PCIe × I_amort）。在线搜索 argmax_k Θ(k) = (Σ∏p_j) / [max(T_draft(k), T_pcie,init) + T_pcie,new(k) + T_verify(k+1)]，受离线 SLO 约束 k_SLO。k 的选择权衡：k 大 → amortization 效果好但 expert union 大（VRAM 压力 + 若草稿频繁被 rejected 则浪费计算）；k 小 → overhead 低但 I/O hiding 不足。
    **解决 Baseline 缺陷(2)**：将"等待 I/O → 传输 → 计算"的串行执行变为"草稿生成（与初始 I/O overlap）→ 预取（与草稿 overlap）→ 验证（无 I/O stall）"的流水线执行。Figure 13 显示 Phi-MoE 上 TPOT 从 536.7ms (Mixtral-Offloading-SC) 降至 163.1ms (3.3× speedup)。
  - **Kernel调度层（fuseMoE CUDA Kernel）**：细粒度 MoE 中每个 expert 的 GEMM 维度太小（K=1408, N=2048），单独 launch 无法占满 GPU SM。fuseMoE 将 per-layer 所有 expert 的 gate_proj + up_proj + SiLU + gate×up + down_proj 融合为单次 kernel launch，batch 不同 expert 的 token hidden states 使有效矩阵维度增大 → GPU occupancy 提升 → kernel launch overhead 减少。消融显示 fused kernel 贡献 31.8% speedup (Fig. 消融: w/o fused kernel = 68.2% of full speed)。
    **解决 Baseline 缺陷(3)**：解决细粒度 MoE 下量化推理 kernel 利用率低的问题，使 draft 阶段足够快（开销 < I/O latency），从而整个 speculative pipeline 有意义。
  - **编译框架/硬件架构/芯片设计**：论文未明确说明。
