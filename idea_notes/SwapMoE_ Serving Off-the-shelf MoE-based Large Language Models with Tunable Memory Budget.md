## SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget

- baseline方法是什么？
  Baseline 方法包括两类：(1) **On-demand Loading（内存交换）**——按需通过 PCIe 从外部存储加载 MoE layer 的 expert 参数到 GPU memory，推理完成后释放。每次 MoE layer 参数传输引入显著延迟（6.2×-8.9×），参数传输占推理时间的绝大部分，即使异步加载也无法避免 I/O 阻塞计算（因为 loading 始终慢于 computation）；(2) **Expert Pruning（专家剪枝）**——基于 expert 权重的 magnitude（||E_i||）永久剪除不重要的 experts，缩小模型后直接推理。剪枝后不可恢复，导致显著的准确率损失（例如 SwitchT-32 减少 30% memory 时准确率下降 14%），且需要额外训练来恢复性能。
  全栈执行例子（Baseline: On-demand Loading，SwitchT-16，Jetson AGX ORIN，summarization task）：
  - **算法Pipeline层**：输入 token sequence X → Self-Attention → Router gating G(x) = softmax(W_r @ x) → 路由到 expert k = argmax G(x) → 此时 E_k 不在 GPU memory 中 → 触发 PCIe 加载 E_k 参数（W_in, W_out）→ 加载完成后执行 E_k(X) → 完成后释放 E_k → 下一个 MoE layer 重复此过程。每层需等待 expert 参数传输，expert 参数大小 = 2 × d × d_ff × 4 bytes（FP32），在 14 GiB 模型下单层传输可达数 GB。Baseline 准确率无损失（使用完整 expert set），但延迟极高。
  - **系统框架层**：HuggingFace Transformers 标准 MoE layer forward，每层执行前检查 expert 是否在 device，不在则触发 load_state_dict。无异步机制或 IO/计算 overlap 优化，expert loading 完全阻塞 computation pipeline。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准 cuBLAS GEMM 执行 expert FFN 计算。无 expert 选择或跳过逻辑，无 IO 调度。
  - **硬件架构层**：Jetson AGX ORIN，GPU-CPU 通过 PCIe 连接，expert 参数存储在 CPU memory。PCIe 带宽限制（10-30 GiB/s）成为瓶颈。
  Baseline 的核心缺陷：(a) On-demand loading 用延迟换内存——延迟开销不可接受（6.2×-8.9×）；(b) Pruning 用准确率换内存——永久性准确率损失无法恢复；(c) 两者都无法实现可调的 memory-accuracy-latency tradeoff。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法通过三个层面解决 baseline 缺陷：(1) **Virtual Experts 动态子集维护**——不再按需加载单个 expert 或永久剪枝，而是维护一个动态更新的 expert 子集（Virtual Experts），基于数据分布 locality（连续 token 语义相关、同一用户 conversation 上下文连续）预测未来最可能需要的 experts，预加载到 GPU memory。这样，每次推理的计算量 = 小型 MoE 模型（仅 VE 参与），但大模型的能力保留（每个 expert 都有机会参与）；(2) **Importance-aware Expert Selection + Masked Gating**——设计 expert importance score = Σ ||x|| · |G(x)_i| · ||E_i||（综合 token norm、routing weight 和 expert weight magnitude），高效量化每个 expert 对当前数据分布的贡献。使用 Masked Gating 将推理请求重定向到 VE，避免运行时 routing 到不在内存中的 expert；(3) **Profiling-guided Memory Planning + Genetic Search**——离线 profile 每个 expert 的性能特征（memory/latency/loading time/IO bandwidth），训练小型 DNN 建模 config→accuracy 映射，用遗传算法在巨大搜索空间中搜索最优层间 expert 分配方案（而非枚举 12^16 种组合）。
  全栈执行例子（论文方法：SwapMoE，SwitchT-16，Jetson AGX ORIN，summarization task，memory budget 4.7 GiB）：
  - **算法Pipeline层**：输入 token sequence X → 对每个 MoE layer l：(a) Router 计算 gating_scores = softmax(W_r @ X)；(b) Masked Gating：mask[i] = 1 if i ∈ VE else 0，masked_scores = normalize(gating_scores ⊙ mask)；(c) Expert 计算：仅 i ∈ VE 执行 E_i(X)，output y = Σ masked_scores[i] · E_i(X)；(d) 收集 importance score：对每个 expert E_i 和其处理的 tokens X_i，importance = Σ ||x|| · |masked_scores[i]| · ||E_i||_F；(e) 每 frequency 个 sample 后：异步加载 top-k experts 到 GPU memory，释放 bottom experts。结果：memory 14.2 GiB → 4.7 GiB (67% reduction)，latency 降低 50%（因为仅计算 VE subset），ROUGE-2 仅下降 0.041。
  - **系统框架层**：HuggingFace Transformers 中修改 MoE layer forward：(a) 插入 Runtime Scheduler——在 router 和 expert FFN 之间插入 VE selection 和 Masked Gating；(b) Amortized Expert Loading——跨多个 sample 摊销 expert 加载开销（图6 ii），避免每 sample 后同步更新；(c) Asynchronous Expert Loading——使用 async copy engine 加载 expert 参数，与 computation pipeline overlap。IO overhead 极低（peak ~40 MiB/s vs PCIe 10-30 GiB/s）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：standard GEMM kernel 不变，但计算量大幅减少——仅计算 VE subset 的 experts。Masked Gating 避免了 runtime routing 到缺失 expert 的 penalty。IO 调度通过 async copy engine 实现。
  - **硬件架构层**：Jetson AGX ORIN 和 Jetson Nano。Expert 参数分层存储：VE 在 GPU memory（快速访问），其余 experts 在 CPU memory 或 SSD（通过 PCIe 或存储总线访问）。每个 expert loading 的时间被 precise profiling 并纳入 offline planning。
  方法 vs Baseline 对比核心差异：(a) VE 动态子集 vs Static pruning——保留所有 expert 参数完整性，按需 swap 而非永久丢弃；(b) Amortized + Async loading vs On-demand synchronous loading——将延迟峰值从 per-layer 平摊到 per-N-samples，通过 async I/O 与计算 overlap；(c) Genetic search configuration vs Manual fixed allocation——在 12^16 搜索空间中找到接近最优的层间 expert 分配，而非均匀分配或手工调优；(d) Memory-Accuracy-Latency tunable tradeoff vs 二元选择（全 accuracy 高延迟 vs 低 accuracy 低内存）。
