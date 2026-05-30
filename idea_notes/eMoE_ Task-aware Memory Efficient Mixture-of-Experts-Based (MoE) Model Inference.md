## eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference

- baseline方法是什么？
  Baseline 是 MoE-based LLM 的标准推理系统（如 vLLM、DeepSpeedFastGen），其中 MoE 模型的所有 expert 在推理前全部预加载到 GPU 显存中（static pre-loading）。该方案直接导致 MoE 模型消耗 4×-14× 于同等 dense 模型的 GPU 显存。另一种 naive 方案是动态加载（dynamic loading）：所有 expert 存放在 CPU，推理时 instant transfer 所需 expert 到 GPU，但实验显示这会增加 3.2×-5× 推理延迟。Pre-gatedMoE 和 MoEInfinity 通过 prefetching 在计算当前层时预取下一层 expert 来 overlap 通信，但仍引入 2.5×-3.5× 的额外延迟。

  **Baseline 全栈执行例子**（以 Mixtral-8x7B 在 4× A100 40GB GPU 上为例）：

  **算法 Pipeline 层**：每个 token 通过 router gate 计算 top-2 expert → 仅激活 2/8 experts per layer。MoE layer 输出：`O = Σ_{i∈top-2} g_i · E_i(x)`。但所有 8 experts 的权重矩阵 (W_in, W_out) 均驻留在 GPU HBM 中。

  **系统框架层**：DeepSpeed-FastGen 或 vLLM 接受请求 → continuous batching → 每个 MoE layer 的 forward pass 执行 router gating → all-to-all dispatch → expert FFN → all-to-all combine → attention。所有 32 MoE layers（Mixtral-8x7B）的 8 experts/layer = 256 expert matrices 全部在 GPU 上。

  **编译框架层**：论文未明确说明。baseline 使用 PyTorch eager execution + DeepSpeed 的 MoE kernel 优化。

  **Kernel 调度层**：DeepSpeed-FastGen 的 MoE kernel 执行 all-to-all 通信（NVLink/PCIe between GPUs）+ expert FFN GEMM。每个 expert 的 weight 已固定在 GPU 上，expert 选择 → 直接 GEMM 无传输开销，但所有 expert 权重占用显存。

  **硬件架构层**：4× A100 40GB GPU（NVLink 互联）+ 128GB CPU host memory。Mixtral-8x7B ~47B 参数全部在 GPU 上：attention 权重 + 32 layers × 8 experts × (W_in+W_out) ≈ 96GB。512 tokens prompt 仅 memory 就占用 ~96GB（接近 4×40GB=160GB 的 60%）。

- 论文方法是什么？如何对应解决Baseline的缺陷？

  eMoE 通过四个协同组件将 expert 从"全量常驻 GPU"变为"按需预测加载 + 周期性复用 + 任务感知过滤 + SLO 感知调度"，从而同时优化记忆体消耗和推理延迟。

  **eMoE 全栈执行例子**（以 Mixtral-8x7B，60% experts loaded，p=40 为例）：

  **算法 Pipeline 层**：
  - Expert Prediction 模型（BERT-XLNet, 0.108B params）学习 expert 路由序列的时序依赖。每个 prompt 的 expert 序列是跨层的 top-k 索引序列 `[e_1, e_2, ..., e_m]`，predictor 基于 consecutive layer 间的 cross-correlation（~0.50）和 consecutive prompt 间的 cross-correlation（0.75-0.95）预测未来 expert。
  - eMoE-A：`f([e_1^{r1}, ..., e_m^{r1}]) → [e_1^{r2}, ..., e_m^{r2}]`，用前一条 prompt 的 expert 分布预测当前。
  - eMoE-L：`f(e_{i-1}^{r1}) → e_i^{r1}`，逐层预测。
  - Predictor memory：仅 0.24%-1.3% of MoE model size。
  - 预测错误时：token 被路由到已加载的 next top-k expert（fallback routing）。

  **系统框架层**（DeepSpeed-FastGen 修改）：
  1. **Task Type Extraction**（CPU）：关键词匹配识别任务类型（SUM/CLSFY/QA/COMP/CONV）。
  2. **Task-aware Request Scheduling**（CPU, Algorithm 1）：从等待队列按 SLO stringiness 排序 → 遍历检查 `t_i = ΔE + (W + n_i·G_i)·c + r_i < SLO` → 贪心调度。G_i 为 profiled 任务特定生成 token 数，运行时递减。
  3. **Task-aware Expert Loading**：对每个 MoE 层计算 `N_i = (ΣW_j + T·W_o) · s · f_i`。s=0 时（任务对该层 routing 不敏感）跳过 expert 加载，直接复用已加载 expert。排序 N_i 后仅加载 top L（L 由 memory budget 决定）。
  4. **Periodic Expert Invocation**：每 40 prompts 调用一次 predictor；中间的 39 prompts 复用已加载 expert。Correlation 分析显示 consecutive prompts 间 correlation 为 0.48-0.55，perplexity 在 ≤60 prompts 内基本不变。

  **编译框架层**：论文未明确说明。

  **Kernel 调度层**：
  - Expert 加载：`torch.Tensor.copy_(non_blocking=True)` 异步 CPU→GPU 传输，与 self-attention layer（non-expert）计算重叠。
  - 同步机制：Python multiprocessing lock per MoE layer + CUDA event 防止使用 stale weights。
  - PCIe 带宽管理：当前 MoE 层的 expert 加载以前一层加载完成为条件（conditioned loading），防止多路并发 DMA 饱和 PCIe 通道。
  - Expert 卸载：不在预测列表中的 expert → 从 GPU 移到 CPU（释放显存）。

  **硬件架构层**（同 baseline）：4× A100 40GB GPU + 128GB CPU host memory + PCIe 总线。60% experts 加载时 Mixtral-8x7B 仅占用约 59GB（vs baseline 96GB），可处理 40× longer prompts 和 4.5× larger batches。

  **Baseline 缺陷 → eMoE 方法设计映射**：

  - **缺陷 1**：所有 expert 全量常驻 GPU 导致 4×-14× 记忆体开销 → **设计 1**：Expert Prediction（BERT-XLNet）基于 recurrent routing patterns 预测并仅加载所需 expert，memory 减少 up to 80%。
  - **缺陷 2**：per-prompt 动态加载导致 3.2×-5× 延迟增加 → **设计 2**：Periodic Expert Invocation（p=40），利用 consecutive prompts 的 high correlation（0.48-0.55）复用 expert，amortize 预测/加载开销至 0.24%-3.11%。
  - **缺陷 3**：所有任务无差别对待，导致不必要的 expert 加载 → **设计 3**：Task-aware Expert Loading，发现 Classification/Comparison 任务即使用 random routing 仍保持 >90% similarity，仅对敏感任务（Conversation/Summarization）精确加载，跳过不敏感任务的预测开销。
  - **缺陷 4**：现有调度器（vLLM, Orca, Sarathi-Serve）不考虑 expert loading latency 和 task-specific characteristics → **设计 4**：Task-aware Request Scheduling 联合 SLO + profiled output length + ΔE expert loading latency 做贪心调度，delay 宽松 SLO 请求以减少对 running requests 的干扰。
  - **缺陷 5**：Pre-gatedMoE/MoEInfinity 的 continuous prefetching 导致 CPU-GPU 带宽争抢 → **设计 5**：eMoE 仅周期性加载（而非每层 prefetch），配合 conditioned loading 避免 PCIe 饱和。（Pre-gatedMoE 2.4×-3.5× slower than eMoE）
