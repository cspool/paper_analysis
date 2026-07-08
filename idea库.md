
## Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices

- baseline方法是什么？
  - Baseline 方法有三类：
    1. **FMD (Federated MoE fine-tuning with offloading)**：将 inactive experts 从 GPU offload 到 CPU RAM，需要时动态加载回 GPU。所有 experts 参与 forward/backward，保证精度但引入大量 CPU-GPU I/O 延迟。
    2. **FMQ (Federated MoE fine-tuning with quantization)**：所有 expert 参数从 FP32 量化到 INT4，使 participant 能在本地加载完整 MoE 模型。但量化误差在 backprop 中累积导致收敛不稳定。
    3. **FMES (Federated MoE fine-tuning with expert selection)**：按 expert activation frequency 选出高频 activated experts 进行 fine-tuning，丢弃低频 experts（类似 FedMoE [50] 的做法）。
  - Baseline 全栈执行例子（以 LLaMA-MoE 6.7B 在 NVIDIA L20 48GB 上，participant 本地数据 D_i 为例）：
    - **模型推理算法**：MoE transformer decoder。每 token 经 gating network（softmax top-k）选择 activated experts。FMD 保持完整 MoE（32 layers × 16 experts），FMQ 将所有 expert 参数量化至 INT4，FMES 只保留 top-K 高频 experts 并丢弃其余。
    - **系统框架**：parameter-server-based federated learning。Server 下发全局模型 → participant 本地 fine-tuning → 上传 expert updates → FedAvg aggregation。FMD 使用 PyTorch + 自定义 offloading logic 管理 GPU↔CPU 数据传输。FMQ 使用标准 quantization library。FMES 使用 activation frequency counting + expert filtering。
    - **编译框架**：论文未明确说明（使用 PyTorch eager mode，无额外编译优化）。
    - **Kernel 调度**：标准的 PyTorch CUDA backend。MoE FFN experts 的 GEMM 由 cuBLAS 执行，gating softmax/top-k 由 PyTorch 原生 CUDA kernel 执行。FMD 中 expert offloading 触发 cudaMemcpy 在 GPU↔CPU 之间传输 expert 参数张量（32 layers × 16 experts × ~85M params/expert for LLaMA-MoE ≈ 每次传输数 GB）。论文未明确说明 offload 策略的具体 kernel 调度细节。
    - **硬件架构**：NVIDIA L20 GPU（48GB VRAM, Ada Lovelace SM）+ host CPU + PCIe Gen4 互联。FMD 的 expert offloading 受 PCIe bandwidth 瓶颈限制（每轮 fine-tuning 需多次 CPU↔GPU expert 参数传输）。
  - Baseline 痛点：
    1. **FMD**：expert offloading 的 CPU↔GPU 数据传输延迟巨大，严重拖慢 fine-tuning 速度。PCIe bandwidth 成为瓶颈。
    2. **FMQ**：INT4 量化误差在 backprop 梯度计算中累积放大，导致训练不稳定、收敛慢甚至不收敛（Figure 10/11 中 FMQ 曲线震荡）。
    3. **FMES**：丢弃"低频" non-tuning experts 严重损害模型精度（Figure 3a：discarding non-tuning experts 导致 ROUGE 显著下降）。原因是 token 路径中 non-tuning experts 的输出错误会逐层传播累积（Figure 3b, Figure 8）。此外，仅用 activation frequency 选 expert 不准确——部分低激活频率 expert 处理的 token 具有高 attention score，对模型输出影响巨大（Figure 9）。
    4. **通用痛点**：expert activation pattern 在 training 过程中变化（Figure 6a），静态 profiling 随时间失效；expert role assignment 在 participant 异构计算资源下难以优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法（FLUX 三大模块）：
    1. **Quantization-based Local Profiling + Stale Profiling（§4）**：用 INT4 量化 MoE 模型在本地估计 expert activation frequency、data subset D_i^e 和 attention scores ā_e，替代运行全精度模型的开销。通过 stale profiling 使 profiling 与 parameter aggregation 并行，隐藏 profiling 延迟。
    2. **Adaptive Layer-Aware Expert Merging（§5）**：不丢弃 non-tuning experts，而是按层自适应分配 merge budget（浅层+高 variance 层给更多 budget），PCA+K-Means 聚类相似 expert 后按 attention×frequency 加权合并。
    3. **Dynamic Expert Role Assignment（§6）**：定义 expert utility u_i^e = |D_i^e|√(avg gradient)，用 exploration-exploitation（动态 ε）选择 tuning experts；exploration experts 用 forward-only gradient estimation 省 backprop 开销。

  - 论文方法全栈执行例子（同上 LLaMA-MoE 6.7B, L20 48GB, participant i）：
    - **模型推理算法**：MoE transformer decoder，expert 分三类：tuning（FP32 完整更新）、exploration（FP32 forward-only 梯度估计）、merged non-tuning（frozen，加权合并后的单一 expert 参与 forward）。每次 forward 仍走 gate→top-k→expert compute 流程，但 expert 数量减少为 B_i^{tune} + B_i^{non}(l)。tuning experts 的 training 使用 profiling 得到的 D_i^e（仅用流经该 expert 的数据），提升数据效率。
    - **系统框架**：parameter-server-based FL + FLUX 定制模块。Flux.moe.customized_moe(model, exps_config) 构建每层不同 expert 数的定制 MoE。Flux.moe.load_model() 从原始 checkpoint 分离加载 expert 参数和非 expert 参数。Gate re-routing 在 merging 后更新 gating mapping。Stale profiling 使 profiling 与 server aggregation 并行（Figure 7b），round time 减少 ~28.2%。FedAvg 仅聚合 tuning experts 的 updates。
    - **编译框架**：论文未明确说明（同 baseline，PyTorch eager mode）。论文提到支持集成 Adapter 和 LoRA 等 PEFT 方法。
    - **Kernel 调度**：同 baseline PyTorch CUDA backend。merged non-tuning experts 减少 expert GEMM 调用次数（每层从 16 experts 降至 B_i^{non}(l) 个 merged experts + B_i^{tune} 个 tuning experts），降低 kernel launch overhead。exploration experts 的 forward-only gradient estimation 使用小扰动（ξ ~ N(0,σ²)）加法 + 两次 forward pass 差商近似梯度，避免 backprop kernel（省去 grad GEMM 和 grad accumulation kernel）。
    - **硬件架构**：同 baseline NVIDIA L20 + PCIe。FLUX 通过减少 GPU memory 占用（合并 non-tuning experts）使 consumer-grade GPU（48GB）能 fine-tune 原本需要更大显存的 MoE 模型。round time 中 FLUX 额外开销约 5%（Figure 20）。

  - 对应解决 Baseline 缺陷：
    1. **FMD offloading 延迟巨大** → FLUX 不 offload，而是合并 non-tuning experts + 选择 tuning experts，使模型 fit 进 GPU memory，消除 CPU↔GPU 传输开销。4.75× time-to-accuracy speedup。
    2. **FMQ 量化误差导致收敛不稳定** → FLUX 仅用量化模型做 profiling（forward only），fine-tuning 本身使用 FP32，避免 backprop 中量化误差累积。profiling 估计误差约 11.01%（4-bit），不影响 fine-tuning 精度。
    3. **FMES 丢弃 non-tuning experts 损害精度** → FLUX 保留并合并 non-tuning experts（加权合并保留关键信息），output error 相比 single expert 减少 65.6%，相比 uniform layer size 减少 47.6%（GSM8K）。最终精度与 FMD（full model）接近（Table 2：FLUX ROUGE-L 0.527 vs FMD 0.528 on Dolly）。
    4. **仅用 activation frequency 选 expert 不准确** → FLUX 定义 expert utility 结合 gradient magnitude + data utilization（公式 3），merging 权重结合 attention score + activation frequency（公式 2），更准确反映 expert 重要性。Figure 17 显示 Att.+Frq. merging 比纯频率加权减少 19.2% output error。
    5. **静态 profiling 随时间失效** → stale profiling 机制：profiling 与 aggregation 并行，每轮更新 profile 但隐藏延迟（Figure 14：误差增长 <2%，round time 减少 28.2%）。
    6. **异构 participant 下 expert role assignment 难优化** → parameter server 求解全局优化问题（公式 4）+ exploration-exploitation（动态 ε，early stage 多探索，later stage 多利用），Figure 19 显示动态 ε 比固定 ε=0.3 或 0.7 更快收敛。
