## Mixture of Lookup Experts

- baseline方法是什么？
  MoE（Mixture-of-Experts）是 baseline，以 Mixtral 风格 MoE（top-2 routed experts，无共享 expert）为例说明全栈执行路径：
  - **算法层（MoE 推理）**：每层 Decoder 执行 Self-Attention → 中间特征 h → Router 计算 ArgTopK({h·r_j}) 选择 top-K expert → 加载对应 expert FFN 权重 → 计算 h' = Σ g_j·FFN_j(h) + h。每个 expert FFN 需要 h 作为输入执行 Standard MLP 计算（通常含 SwiGLU 激活）。Router 和 expert FFN 的输入都是中间特征（含上下文信息）。
  - **系统框架层**：HuggingFace Transformers（PyTorch）标准推理 pipeline。MoE 模型总参数大（如 Mixtral 8×7B: 46B 参数，仅 13B 激活），需多 GPU 或 expert offloading。
  - **编译框架层**：论文未明确说明（标准 PyTorch CUDA kernel）。
  - **kernel 调度层**：标准 cuBLAS GEMM 计算 expert FFN（W1·h, SiLU, W2·h 等）。若使用 expert offloading，需 GPU→CPU/disk 间 PCIe 传输完整 expert 权重（~176M/expert for Mixtral 8×7B）。
  - **硬件架构层**：NVIDIA V100 GPU（PCIe 4.0×16, 16 GB/s 带宽），expert offloading 到 CPU RAM 或 disk。
  - Baseline 核心缺陷：
    1. **VRAM 占用大**：虽然每 token 仅激活 top-K expert，但所有 expert 权重必须常驻 VRAM（因 Router 动态选择、无法预知哪个 expert 被激活）。Mixtral 8×7B 需 ≥92 GB VRAM (FP16)，单卡 80GB 无法容纳。
    2. **Expert offloading 延迟高**：将 expert 权重 offload 到 CPU RAM/disk → 每 inference step 加载 k 个 activated experts 到 VRAM。Mixtral 8×7B 中每 expert ~176M，k=2 时需传输 ~22.6B 参数/step（k=2 含 layer 数 32）。PCIe 4.0×16 下传输延迟 ~0.7s/step，disk 下 >10s/step，不可接受。
    3. **Batch generation 不友好**：不同样本在同一 step 可能选择不同 experts → batch size >1 时需加载所有被选中 experts（可能等于全部 N 个 experts），VRAM 使用量和通信延迟同步增加。
    4. **动态 routing 的不可预测性**：因 expert 选择由 Router 根据中间特征动态决定，prefetching 无法准确预测，CPU-GPU 通信无法被计算隐藏。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoLE 通过"embedding tokens 替代中间特征作为 expert 输入 + 全激活训练 + 推理前重参数化为 LUT"三个关键设计解决上述缺陷。全栈执行路径（以 MoLE-16E, 1B 激活参数为例）：

  - **算法层 — 训练阶段结构修改**：
    1. Routed experts 的输入从中间特征 h 改为 embedding tokens e = Embedding(input_ids)。因为 e 仅由 input_ids 决定，输入空间从无穷连续空间收缩为有限离散集 |V|（vocabulary size = 50k）。
    2. 所有 N 个 routed experts **同时激活**（不做 top-K 稀疏选择）。Router 输出 SoftMax({h·r_j}) 为全 N 维向量，对所有 expert 输出加权求和。共享 expert FFN_shared 保持标准中间特征输入 → SwiGLU 计算。
    3. 仅使用 LM cross-entropy loss 训练，无需 auxiliary loss（z-loss / load balance loss），因所有 experts 始终激活、梯度全程可微、无 collapse 风险。
    4. 前向：h' = Σ_{j=1}^N g_j·FFN_j(e) + FFN_shared(h) + h，其中 g = SoftMax(Router(h)), e = Embedding(i)。

  - **算法层 — 推理前重参数化（LUT 预计算）**：
    1. 以 embedding layer 权重 W_emb ∈ R^{|V|×d} 为输入，对每个 expert FFN_j 做一次 forward pass：v_j = FFN_j(W_emb) ∈ R^{|V|×d}。得到 |V| 个 token 对应的 expert 输出。
    2. LUT_l = {{v_j^i}_{j=1..N}}_{i=1..|V|}，大小为 N × |V| × d。
    3. LUT 整体 offload 到 CPU RAM/disk。与 MoE expert offloading 不同，LUT offload 不参与计算，仅存储。

  - **算法层 — 推理阶段（zero-computation experts）**：
    1. Lookup：根据 input_ids 从 offloaded LUT 检索 v_j^i（仅加载当前 batch 的 token 对应输出），传输量 = dN per token（与 |V| 无关）。
    2. Router 计算（同 MoE）：g = SoftMax(Router(h))。
    3. Expert 组合（无计算）：h' = Σ_j g_j·v_j^i + FFN_shared(h) + h。routed experts 仅需一次 lookup + weighted sum，零 FLOPs。
    4. Per-token 加载参数量：dN（如 1B MoLE-4E: d=2048, N=4 → 8KB），vs MoE expert offloading 的 2dkD_r（~537MB），小 60000× 以上。

  - **系统框架层**：HuggingFace Transformers + PyTorch。推理时 LUT 存储在 CPU/disk，通过 PCIe 传输 lookup results（dN 级别，<KB 量级）。共享 expert 和 attention 权重常驻 VRAM。

  - **编译框架层**：论文未明确说明（标准 PyTorch CUDA kernel）。

  - **kernel 调度层**：论文未明确说明。推理时 routed experts 无 compute kernel（仅 lookup + 加权求和），共享 expert 使用标准 cuBLAS GEMM。

  - **硬件架构层**：NVIDIA V100 GPU。CPU RAM 或 disk 作为 LUT 存储设备，通过 PCIe 4.0×16 传输。LUT 存储开销（7.4B 参数 for MoLE-16E 160M）虽比 MoE offloaded experts（1.0B）大 2.4-7.4×，但存储设备可扩展，且随模型增大（1B 激活参数），LUT/Expert 比例下降至可比较水平。

  - 对比 baseline 的改进映射：
    - **VRAM 占用大 → LUT offloading + 计算-free experts**：MoE 需常驻所有 expert 权重于 VRAM → MoLE 的 LUT 整体 offload 到 CPU/disk，VRAM 仅保留共享 expert + attention 权重。VRAM 使用等同于同激活参数量的 dense model。
    - **Expert offloading 延迟高 → Per-token 仅加载 LUT lookup results (dN)**：MoE 每 step 需加载 2dkD_r 完整 expert 参数（数十 MB 至数百 MB）→ MoLE 每 step 仅加载 dN lookup results（KB 级别），延迟可忽略（Figure 3 验证 MoLE latency ≈ Dense model latency）。通信开销降低 1000-2000×。
    - **Batch generation 不友好 → LUT offloading 通信量与 batch size 天然友好**：MoE 不同样本可能选择不同 experts → batch 增大时加载全部 expert → MoLE 加载的 LUT lookup results 是 per-token 的 dN，batch 增大仅线性增加 KB 级传输量，通信量仍可忽略。
    - **Dynamic routing 不可预测 → routing 与 LUT 解耦**：Router 仍动态运行在中间特征上（含上下文），但 expert output 已预计算为 LUT。router 只需输出 g_j 权重，LUT 存储所有可能 v_j^i，两者独立——避免了 prefetching 的预测难度。
    - **Router collapse 需 auxiliary loss → 全激活训练无需 auxiliary loss**：MoE 的 top-K sparsity 导致 router 需要 load balance loss 和 z-loss 防止 collapse → MoLE 所有 experts 始终激活并接收梯度，天然避免了 collapse。Ablation（Table 4）验证添加 auxiliary loss 反而降低性能。
    - **Embedding as input 的性能损失 → 全激活 + 更多 experts 补偿**：将 expert 输入从中间特征改为 embedding tokens 仅带来 0.7 point 性能下降（Table 7, 160M: 41.5 → 40.8），但全激活带来 1.5 point 提升（40.3 → 41.8），净收益 +0.5 point。更多 experts（N=16 vs 4）持续提升性能（Table 6: 39.7 → 42.3），证明了可扩展性。
    - **实验结果**：同 FLOPs 和 VRAM 下，MoLE 性能与 MoE 可比（MoLE-4E 1B: AVG 47.4 vs MoE-10E 1B: 46.6），推理速度与 dense model 相当，比 MoE expert offloading 快 1000× 以上。
