## MetaAttention: A Unified and Performant Attention Framework across Hardware Backends

- baseline方法是什么？
  **手写 CUDA/Triton attention kernel（FlashAttention/FlashMLA/Mamba2 chunk kernel）+ compiler fallback（PyTorch SDPA/TorchInductor）**：现有 attention 生态中，高性能 attention kernel 均为手工优化实现，每种 attention 变体（Softmax/Sigmoid/ReLU/MLA/Mamba2/RetNet/Gated Retention 等）需要独立手写完整的 CUDA 或 Triton kernel（如 FlashMLA 1000+ 行 CUDA, Mamba2 3000+ 行 Triton）。Kernel 内部执行策略（tiling scheme, memory placement, pipeline stages, warp specialization）全部 hardcode 并针对特定 attention pattern 和特定 GPU（如 H100）优化。对于不支持的变体（如 ReLU Attention, RetNet Parallel），用户被迫 fallback 到 PyTorch native implementation——每个操作（matmul + normalization + mask）作为独立 kernel launch，中间 tensors 全部经 HBM round-trip，性能极差。Compiler-based 方案（TorchInductor, TVM, TensorRT）虽减少开发量，但无法理解 attention 的语义（如 online softmax），将 attention 视为离散 opaque 操作序列，无法生成 fused attention kernel。

  全栈执行例子（FlashAttention-3 on H100, LLAMA-3.1-8B Softmax Attention, seqlen=8K, head=32, dim=128, bf16）：
  - **模型推理算法层**：Standard scaled dot-product attention: O = softmax(QK^T/√d)V。FlashAttention-3 使用 tiled online softmax + warp-specialized asynchronous pipeline。
  - **系统框架层**：PyTorch v2.5.0 通过 SDPA API 调用 FA3 CUDA kernel。若遇到不支持变体（如 ReLU Attention），fallback 到 `torch.nn.functional.scaled_dot_product_attention` → PyTorch decomposes 为独立 cuBLAS matmul + softmax kernel + matmul → 3 次 kernel launch，中间 N×N attention matrix 经 HBM round-trip。
  - **编译框架层**：FlashAttention-3 为手写 CUDA C++ kernel，nvcc 编译。无自动 lowering pipeline——修改 attention pattern（如从 softmax 改 sigmoid）需重写 kernel。TorchInductor 无法自动生成 fused attention kernel（缺乏 online normalization 语义理解和双 GEMM fusion 能力）。
  - **kernel调度层**：FA3 kernel 内 hardcoded 调度——producer warpgroup (TMA load K/V tiles from HBM→SMEM) ∥ consumer warpgroup (wgmma QK^T → CUDA core online softmax: FMNMX + MUFU.EX2 + rowsum → wgmma PV → rescale O)。Tile size B_r/B_c 根据 head_dim=128 手写固定；若 head_dim 改为 192（DeepSeek-V2-Lite），FA3 需 padding 到 256 对齐 MMA tile，浪费 compute。Pipeline stages=2 (Pingpong scheduling)，register allocation 通过 setmaxnreg 手动分配。Strategy 硬编码：任何 attention 变体或 hardware 变更（如移植到 AMD MI250）需重写全部 execution strategy。
  - **硬件架构层**：NVIDIA H100 SXM5 (132 SMs, Tensor Core wgmma, TMA, 228KB SMEM/SM)。FA3 forward 达到 ~740 TFLOPs（~75% peak），但仅限 H100+Softmax Attention+causal mask。移植到 AMD MI250 需重写全部 kernel（ROCm Matrix Core, async copy → non-trivial porting）。

  Baseline 缺陷：
  - (a) **"Software lottery"——注意力变体性能取决于是否有手写 kernel**：支持的变体（Softmax+causal）性能优异，不支持变体（Sigmoid/ReLU/RetNet parallel）fallback 到 PyTorch native，性能差 5-10×
  - (b) **Hardcoded scheduling 不适应非标准 shapes**：FA3 固定 B_r/B_c 基于 head_dim=128 优化；Diff-Transformer-3B (dimqk=128≠dimv=256) 或 DeepSeek-V2-Lite (dimqk=192) 需 padding 对标对齐，浪费 compute 和 memory
  - (c) **Hardware lock-in——每 GPU 重新手写 kernel**：FA2 在 A100 上达 70% peak throughput，但移植到 H100 仅 30% peak；需引入 register-level pipelining 和 ping-pong kernel design 才能达到 H100 peak；移植到 AMD GPU 更困难
  - (d) **Recurrent attention（Mamba2/RetNet）缺乏 fused kernel**：手写 Triton kernel（Flash-Linear-Attention）有优化，但仍有大量 HBM intermediate traffic；无法像 FA3 那样利用 online computation + hardware-specific asynchrony
  - (e) **Compiler 无法理解 attention 语义**：TorchInductor/TVM/TensorRT 将 attention 视为 opaque 操作序列，无法实现 online softmax/sigmoid/norm 等 attention-specific fusion

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **MetaAttention：统一 attention 抽象 + 可定制模板 + 跨硬件自动调度框架**。

  对应关系：
  - (a) → **统一 abstraction + customizable templates**：将 attention 抽象为 relevance scoring (QK^T) + aggregation (PV) 两个固定操作，通过 Parallel Pattern（全局上下文，matmul-based）和 Recurrent Pattern（压缩 state，iterative update）两种模式实例化。Customizable functions（Mod 元素级变换 + RowNorm 行归一化 + RowNorm online interface）提供任意 attention 变体的表达能力——用户仅需 22-90 行 Python 定义 template+functions 即可获得高性能 kernel。支持 10+ attention variants（Softmax/Sigmoid/ReLU/MLA/Mamba2/RetNet/YOCO/RFA/Sparse GQA/Sliding Window），消除 "software lottery"。
  - (b) → **IntermediateTensor-based scheduling 自适应 shape**：不 hardcode tile size——外层 Tile Config Scheduling 枚举所有合法 output tile sizes（对齐 basetile 但不受限于固定 head_dim），通过 computation graph 自动传播 tile shape 到所有 IntermediateTensors。对 dimqk≠dimv（如 Diff-Transformer-3B dimqk=128, dimv=256），scheduler 自动选择非等长 tile sizes 避免 padding waste，在 Diff-Transformer-3B forward 实现 1.61× speedup over FA3。
  - (c) → **DeviceConfig + multi-backend runtime**：硬件约束抽象为 DeviceConfig（BaseTileShape + MemoryInfo）→ 同一套 scheduling policy 适配不同 hardware。NVIDIA backend 通过 TileLang 和 CUTE 使用 TMA + Tensor Core；AMD backend 通过 TileLang 使用 Matrix Core + async copy。MI250 上平均 3.3× forward speedup over baselines，无需 per-GPU 重写 kernel。
  - (d) → **Recurrent Pattern 统一支持**：Recurrent pattern 将 Mamba2/RetNet Recurrent 等 stateful attention 统一为 "matmul(Q, h) + h = h + matmul(K^T, V)" 的固定模板 + h_mod customizable function。Attention runtime 对 recurrent pattern 实现 chunk parallelism 技术 [32] 最大化 hardware utilization。Mamba2 forward 1.66×/backward 1.78× vs Flash-Linear-Attention。
  - (e) → **Two-layer scheduling policy 理解 attention 语义**：IntermediateTensor 建模揭示 attention computation graph 的 dataflow——tile propagation 确保 online normalization 的 tile dependency 正确传播；RowNorm online interface 将 online softmax/sigmoid/L2-norm 等标准化为统一接口供 scheduler 推理；code inlining 将 customizable functions fused 到 attention mainloop，实现与 handcrafted kernel 同等的 memory-efficient pipelining。

  全栈执行例子（MetaAttention Diff-Transformer-3B Softmax Attn, H100, seqlen=8K, dimqk=128≠dimv=256, bf16）：
  - **模型推理算法层**：用户定义 Parallel Pattern + RowNorm online softmax (scores_RowNorm_Online) + scores_Mod (causal mask) + Q_mod (scale by 1/√d)，约 87 行 Python。
  - **系统框架层**：用户调用 MetaAttention Python API → 生成 scheduling plan → attention runtime 生成 kernel → 替换 PyTorch Transformers 中的 attention 调用。无 PyTorch decomposition，单次 kernel launch 完成全部 attention computation。
  - **编译框架层**：
    1. Customizable Function Lowering: trace scores_Mod (Mul with mask) + scores_RowNorm_Online (ReduceMax+Exp+ReduceSum+Div chain) → elementwise + row-reduce DAG → hardware-mapped code snippets
    2. Scheduling Space: IntermediateTensors = {Q, K, scores, weights, V, output} + customizable function internal tensors
    3. Tile Config Scheduling: Enumerate output tiles for dimv=256（vs FA3 固定 pad 到 256 对齐 dimqk=128→256 padding）→ 可自然支持 dimqk≠dimv 的 tile shapes
    4. Tile Resource Scheduling: 分配 memory——Q(128×128×2B=32KB)→SMEM, K(128×128×2B=32KB)→SMEM, scores(128×128×2B=32KB)→SMEM, V(128×256×2B=64KB)→SMEM, weights(128×128×2B=32KB)→RF, output(128×256×2B=64KB)→RF accum then SMEM；检查 SMEM total=32+32+32+64+64=224KB ≤ 228KB → valid
    5. Profiling 选最优 plan
  - **kernel调度层**：Attention runtime 根据 plan: TMA async load Q tile → SMEM, TMA load K,V tiles → SMEM (pipeline stage=2: prefetch next tile while computing current) → wgmma QK^T [128,128]×[128,128]^T → CUDA core online softmax (ReduceMax + MUFU.EX2 + ReduceSum + rescale, all in RF) → wgmma PV [128,128]×[128,256] → rescale output → TMA store。Customizable functions (mask, scale) inline 到 CUDA core region，零额外 launch overhead。
  - **硬件架构层**：NVIDIA H100 SXM5 (TMA + wgmma + async pipeline)。与 FA3 不同：tile size 自适应 dimqk=128/dimv=256 无需 padding，tiling scheme 由 scheduler 而非手写决定。同时同一套 template 可通过 DeviceConfig(MI250) 移植到 AMD MI250——BaseTileShape 适应 Matrix Core (64×64)，MemoryInfo 适应 MI250 hierarchy，无需改任何 Python 代码。

  关键设计选择映射到 baseline 缺陷：
  - attention variants 不统一 → relevance scoring + aggregation abstraction，两种 pattern 覆盖全部变体
  - hardcoded scheduling → IntermediateTensor + DeviceConfig + two-layer scheduling policy 自动化
  - hardware lock-in → DeviceConfig 抽象 + TileLang/CUTE multi-backend mapping
  - compiler 不理解 attention → RowNorm online interface 标准化 online normalization，scheduler 传播 tile dependency
  - 开发成本高 → 22-90 LoC Python vs 400-3000 LoC CUDA/Triton
