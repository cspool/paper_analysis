## Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

- baseline方法是什么？
  - **Baseline 1: Offloading-based 方法（DeepSpeed-MII ZeRO-Infinity, Mixtral-Offloading）**：将全部或大部分 expert 权重存储在 CPU memory 中，推理时按需通过 PCIe 将所需 expert 权重从 CPU memory 拷贝到 GPU memory 后执行 GPU 计算。DeepSpeed-MII 使用 ZeRO-Infinity 将模型参数 offload 到 CPU 并动态加载；Mixtral-Offloading 使用 per-layer 的 offload_per_layer 参数控制每层多少 expert 放在 CPU。
  - **Baseline 2: CPU-based 方法（llama.cpp）**：将部分层放在 GPU 执行，其余放在 CPU 执行（通过 ngl 参数控制 GPU 层数）。CPU 部分直接在 CPU memory 中计算，避免 CPU↔GPU 数据传输开销，但不考虑 MoE 的 expert 级别稀疏性和 CPU/GPU 的不同 batching 效应。
  - **全栈执行例子（以 Mixtral-8x7B 16-bit, Environment 1 Quadro RTX 6000 24GB, single-batch decode 为例）**：
    - **模型推理算法层**：Mixtral-8x7B, 32 layers, 每层 8 experts (top-2 routing), 16-bit precision, >90GB 参数总量
    - **系统框架层**：
      - Offloading-based（DeepSpeed-MII）：每 token 每层 → gate 选 top-2 expert → 检查 expert 权重是否在 GPU → 若不在，CPU→GPU PCIe 传输 ~300MB/expert → GPU GEMM 计算 → 下一层。**瓶颈**：每次 expert cache miss 需要 PCIe 传 300MB 权重（2-5× GPU 计算时间），MoE 的 2/8 expert 稀疏性未被充分利用来减少传输。
      - CPU-based（llama.cpp ngl=8）：前 8 层 GPU 执行，后 24 层 CPU 执行。GPU 层直接在 GPU 端计算；CPU 层无需 PCIe 传输（权重已在 CPU RAM）。**瓶颈**：长 prefill (1024+ tokens) 时 CPU 计算延迟随 token 数线性增长，成为严重瓶颈；beam search 时多 beam 并行放大 CPU 计算量，性能崩溃。
    - **编译框架层**：论文未明确说明。PyTorch eager mode。
    - **kernel调度层**：GPU 端使用 PyTorch CUDA GEMM kernel；CPU 端使用 PyTorch 默认 CPU GEMM（无 AVX512_BF16 优化）。Offloading-based 方法的执行顺序：PCIe copy → GPU GEMM（串行，无法重叠）。llama.cpp 的执行顺序：CPU GEMM（串行，按层执行）。
    - **硬件架构层**：NVIDIA Quadro RTX 6000 24GB + Intel Xeon Gold 6126 48-core, PCIe Gen3 x16 32GB/s。
  - **Baseline 痛点**：
    1. **Offloading 的 PCIe 传输开销**（核心痛点）：每次 expert cache miss 需通过 PCIe 传输 ~300MB 的 expert 权重（3 个 4096×14336 矩阵），传输延迟是 GPU 计算的 2-5×。在 single-batch decode 场景（s=1），传输开销占主导，offloading 方法延迟显著高于纯 CPU 方法。
    2. **CPU-based 方法忽略 batching 效应**：llama.cpp 不区分 CPU 和 GPU 的不同 batching 行为——GPU 延迟近乎恒定（受限于内存带宽），CPU 延迟随输入量线性增长（受限于计算能力）。在长 prefill（s>512）和 beam search（多 beam 并行）场景，CPU 计算成为不可接受的瓶颈。
    3. **静态执行策略无适应性**：Offloading 方法总是 GPU 执行（传输+计算），CPU 方法总是按层静态分配 CPU/GPU。两者都不根据实际输入量 s 动态选择最优策略——对小 s 应避免 PCIe 传输用 CPU 计算，对大 s 应忍受传输换 GPU 加速。
    4. **Expert 放置不考虑访问频率**：llama.cpp 按层连续分配 GPU/CPU（前 ngl 层放 GPU），而非按 expert 热门度选择性地将热门 expert 放 GPU。这导致 GPU memory 中放置了冷门 expert，而热门 expert 反而在 CPU memory。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Fiddler 方法**：通过三类技术协同解决 baseline 的痛点：
    1. **动态 Per-Expert 执行策略（Algorithm 1）**（解决痛点 1、3）：每个 expert 在运行时根据输入 token 数 s 独立决定 CPU 还是 GPU 执行。决策基于 latency model——`cpu_lat(s) ∝ s`（CPU 线性增长），`gpu_lat(s) ≈ constant`（GPU 恒定），`trans_lat()`（PCIe 传输恒定）。当 `cpu_lat(s) < gpu_lat(s) + trans_lat()` 时选 CPU 执行（小 s，避免 PCIe 传输）；否则选 GPU 执行（大 s，忍受传输换加速）。关键洞察：对 small s（decode 阶段 s=1），CPU execution strategy (c) 的 activation 拷贝量（s×4096 floats）远小于 weight 拷贝量（3×4096×14336 floats），因此 PCIe 开销可以忽略。
    2. **Expert 热门度导向的 GPU 放置**（解决痛点 4）：离线用 calibration data（ShareGPT）profiling 各 expert 激活频率，initialization 时按热门度降序放置 expert 到 GPU（尽可能多，不超过显存）。相比随机放置提升 hit rate 约 3-5 个百分点（Env1: 25.2% vs 21.9%, Env2: 53.0% vs 48.8%）。
    3. **CPU AVX512_BF16 Expert Kernel**（强化策略 c）：利用 Intel AVX512_BF16 的 VDPBF16PS 指令（每周期 32 BF16 MAC）加速 CPU 端 expert FFN 计算，弥补 PyTorch 默认 CPU GEMM 无法利用 BF16 硬件加速的不足。
  - **全栈执行例子（Fiddler, 与 baseline 同配置）**：
    - **模型推理算法层**：与 baseline 相同（Mixtral-8x7B, top-2 routing, 16-bit），不修改模型架构或 router 逻辑。
    - **系统框架层**：基于 PyTorch 自建调度系统。初始化阶段：non-expert 层常驻 GPU → 按热门度排序填充 GPU expert → 其余 expert 放 CPU pinned memory → 校准 latency model 参数。执行阶段：每层 gate 输出后统计各 expert 输入量 s → Algorithm 1 决策 → 对每个 expert 独立执行 Strategy (a)/(b)/(c) → 聚合加权输出。对比 baseline 的关键差异——Offloading 方法缺少 strategy (c)，llama.cpp 缺少 strategy (b) 和动态决策。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + 自定义 C++ CPU kernel。
    - **kernel调度层**：
      - GPU 端：与 baseline 相同（PyTorch CUDA GEMM），但通过 latency model 避免不必要的小 batch GPU 执行+传输。
      - CPU 端：自定义 AVX512_BF16 kernel 替代 PyTorch 默认 CPU GEMM，tile 分块最小化 cache miss。
      - 数据传输：Strategy (b) 使用 cudaMemcpyAsync 异步 CPU→GPU 传权重；Strategy (c) 使用 cudaMemcpyAsync GPU→CPU 传 activation（可忽略 <1%）。
      - 调度时序（以 layer l 含热门/冷门 expert 混合为例）：
        ```
        Time →
        Expert 0 (GPU hit):     |== GPU GEMM ==|
        Expert 3 (GPU miss, s大): |== PCIe W copy ==|== GPU GEMM ==|
        Expert 5 (GPU miss, s小): |== PCIe A copy ==|== CPU AVX512 GEMM ==|== PCIe out copy ==|
        ```
    - **硬件架构层**：与 baseline 相同（Quadro RTX 6000 24GB + Xeon Gold 6126 / RTX 6000 Ada 48GB + Xeon Platinum 8480+）。
    
    **关键性能对比**：
    | Scenario | Fiddler vs Best Baseline | 核心获益来源 |
    |----------|--------------------------|-------------|
    | Single batch (avg) | 1.26× vs llama.cpp | 小 s 时 strategy (c) 避免 offloading 的 PCIe 传输 |
    | Long prefill TTFT (avg) | 1.30× vs DeepSpeed-MII | 大 s 时 strategy (b) 利用 GPU 并行能力 |
    | Beam search (avg) | 11.57× vs llama.cpp | 多 beam 放大 s，strategy (b) 避免 CPU 线性增长灾难 |
    | Phi-3.5-MoE | 6.5× vs DeepSpeed-MII | 验证方法跨模型通用性 |

    **核心设计洞察**：Fiddler 的本质是将 MoE 推理的"what to offload/where to compute"问题建模为一个极小开销的 per-expert latency-driven 决策问题。其 design philosophy 是"不预设 CPU 或 GPU 哪个更好，而是让运行时数据（输入量 s）驱动决策"。这使 Fiddler 能同时覆盖两种 baseline 各自擅长的场景（offloading 擅长大 batch prefill，CPU 擅长小 batch decode），实现 Pareto 改进而非 trade-off。MoE 的 expert 稀疏性（每 token 仅 2/8 expert 需要计算）是使此策略可行的关键前提——每个 expert 的输入量 s 通常在 0 到数百之间，跨度极大，正是这种方差使得动态决策有意义。
