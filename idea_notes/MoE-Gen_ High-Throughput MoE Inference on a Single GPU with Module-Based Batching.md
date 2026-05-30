## MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching

- baseline方法是什么？
  Baseline 是 **model-based batching**，即在整个 MoE 模型的 forward pass 中统一使用一个全局 batch size。以 Mixtral-8x7B（32 layers, 8 experts/layer, k=2）在 NVIDIA A5000 24GB + 512GB Host Memory 上的离线推理执行路径为例：
  - **算法层（MoE Routing）**：标准 top-k gating。Router 对每层 self-attention 输出计算 logits → Softmax → SelectTopK(k=2)。每个 token 分配给 2 个 expert。在 model-based batching 下，若全局 batch size=16（受限于 attention peak memory），prefill 阶段每个 expert 平均收到 $16 \times 512 \times 2 / 8 = 2048$ tokens，解码阶段每个 expert 平均收到 $16 \times 1 \times 2 / 8 = 4$ tokens。**缺陷(1)**：解码阶段 expert batch size 极小（4 tokens），远低于充分 GPU 利用所需的最小 $2^{10}$ tokens（图 3 Left），GPU FLOPs 利用率仅 0.1%（表 1）。**缺陷(2)**：batch size 受 attention 模块 peak memory 限制，而 expert 模块需要的 batch size 未被独立考虑。
  - **系统框架层**：FlexGen/DeepSpeed-Inference/MoE-Lightning。均使用 model-based batching。以 DeepSpeed-Inference 为例：将整个 MoE layer 视为 dense MLP，一个统一 batch 从 input → attention（QKV proj → self-attn → output proj）→ MoE layer（router → 逐 expert 加载权重 → expert 计算 → weighted sum）→ next layer。CPU 到 GPU 的 expert weight 传输按需触发（on-demand fetch），KV-cache 部分 offload。**缺陷(3)**：每个 forward pass 中同一 expert 可能只被少量 token 激活，但 expert weights 仍需完整传输（反复 HtoD copy），PCIe 带宽浪费严重。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch 和 CUDA kernel。
  - **kernel调度层**：论文未明确说明。FlexGen/MoE-Lightning 支持 CPU attention（使用 PyTorch/CBLAS），但未针对 MoE 解码场景优化 cache 性能。
  - **硬件架构层**：NVIDIA A5000/A6000 GPU（单卡）+ CPU DRAM（host memory），PCIe 4.0 互连。Model-based batching 在解码阶段 GPU 几乎完全 idle，等待少量的 token 完成计算后立即等待下一个 batch。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoE-GEN**，核心是 **module-based batching**——将 MoE 模型分解为 attention 和 expert 两个计算密集型模块，分别为其分配不同的微批次大小，累计 token 形成大 batch 才在 GPU 执行。以 Mixtral-8x7B 在 A5000 24GB + 512GB Host Memory 上的执行路径为例（解码阶段，$B=3640, b_a=75, b_e=150, \omega=0.6$）：
  - **算法层（MoE Routing 不变）**：Router 逻辑不变（top-k selection），但 batch 规模因 module-based batching 大幅提升。解码阶段累计 batch $B=3640$ 个 token 进入 MoE layer 时，每个 expert 平均收到 $3640 \times 2 / 8 \approx 910$ tokens（vs baseline 的 4 tokens），GPU 利用率提升至 41%（表 1）。**解决缺陷(1)**：通过多轮 attention 小批次累计，最终在 expert 模块形成大 batch，使 GPU FLOPs 接近饱和。
  - **系统框架层（MoE-GEN Engine）**：
    1. **Module-based batching**：attention 模块以 $b_a=75$ 为微批次循环执行约 $B/b_a \approx 49$ 轮。每轮：Pre-Attention → (CPU: 60% tokens self-attn via AVX kernel，GPU: 40% tokens self-attn + KV-cache HtoD copy) → Post-Attention。所有 49 轮的 output tokens 累计后在 expert 阶段一次性处理。**解决缺陷(2)**：attention batch 由 peak memory 决定（small），expert batch 由累计决定（large），两者解耦。
    2. **DAG-based scheduling + search**：将整个 layer 的执行建模为 DAG（图 6），每个节点为 computation 或 memory copy，边为依赖关系。Scheduler 在 search space（$B, b_a, b_e, \omega, S_{Expert}, S_{Params}$）中枚举候选，通过 DP 计算 critical path 选择最短执行时间的配置。**解决缺陷(3)**：通过最优 $S_{Expert}$ buffer 实现 expert weight 预取与 GPU 计算 overlap，消除 expert weight 反复传输的带宽浪费。大 batch 下 expert 顺序执行，每个 expert weight 只需加载一次处理大量 token。
    3. **Full KV-cache offloading**：KV-cache 全部保留在 host memory，GPU 仅保留当前需要的 KV-cache 窗口，减少 expert weight fetching 流量达 20×（图 4）。DtoH engine 异步更新 KV-cache。
    4. **CPU attention offloading**：60% attention 计算在 CPU 执行（$\omega=0.6$），CPU 直接访问 host memory 中的 KV-cache 无需 HtoD copy，节省的 PCIe 带宽用于 expert weight 预取。**解决缺陷(3) 的带宽瓶颈**：attention 阶段的 KV-cache HtoD 传输与 expert 阶段的 weight 预取竞争 PCIe 带宽，CPU attention 直接消除这部分竞争。
  - **编译框架层**：论文未明确说明。使用标准 C++/CUDA 编译工具链。
  - **kernel调度层（CPU Attention Kernel）**：
    - 基于 AVX intrinsics 的 Grouped Query Attention（BF16 格式），FP32 累加，每次点积累加后按 BF16 舍入规则舍入。设计类似 FlashAttention CPU 版本的 cache 优化策略。**解决缺陷**：PyTorch/CBLAS 的 CPU attention 实现未针对 MoE 解码场景的 GEMV 中等算术强度优化 cache，MoE-GEN 的 AVX kernel 使 CPU 处理 self-attention 速率达到与 PCIe4.0 传输 KV-cache + GPU 计算时间可比，使得 CPU offloading 有 net throughput gain。
  - **硬件架构层**：同 baseline（单 GPU + CPU + Host Memory, PCIe 4.0）。区别在于 module-based batching 下 GPU 利用率显著更高：expert 模块 GPU 计算与下一个 expert weight HtoD 预取完全 overlap（图 3 Right 的 $>2^{11}$ tokens 区域），GPU idle 时间降至接近零。

- baseline方法是什么？
  Baseline 为 Top-K routing with quantization + LRU caching（基于 `dvmazur/mixtral-offloading`），即在资源受限的 GPU 上运行 Mixtral-8x7B 时，部分 expert 参数 offload 到 CPU DRAM，使用量化压缩和 LRU 缓存来缓解 offloading 带来的延迟。以 Mixtral-8x7B（32 layers, 8 experts/layer, k=2）在 H100 + CPU DRAM 上的执行路径为例：
  - **算法层（MoE Routing）**：Gating network 接收 self-attention 输出 H_i，计算 Logits=H_i·W_exp → Softmax → SelectTopK(k=2)。纯粹基于模型 logits 选择 top-2 experts，完全无视 expert 的物理位置（HBM vs CPU）。**缺陷**：(1) 若 Top-2 中任一 expert 在 CPU DRAM 中，必须等待 PCIe 传输完成才能继续——图 2 显示 CPU read time 比 GPU read time 高数个数量级；(2) gating 的选择无记忆性，可能连续选中冷门 off-chip expert，导致频繁的 CPU↔GPU swap；(3) 当 offload 比例增加（更少 VRAM）时，性能劣化加剧，因为更多 expert 不在 HBM 上。
  - **系统框架层**：`dvmazur/mixtral-offloading` 框架。包含 expert 量化（压缩权重减少传输量）、LRU caching（保留最近使用的 expert 在 HBM）、expert offloading manager。每次解码步骤：gating → Top-K → 检查 expert residency → 若缺失则触发 CPU→GPU load → 可能触发 LRU eviction → expert 计算。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。CPU↔GPU 传输通过 PCIe，无自定义 kernel。
  - **硬件架构层**：NVIDIA H100 GPU + CPU DRAM host memory。GPU HBM 带宽远高于 PCIe 带宽（图 2：CPU read 延迟 >> GPU read 延迟）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 MoE-ERAS——在 gating 阶段引入 expert residency awareness，通过 thresholding 和 biasing 两种技术修改路由决策，使模型倾向选择已驻留在 HBM 的 expert，从而减少 costly 的 CPU→GPU 传输。以 Mixtral-8x7B 在 H100 + CPU DRAM（3 experts offloaded/layer）上的执行路径为例：
  - **算法层（Residency-Aware Routing）**：
    1. **Thresholding**：Weights=Softmax(Logits) 后，对 HBM 中 expert 加 α bias → SelectTopK。当 off-chip expert 与 on-chip expert 的 logits 接近时（"无绝对赢家"场景），α 使 on-chip expert 胜出。**解决 baseline 缺陷(1)**：避免为微小 logit 优势触发 costly CPU→GPU 传输。α=0.15 时在 3 experts offloaded 场景下减少 10-13% 解码延迟。
    2. **Biasing**：Logits 中 off-chip expert 减去 β(1-freq(E_i)) 惩罚 → Softmax → SelectTopK。freq(E_i) 是从 profiling（500k tokens）收集的归一化激活频率。**解决 baseline 缺陷(2)**：冷门 off-chip expert 惩罚大（避免"加载-立即换出"的双重 swap），热门 off-chip expert 惩罚小（值得加载因为后续 token 会复用）。
    3. 两者均通过超参数（α 或 β）提供 controllable speedup-quality trade-off。**解决 baseline 缺陷(3)**：offload 越多、α 越大，speedup 越显著——在极端 offload 下减少 21.2% 延迟。
  - **系统框架层**：在 `dvmazur/mixtral-offloading` 基础上增加：(1) residency table 维护模块（每层 expert 的 HBM/CPU 状态）；(2) profiling 模块（收集 expert activation frequency 用于 biasing）；(3) residency-aware routing 模块（在 gating→TopK 之间插入 thresholding/biasing 逻辑）。与 quantization、LRU caching、prefetching 正交，可叠加使用。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。方法在路由层面操作，不涉及 kernel 修改。
  - **硬件架构层**：NVIDIA H100 GPU + CPU DRAM。MoE-ERAS 的效果随硬件不对称性（GPU HBM bandwidth >> PCIe bandwidth）加剧而更显著——资源越受限，residency-aware selection 越有价值。
