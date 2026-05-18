## Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

- baseline方法是什么？
  Baseline方法分为三类：(1) **通用DNN框架**（PyTorch v2.1, PyTorch-Inductor v2.1, TensorRT v8.6）：以operator-by-operator方式执行扩散pipeline，对所有请求独立处理，batching要求完全相同的shape。全栈执行例子：算法层扩散模型denoising loop→系统框架层PyTorch eager execution或Inductor/TensorRT compile→编译框架层operator fusion / kernel selection但无数据属性感知→kernel调度层cuBLAS/cuDNN标准kernel→硬件架构层A100/H100 GPU。缺陷：(a) 无跨请求优化——大量相同prompt/input的冗余计算不被消除；(b) ragged shape请求无法batch——仅支持uniform shape batching，ragged请求需单独执行；(c) 无invariant tensor优化——loop-invariant计算在每次denoising iteration中重复执行。

  (2) **扩散专用框架**（Stable Fast v1.0, Diffusers）：手动优化pipeline针对特定uniform数据属性场景（如batch相同prompt）。全栈执行例子：算法层扩散pipeline→系统框架层Stable Fast/Diffusers提供预设优化pipeline→编译框架层manual optimized pipeline→kernel调度层标准kernel→硬件架构层A100/H100 GPU。缺陷：(a) 仅针对uniform properties预设场景优化——一旦prompt/control input/image shape等出现heterogeneity，退回通用执行；(b) 手动优化覆盖极窄——应用开发者和用户的grid search/correlative requests工作流产生大量异构数据属性，manual optimization枚举组合不可行；(c) 无ragged batching支持——不同shape请求只能串行或pad到最大shape（浪费计算和显存）。

  (3) **Katz[37]** (Diffusion Serving System)：面向ControlNet-as-a-service的多GPU serving，sequential执行每个ControlNet请求。全栈执行例子：算法层ControlNet+LoRA pipeline→系统框架层Katz multi-GPU worker→编译框架层论文未明确说明→kernel调度层标准kernel+multi-GPU通信→硬件架构层4×H100 GPU。缺陷：(a) 多GPU通信overhead（edit SDXL-Turbo单iteration场景下严重限制throughput）；(b) sequential execution无batching (latency ~0.03s/request但per-GPU throughput低)；(c) LoRA serving数学上不等价。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**Difflow (ChituDiffusion)**：一个data-characteristic-aware diffusion model serving系统，通过symbolic data property propagation将pipeline分解为dGraphs→编译为多版本dEngines→运行时DP调度dTasks batching。

  论文方法全栈执行例子（以edit应用：SDXL Turbo + ControlNet + 16 LoRA styles, A100为例）：
  - 算法层：SDXL Turbo diffusion pipeline(1 denoising step) + ControlNet(Canny edge spatial control) + 16 LoRA weights for diverse styles→固定latent noise和U-Net conditioning，每个request调用不同LoRA
  - 系统/Serving框架层：Application characteristics标注——prompts可能相同(symbolic)、input images vary、ControlNet conditioning inputs shared→Difflow compiler执行符号化属性传播→SDXL U-Net分解为3个dGraphs：dGraph1 (input-dependent U-Net layers)、dGraph2 (input-independent U-Net layers, 16 styles共享)、dGraph3 (post-processing)→每个dGraph编译为多个dEngines（redundant/ragged/uniform variants）→运行时2 unique dTasks for dGraph1, 3 for dGraph2→DP scheduler枚举dEngines寻找最优batching→data-aware batching dispatch到matched dEngine执行→asynchronous property inference与execution overlap
  - 编译框架层：Symbolic property propagation (Table 1规则覆盖elementwise/linear/convolution)→DFG unroll+stabilize denoising loop→group consecutive operators by identical output property expression→enumerate property conditions→prune conflicting+inessential(<5% speedup) cases→compile surviving conditions to specialized dEngines→Redundancy elimination rules (dimension-level computation+memory access)→Ragged operation regularization (transpose/reshape/im2col graph transforms)→Invariant tensor detection (four-state: constant/loop-invariant/loop-variant/unknown)
  - kernel调度层：Redundancy elimination——相同prompt请求的K/V tensors compress沿batch dim + concat Q tensors→FlashAttention标准kernel→broadcast恢复；Ragged batching——不同shape请求通过transpose/reshape转regular→标准Matmul/Conv kernel；Invariant tensor——dGraph2输出识别为constant→compile-time caching with multi-value support (16 LoRA outputs precomputed)
  - 硬件架构层：NVIDIA A100 40GB PCIe (CUDA 12.1) / H100 80GB PCIe

  对应解决Baseline缺陷：
  **(1) 通用DNN框架无跨请求优化 → dGraph decomposition + redundancy elimination**：Difflow通过符号化属性传播识别pipeline中的共享优化条件——dGraph decomposition将pipeline分片为共享相同优化条件的子图（如U-Net中的prompt-embedding processing、ControlNet conditioning），使跨请求冗余（如相同prompt的CLIP embedding、相同conditional inputs）被系统性地检测和消除。Redundancy elimination在dimension-level移除冗余计算和内存访问——以attention为例，相同K/V tensors compress + Q tensors concat → 一次attention计算替代多次独立计算。在refine/edit/video应用上（含correlative requests with shared inputs），A100上throughput最高提升2.1×，H100上2.2×。

  **(2) 扩散专用框架仅支持uniform properties → multi-version dEngine compilation**：Difflow通过selective dEngine generation为每个dGraph编译多个specialized engines，每个engine针对特定data property配置优化——涵盖uniform/redundant/ragged等不同property组合，通过pruning unsatisfiable+inessential conditions控制编译开销（SDXL U-Net从monolithic 16384 engines→4 engines, 从11天→4分钟）。运行时DP scheduler在选择dEngines时兼顾batch size (ragged dEngines批处理更多请求)和per-engine efficiency (uniform dEngines无ragged overhead)——如图15所示，raggedness ratio 25%-50%时混合使用两种dEngines比单一类型额外提升10%。

  **(3) ragged shape请求无法batching → ragged operation regularization**：Difflow不要求手写所有ragged operator kernel（开销大且auto kernel generators难以处理），而是通过graph transformation rules (transpose/reshape/im2col)将ragged data-sharing operations转化为regular operations——ragged dim与batch dim fuse后变为regular dim→直接使用cuBLAS/cuDNN等成熟kernel库。ragged data-independent ops通过round-robin tile-to-thread-block mapping并行执行。在venti (SD1.5) ragged-only workload上，Difflow达1.4× speedup（通过ragged batching实现并行+减少weight重复内存访问）；在grande (SDXL)上1.1× speedup（SDXL模型更大，batching gain收窄）。

  **(4) 手动优化枚举所有property组合不可行 → symbolic property propagation**：Difflow用symbolic boolean variables表示数据属性——初始由application characteristics提供（fixed/varying/symbolic）→通过operator-specific propagation rules (Table 1)传播→output property expression自动推断→operators with identical expressions grouped to dGraph→state space exponentially reduced。相比monolithic optimization需枚举2^n property combinations (n=input count)，dGraph decomposition将问题分解为多个小dGraph的独立枚举。

  **(5) loop-invariant计算重复执行 → invariant tensor elimination + multi-value caching**：Difflow的four-state detection (constant/loop-invariant/loop-variant/unknown)识别constants和loop-invariants→compile-time precompute→loop-invariants hoisted outside denoising loop→multi-value constants selective fixing (如16 LoRA weights对应16 precomputed outputs)。sequential execution (无batching)下IRE单独贡献1.3× speedup (Figure 13b)。

  **(6) Katz multi-GPU communication overhead → single-GPU data-aware execution**：Difflow在单GPU上通过dGraph decomposition+data-aware batching同时服务所有请求，避免Katz的多GPU通信overhead。在edit应用上Difflow per-GPU throughput显著高于Katz (normalized to per-GPU, 图10)。

  Trade-off：(1) 符号化分析假设数据属性可被有限symbolic variables表达——对极端irregular或不可预测的workload可能退化；(2) dEngine pruning (5% speedup threshold) 可能在edge cases下丢失marginal优化机会；(3) 性能模型虽R²=0.998但基于线性假设——对complex computation patterns (如video temporal/spatial attention)需手动调整input metrics；(4) dGraph decomposition依赖pipeline DFG结构——高度irregular control flow（如dynamic ControlNet activation per iteration）需runtime adaptation。
