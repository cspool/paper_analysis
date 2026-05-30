## MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs

- baseline方法是什么？
  **FlexGen** [42] 是 state-of-the-art 的 memory-constrained 高吞吐批量推理系统。采用 zigzag 计算顺序（逐层加载 weights → GPU 计算 → 卸载输出），支持两种调度模式：
  - **$S_4$ (GPU Attention 模式)**：KV cache 从 CPU H2D 传输到 GPU 做 attention，与 weights transfer 竞争同向 PCIe 带宽。FlexGen 倾向于小 batch size，导致 GPU compute 和 I/O 利用率不足。需要将所有请求 padding 到最大 prompt length。
  - **$S_3$ (CPU Attention 模式)**：attention 在 CPU 执行，但整层 weights 一次性 H2D 传输会长时间阻塞后续微批次的 hidden states H2D，产生大量 I/O 气泡，实际吞吐可能低于 $S_4$。
  
  FlexGen 的 policy 搜索基于 extensive offline data fitting（耗时数小时/天），固定 hardware-model-workload 映射，不考虑 bottleneck resource 随 workload 变化而变化。

  **全栈执行例子（FlexGen $S_4$ — Mixtral 8x7B on T4 16GB, MTBench）**：
  - **算法Pipeline 层**：Mixtral 8x7B 标准 MoE，每层 Top-2 expert gating + 8 expert FFNs + GQA attention。无额外算法优化（无量化/稀疏化/蒸馏）。
  - **系统框架层**：FlexGen 逐层串行执行——(1) 加载所有 experts weights CPU→GPU（一次性传输，占用 PCIe 带宽），(2) GPU 上执行 Attention（KV cache CPU→GPU H2D）+ MoE FFN，(3) 卸载中间结果 GPU→CPU。所有微批次共享同一轮 weights 加载（amortize I/O overhead）。无跨层 I/O 重叠。
  - **编译框架层**：论文未明确说明。PyTorch eager mode，无自定义编译器。
  - **kernel调度层**：GPU attention kernel（FlashInfer/PyTorch SDPA），KV cache D2H/D2D。MoE FFN 使用 PyTorch cuBLAS GEMM。CPU 端仅做 KV cache malloc/free 和 host-device synchronization。
  - **硬件架构层**：T4 GPU (16GB HBM, 65T FP16 FLOPS, 320GB/s BW) + Intel Xeon 24-core (192GB DRAM, ~200GB/s BW) + PCIe Gen3 (~16GB/s BW)。GPU 计算受限于 HBM capacity（仅能容纳少量层的 weights），大量时间消耗在 PCIe weight transfer 和 KV cache H2D 等待。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoE-Lightning 提出 **CGOPipe**（高效 GPU-CPU-I/O 流水线调度 + CPU attention + weights paging）和 **HRM**（瓶颈感知的 performance model）两大组件，系统性地解决 FlexGen 的资源利用率不足和次优 policy 问题：

  **CGOPipe 解决 I/O 调度缺陷**：
  - FlexGen $S_4$ 的 KV cache transfer 与 weight transfer 竞争 PCIe 带宽 → CGOPipe 将 decode attention 完全放在 CPU 执行（基于 HRM 分析：attention 的 operational intensity 低于 $P_1$ turning point，传 KV cache 不如在 CPU 算），仅传 hidden states（比 KV cache 小 3-4×），释放 PCIe 带宽给 weight transfer。
  - FlexGen $S_3$ 整层 weights 一次性传输阻塞 hidden states H2D → CGOPipe 的 **weights paging** 将每层 weights 分 n 页（n = 微批次数），在 PostAttn(i,j) 执行时仅传输第 j 页 weights，交错传输 hidden states H2D，消除 I/O 气泡。
  - FlexGen 缺少跨层 pipeline → CGOPipe 通过 **两步超前的 CPU attention**（Algorithm 1，PreAttn(i, j+2) 和 CPUAttn(i, j+2) 提前两个微批次），确保 GPU 始终有可执行的 PostAttn 任务，减少 GPU idle。

  **HRM 解决次优 policy 问题**：
  - FlexGen data fitting 不考虑 bottleneck 变化 → HRM 扩展 Roofline Model 到多层内存层次，明确定义 **turning points**：$P_1$（低于此前不值得跨层传输数据计算）、$P_2$（低于此前吞吐受限于 PCIe 带宽），以及 **balance point**（Eq. 11，GPU BW × $I_{GPU}$ = PCIe BW × $I_{CPU}$，此时达到资源利用均衡点）。MILP 搜索出的策略可精确匹配当前 H/W 和工作负载的瓶颈资源。
  - FlexGen 倾向小 batch size → HRM 识别 GPU memory capacity 决定 throughput 上界，指导尽可能增大 batch size（在满足 CPU memory 约束下）直到达到 balance point。这使得 MoE-Lightning 可以用更少的 CPU memory 达到更高的 throughput（Fig. 1）。

  **全栈执行例子（MoE-Lightning — Mixtral 8x7B on T4 16GB, MTBench, gen_len=128）**：
  - **算法Pipeline 层**：同 baseline（Mixtral 8x7B 标准 MoE）。论文未明确说明对算法层的修改。策略搜索中 $A_g=0, F_g=1$ 表明 attention 全在 CPU 执行、MoE FFN 全在 GPU 执行。
  - **系统框架层**：CGOPipe 执行 Algorithm 1——(1) Prologue 预热前两个微批次，(2) Main Pipeline 逐层执行 GPU PostAttn(i,j) + PreAttn(i,j+2) 与 CPU CPUAttn(i,j+2) 重叠。Weights 分 14 页（14 微批次）交错 H2D。最终 MoE-Lightning (p) 达到 30.12 tokens/s（vs FlexGen 9.5 tokens/s, 3.17×），即使 batch size 减半（504 vs 1112）吞吐仍翻倍，因为消除了 KV cache H2D 竞争和 I/O 气泡。
  - **编译框架层**：论文未明确说明。无自定义编译器 pass，依赖 PyTorch eager execution。
  - **kernel调度层**：GPU 端 PostAttn——O projection (GEMM [μ, h1×h1]) + MoE FFN（gate routing: Top-2 selection + 2 expert FFN GEMM [μ, h1×h2]×2），通过 page table 访问 paged weights。CPU 端——MKL GQA kernel (QK dot + softmax + AV weighted sum)。Weight transfer——CPU→pinned (memcpy) || Pinned→GPU (cudaMemcpyAsync)，pages 间流水线化。
  - **硬件架构层**：T4 GPU (16GB HBM) + Intel Xeon 24-core (192GB DRAM) + PCIe Gen3。CGOPipe 使四个资源并行：GPU SM (GEMM via Tensor Cores)、CPU cores (MKL attention)、PCIe bus (weight pages H2D)、CPU memory controller (KV cache R/W + weight CPU→pinned)。达到 balance point 时 GPU memory capacity 为 throughput 上界——因此添加 TP (multi-GPU) 获得 super-linear scaling（S6→S7: 2.77-3.38×）。
