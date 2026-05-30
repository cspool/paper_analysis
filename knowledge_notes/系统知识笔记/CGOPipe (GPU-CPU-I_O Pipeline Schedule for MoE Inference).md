## CGOPipe (GPU-CPU-I/O Pipeline Schedule for MoE Inference)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CGOPipe 是 MoE-Lightning 提出的 GPU-CPU-I/O 细粒度流水线调度策略，专为 GPU 内存受限场景下 MoE 模型的 decode 阶段设计。名称代表 Computation-GPU-CPU-I/O Pipeline，核心思想是将 transformer layer 内的计算拆分为 GPU 计算（QKV projection + O projection + MoE FFN）、CPU 计算（attention softmax）和四种 I/O 事件（D1: QKV DtoH、D2: Hidden H2D、D3: Weights H2D、D4: KV Cache H2D），通过精确的时间交错实现高效重叠。关键机制：(1) CPU Attention——基于 HRM 分析将 attention 放在 CPU 执行（比 KV cache H2D 到 GPU 快 3-4×），仅传 hidden states H2D；(2) Weights Paging——将每层 weights 分 n 页（n = 微批次数），在微批次间交错传输 hidden states 和 next-layer weight pages；(3) 两步超前（Two-step ahead）——PreAttn(i, j+2) 和 CPUAttn(i, j+2) 比当前 PostAttn(i, j) 提前两个微批次执行，确保 GPU 不会 idle 等待 CPU attention 结果。CGOPipe 的伪代码见 Algorithm 1。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
CGOPipe 的执行流程（Algorithm 1，以 Mixtral 8x7B on T4 为例）：
1. **Prologue (j=1,2)**：对 layer i=1，GPU 执行 PreAttn(1,j) = LayerNorm + QKV projection [μ, h1] × W_qkv → Offload QKV DtoH → CPU Attention(1,j) via MKL GQA kernel → Hidden states H2D → GPU PostAttn(1,j) = O projection + MoE FFN (gate routing + Top-2 expert GEMM via page-table lookup)。同时后台 W_CPUtoPIN(2,j)（将 layer 2 weights 从 CPU 拷贝到 pinned memory）。
2. **Main Pipeline (i=2..32, j=1..14)**：对 layer i 和 micro-batch j，并行执行四条路径：
   - GPU: LoadH(i,j) (H2D hidden states from CPU attention) → W_PINtoGPU(i+1,j) (pinned→GPU copy of weight page j) → PostAttn(i,j) (O proj + MoE FFN) → PreAttn(i, j+2) (QKV proj 提前两步)
   - CPU: CPUAttn(i, j+2) (MKL GQA attention for batch j+2) + W_CPUtoPIN(i+1, j+2)
   - I/O directions: D1(QKV DtoH) 与 D2+D3+D4(H2D) 方向相反可并行；D2/D3/D4 同向通过 paging 交错执行
3. **资源利用**：CGOPipe 使 GPU SM (GEMM via Tensor Cores)、CPU cores (MKL attention)、PCIe bus (weight H2D)、CPU mem controller (KV cache R/W + weight CPU→pinned) 四路并行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：基于 vLLM/SGLang，新增 PyTorch + C++ 模块。(1) Pipeline Scheduler（Algorithm 1 的 C++ 实现，所有任务异步执行，通过 synchronization primitives 维护数据依赖）；(2) 双缓冲 weight buffer 大小 = 2 × sizeof(per-layer-weights-on-CPU)；(3) Paged weight transfer：CPU→pinned (memcpy) || pinned→GPU (cudaMemcpyAsync) 两阶段流水线；(4) CPU GQA kernel 基于 Intel MKL SGEMM。
- 使用：CGOPipe 仅在 decode 阶段启用（prefill 为 compute-bound，全 GPU 执行）。通过 HRM policy optimizer 在离线阶段搜索最优超参数 (N, μ, A_g=0, F_g=1, r_w, r_c)。
- 性能：CGOPipe + HRM 使 MoE-Lightning 在 Mixtral 8x7B on T4 上达到 30.12 tokens/s（vs FlexGen 9.5 tokens/s，3.17× 提升），且 batch size 减半（504 vs 1112）。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
