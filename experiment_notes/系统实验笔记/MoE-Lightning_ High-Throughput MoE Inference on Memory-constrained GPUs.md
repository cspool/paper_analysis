## MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-Lightning 是一个面向 GPU 内存受限场景的高吞吐 MoE 批量推理系统，核心包含四个组件：
    1. **CGOPipe（GPU-CPU-I/O Pipeline Schedule）**：一种细粒度的流水线调度策略，将 GPU 计算（QKV projection + O projection + MoE FFN）、CPU 计算（attention softmax）以及四种 I/O 事件（D1: QKV DtoH、D2: Hidden H2D、D3: Weights Transfer、D4: KV cache Transfer）高效重叠。核心创新包括：
       - **CPU Attention**：基于 HRM 分析将 decode 阶段 attention 放在 CPU 执行，仅传输 hidden states（远小于 KV cache），释放 PCIe 带宽给 weight transfer。CPU attention 比 KV cache transfer 快 3-4×。
       - **Weights Paging**：将每层 weights 分 n 页（n = 微批次数），在微批次间交错传输 hidden states H2D 与 next-layer weight pages。每个 expert FFN kernel 通过 page table 访问对应 GPU 上的 weight pages。
       - **双缓冲 Weight Buffer**：分配 2 × sizeof(per-layer-weights-on-CPU) 的 GPU buffer，重叠当前层计算与下一层 weight 预取（CPU→pinned→GPU 两阶段流水线）。
       - **两步超前的 CPU attention**：Algorithm 1 中 GPU PreAttn(i, j+2) 和 CPU Attention(i, j+2) 比当前 PostAttn(i, j) 提前两个微批次，确保 GPU 不被 CPU attention 阻塞。
    2. **HRM (Hierarchical Roofline Model)**：扩展自经典 Roofline Model 的多层内存层次性能模型。引入跨层内存带宽屋顶 $B_{peak}^{j,i} \times I_x^j$ (Eq. 6) 和多个 compute roof ($P_{peak}^i$, $P_{peak}^j$)。定义 turning points $P_1$ (Eq. 9, 低于此则不值得将数据从 CPU 传到 GPU 计算) 和 $P_2$ (Eq. 10, 低于此则受限于 CPU→GPU 带宽)，以及 balance point (Eq. 11, 此时 GPU memory bandwidth × $I_{GPU}$ = CPU→GPU bandwidth × $I_{CPU}$)。基于 HRM 构建性能模型 $T = \max(comm^{cpu\_to\_gpu}, T_{cpu}, T_{gpu})$，以 MILP 搜索最优 6 元组策略 $\mathcal{P} = (N, \mu, A_g, F_g, r_w, r_c)$。
    3. **Dynamic Request Batching (Algorithm 2)**：按 input length 降序排列请求，贪婪地将最长请求分配到当前 token 数最少的微批次，使各微批次大小接近目标 μ，支持 variable-length prompt（无需 padding）。
    4. **Tensor Parallelism**：单节点内 TP 支持，各 GPU 持有权重分片，使用 all-reduce 聚合。TP 下 GPU memory capacity 和 bandwidth 随 tp_size 线性增长，HRM 搜索策略与单 GPU 相同。
  - 实验比较：(1) MoE-Lightning (p) vs FlexGen / FlexGen(c) / DeepSpeed Zero-Inference 的 generation throughput (tokens/sec)；(2) 不同 MoE 模型（Mixtral 8x7B、Mixtral 8x22B、DBRX 132B/16E）下的吞吐对比；(3) 不同 GPU 配置（S1: 1xT4 16G, S2: 1xL4 24G, S6: 2xT4 32G, S7: 4xT4 64G, S8: 2xT4+DBRX, S9: 4xT4+DBRX）下的对比；(4) 不同 workload（MTBench、HELM synthetic reasoning、HELM summarization）和 generation length（32/64/128/256 tokens）下吞吐变化；(5) Tensor Parallelism scaling（2→4 T4，Mixtral 8x22B 2.77-3.38×、DBRX 2.1-2.8× 加速比）；(6) Ablation：Optimizer policy 对比、CPU attention vs KV transfer vs MoE FFN latency、不同硬件配置下最优 policy 变化。

- 硬件平台是什么，配置是什么。
  - 六种 Setting (Table 2)：
    - S1: Mixtral 8x7B + 1xT4 (16GB HBM) + Intel Xeon 2.30GHz 24-core, 192GB DRAM
    - S2: Mixtral 8x7B + 1xL4 (24GB HBM) + Intel Xeon 2.20GHz 24-core, 192GB DRAM
    - S6: Mixtral 8x22B + 2xT4 (32GB total) + Intel Xeon 2.30GHz 32-core, 416GB DRAM
    - S7: Mixtral 8x22B + 4xT4 (64GB total) + Intel Xeon 2.30GHz 32-core, 416GB DRAM
    - S8: DBRX (132B, 16E) + 2xT4 (32GB total) + Intel Xeon 2.30GHz 32-core, 416GB DRAM
    - S9: DBRX (132B, 16E) + 4xT4 (64GB total) + Intel Xeon 2.30GHz 32-core, 416GB DRAM
  - GPU 互联：PCIe（单节点内多 GPU），无 NVLink。
  - Ablation study extra: 2xA100-80GB + 变化的 CPU:GPU bandwidth (100-500 GB/s)。

- 开源Serving框架是什么。修改了什么。
  - 开源 Serving 框架：基于 **vLLM** [26] (https://github.com/vllm-project/vllm) 和 **SGLang** [56] (https://github.com/sgl-project/sglang)，用 Python 和 C++ 编写。
  - 代码仓库：https://github.com/caoshiyi/artifacts/tree/asplos25（ASPLOS 2025 artifact）
  - 主要修改/新增：
    1. **Pipeline Scheduler (CGOPipe)**：在原有 zigzag execution（FlexGen 风格）基础上实现了 Algorithm 1 的 Prologue→Main Pipeline 调度逻辑——GPU PostAttn + PreAttn 与 CPU Attention 两步超前、paged weights 交错传输。
    2. **CPU GQA Kernel**：基于 Intel MKL 实现自定义 CPU Grouped Query Attention kernel（§6.2），在 CPU 端执行 attention softmax + weighted sum，替代 GPU attention + KV cache H2D 路径。
    3. **Paged Weight Manager**：实现双缓冲 weight buffer（2 × sizeof(per-layer CPU weights)）+ 页表查找 + CPU→pinned→GPU 两阶段异步传输。
    4. **HRM Policy Optimizer**：基于 MILP 的 offline policy 搜索器，输入 H/W spec、model spec、workload params，输出最优 (N, μ, A_g, F_g, r_w, r_c)。
    5. **Variable-Length Batcher**：Algorithm 2 按 input length 降序贪婪分配，无需 padding。
    6. **Tensor Parallelism 支持**：在单节点内实现 Megatron-style TP，权重分片 + all-reduce。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源**：代码公开在 https://github.com/caoshiyi/artifacts/tree/asplos25 。
  - **全流程使用例子（Mixtral 8x7B on 1xT4, MTBench workload）**：
    1. **Offline Policy Search**：HRM 性能模型输入——硬件 H（T4: GPU FLOPS=65T FP16, GPU BW=320GB/s, CPU→GPU BW=16GB/s, GPU memory=16GB, CPU memory=192GB）、模型 M（Mixtral 8x7B: 32 layers, h1=4096, h2=14336, n_e=8, k=2）、workload W（MTBench avg prompt=77, gen_len=128）。MILP 求解 min T 输出策略 (N=504, μ=36, A_g=0, F_g=1, r_w=0, r_c=0)。耗时 < 1 分钟。
    2. **Request Batching**：80 MTBench 问题被复制为数千请求。按 input length 降序排列，贪婪分配到 n_ub = N/μ = 14 个微批次，每微批次最多 μ=36 个请求。若某微批次 KV cache 总量超限，对应请求中止并加入下一批。
    3. **Prefill Stage（全 GPU）**：逐微批次在 GPU 上执行——加载该层所有 experts weights → QKV projection (GPU GEMM) → Flash Attention → O projection → MoE FFN（gate routing + Top-2 expert FFN）→ 输出 KV cache offload 到 CPU pinned memory。预填充为 compute-bound。
    4. **Decode Stage（CGOPipe）**：执行 Algorithm 1。
       - Prologue (i=1, j=1,2)：GPU 做 PreAttn(1,1/2) = LayerNorm + QKV proj → Offload QKV DtoH → CPU Attention(1, 1/2) via MKL GQA kernel → Hidden states H2D → PostAttn(1, 1/2) = O proj + MoE FFN（访问 paged weights 通过 page table）。
       - Main Pipeline (i=2..32, j=1..14)：对 layer i 和 micro-batch j，GPU 执行 LoadH(i,j) (H2D hidden states)、W_PINtoGPU(i+1,j) (从 pinned memory copy weights page j)、PostAttn(i,j) (O proj + MoE FFN)、PreAttn(i,j+2) (提前两步)。CPU 同时执行 CPUAttn(i,j+2) 和 W_CPUtoPIN(i+1,j+2) (weights CPU→pinned)。I/O 调度：D1(QKV DtoH) 和 D2+D3+D4(H2D) 方向相反可并行；D2/D3/D4 同向则通过 paging 交错执行。
    5. **输出**：每个 decode step 生成一个 token，追加到各请求序列。全部 gen_len=128 完成后返回完整输出。Throughput = total_tokens / (T_prefill + T_decode)，MoE-Lightning 达到 30.12 tokens/s（vs FlexGen 9.5 tokens/s，3.17× 提升）。
